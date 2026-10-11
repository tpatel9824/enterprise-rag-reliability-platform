from dataclasses import replace

from acmeflow_rag.evidence import EvidenceAssessor, EvidencePolicy
from acmeflow_rag.grounding import GeneratedAnswer, GroundedClaim, validate_citations
from acmeflow_rag.models import Chunk, RetrievalResult
from acmeflow_rag.service import RagService


def result(chunk_id="chunk-1", document_id="doc-1", text="Production access requires manager approval.", score=0.8, version="1"):
    return RetrievalResult(
        Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title="Access Policy 2026",
            department="Security",
            version=version,
            effective_date="2026-01-01",
            access_roles=("security",),
            section="Approval",
            tags=("access",),
            text=text,
            heading_path=("Access Policy", "Approval"),
        ),
        score,
    )


class StaticRetriever:
    def __init__(self, results):
        self.results = results
        self.chunks = [item.chunk for item in results]

    def search(self, query, top_k=5, allowed_roles=None, department=None):
        return [
            item for item in self.results
            if (allowed_roles is None or set(item.chunk.access_roles) & allowed_roles)
            and (department is None or item.chunk.department.lower() == department.lower())
        ][:top_k]


class StaticGenerator:
    def __init__(self, generated=None, error=None):
        self.generated = generated
        self.error = error

    def generate(self, question, retrieved):
        if self.error:
            raise self.error
        return self.generated


def permissive_assessor():
    return EvidenceAssessor({
        "lexical": EvidencePolicy("lexical", 0.0, 0.0),
        "dense": EvidencePolicy("dense", 0.0, 0.0),
        "hybrid": EvidencePolicy("hybrid", 0.0, 0.0),
        "hybrid_rerank": EvidencePolicy("hybrid_rerank", -100.0, 0.0),
    })


def test_evidence_assessment_handles_empty_and_unrelated_results():
    assessor = EvidenceAssessor()
    assert assessor.assess("production approval", [], "lexical").reason_code == "NO_RETRIEVAL_RESULTS"
    unrelated = [result(text="Cafeteria dessert choices are posted on Fridays.", score=0.9)]
    decision = assessor.assess("Which antivirus is installed on phones?", unrelated, "lexical")
    assert not decision.sufficient
    assert decision.reason_code == "LOW_QUERY_EVIDENCE_OVERLAP"


def test_strategy_policies_do_not_share_raw_score_threshold():
    assessor = EvidenceAssessor()
    lexical = assessor.assess("production access manager approval", [result(score=0.05)], "lexical")
    hybrid = assessor.assess("production access manager approval", [result(score=0.02)], "hybrid")
    assert lexical.reason_code == "LOW_RELEVANCE_SCORE"
    assert hybrid.sufficient
    assert lexical.policy.minimum_top_score != hybrid.policy.minimum_top_score


def test_current_question_with_close_policy_versions_abstains_on_conflict():
    old = result("old", "policy-2025", score=0.79, version="2025")
    old = replace(old, chunk=replace(old.chunk, title="Access Policy 2025"))
    decision = EvidenceAssessor().assess(
        "What is the current production access policy in 2026?", [result(score=0.8), old], "dense"
    )
    assert decision.reason_code == "CONTRADICTORY_VERSION_EVIDENCE"


def test_citation_validation_rejects_unknown_or_unauthorized_chunk_ids():
    generated = GeneratedAnswer(
        "Manager approval is required.",
        (GroundedClaim("c1", "Manager approval is required.", ("private-chunk",)),),
    )
    validation = validate_citations(generated, [result("authorized-chunk")])
    assert not validation.valid
    assert validation.invalid_chunk_ids == ("private-chunk",)


def test_service_returns_structured_claim_mapping_and_evidence():
    evidence = result()
    generated = GeneratedAnswer(
        "Manager approval is required. [chunk-1]",
        (GroundedClaim("c1", "Manager approval is required.", ("chunk-1",)),),
    )
    service = RagService(
        retriever=StaticRetriever([evidence]), retriever_type="lexical",
        assessor=permissive_assessor(), generator=StaticGenerator(generated),
    )
    response = service.ask("Who approves production access?", 5, {"security"})
    assert response["answerable"] and not response["abstained"]
    assert response["reason_code"] == "ANSWER_GROUNDED"
    assert response["source_document_ids"] == ["doc-1"]
    assert response["source_chunk_ids"] == ["chunk-1"]
    assert response["claim_to_evidence"] == {"c1": ["chunk-1"]}
    assert response["citation_validation"]["valid"]


def test_service_rejects_generator_citation_outside_authorized_context():
    generated = GeneratedAnswer(
        "Secret answer", (GroundedClaim("c1", "Secret answer", ("unauthorized",)),)
    )
    service = RagService(
        retriever=StaticRetriever([result()]), retriever_type="lexical",
        assessor=permissive_assessor(), generator=StaticGenerator(generated),
    )
    response = service.ask("production access", 5, {"security"})
    assert response["abstained"]
    assert response["reason_code"] == "INVALID_GENERATED_CITATIONS"
    assert response["claims"] == []


def test_service_handles_generator_failure_safely():
    service = RagService(
        retriever=StaticRetriever([result()]), retriever_type="lexical",
        assessor=permissive_assessor(), generator=StaticGenerator(error=RuntimeError("offline")),
    )
    response = service.ask("production access", 5, {"security"})
    assert response["abstained"]
    assert response["reason_code"] == "GENERATOR_FAILURE"
    assert "RuntimeError" in response["generation_error"]


def test_no_authorized_results_prevents_generation():
    generator = StaticGenerator(error=AssertionError("must not be called"))
    service = RagService(
        retriever=StaticRetriever([result()]), retriever_type="lexical",
        assessor=permissive_assessor(), generator=generator,
    )
    response = service.ask("production access", 5, {"sales"})
    assert response["reason_code"] == "NO_RETRIEVAL_RESULTS"

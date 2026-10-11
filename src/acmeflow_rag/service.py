from pathlib import Path

from .config import settings
from .evidence import EvidenceAssessor
from .generator import GroundedGenerator, build_generator
from .grounding import GeneratedAnswer, validate_citations
from .retrieval import Retriever, load_retriever


NO_ANSWER = "I could not find enough relevant evidence in the approved knowledge base to answer this question."


class RagService:
    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        retriever_type: str | None = None,
        artifacts_dir: Path | None = None,
        index_path: Path | None = None,
        assessor: EvidenceAssessor | None = None,
        generator: GroundedGenerator | None = None,
    ):
        selected_type = retriever_type or settings.retrieval_strategy
        if retriever is None:
            if index_path is not None:
                selected_type = "lexical"
                artifacts_dir = index_path.parent
            selected_dir = artifacts_dir or settings.artifacts_dir
            legacy_type = "tfidf" if selected_type in {"lexical", "tfidf"} else selected_type
            expected = selected_dir / ("index.joblib" if legacy_type == "tfidf" else "index.faiss")
            if not expected.exists():
                raise FileNotFoundError(
                    f"{selected_type} index not found at {expected}. Run python -m acmeflow_rag.ingest"
                )
            retriever = load_retriever(legacy_type, selected_dir)
        self.retriever = retriever
        self.retriever_type = "lexical" if selected_type == "tfidf" else selected_type
        self.assessor = assessor or EvidenceAssessor()
        self.generator = generator or build_generator(settings.use_gemini, settings.gemini_model)

    def _abstention_response(self, reason_code: str, results, assessment, error: str | None = None):
        response = self._base_response(NO_ANSWER, True, reason_code, results, assessment, ())
        if error:
            response["generation_error"] = error
        return response

    @staticmethod
    def _base_response(answer, abstained, reason_code, results, assessment, claims):
        document_ids = list(dict.fromkeys(result.chunk.document_id for result in results))
        chunk_ids = [result.chunk.chunk_id for result in results]
        return {
            "answer": answer,
            "answerable": not abstained,
            "abstained": abstained,
            "reason_code": reason_code,
            "source_document_ids": document_ids,
            "source_chunk_ids": chunk_ids,
            "evidence_passages": [
                {
                    "chunk_id": result.chunk.chunk_id,
                    "document_id": result.chunk.document_id,
                    "heading_path": list(result.chunk.heading_path),
                    "text": result.chunk.text,
                    "relevance_score": result.score,
                }
                for result in results
            ],
            "claims": [claim.to_dict() for claim in claims],
            "claim_to_evidence": {
                claim.claim_id: list(claim.cited_chunk_ids) for claim in claims
            },
            "evidence_assessment": assessment.to_dict(),
            "retrieval": [result.to_dict() for result in results],
        }

    def ask(self, question: str, top_k: int, roles: set[str] | None = None, department: str | None = None):
        results = self.retriever.search(
            question, top_k=top_k, allowed_roles=roles, department=department
        )
        assessment = self.assessor.assess(question, results, self.retriever_type)
        if not assessment.sufficient:
            return self._abstention_response(assessment.reason_code, results, assessment)
        try:
            generated: GeneratedAnswer = self.generator.generate(question, results)
        except Exception as exc:
            return self._abstention_response(
                "GENERATOR_FAILURE", results, assessment, f"{type(exc).__name__}: {exc}"
            )
        if not generated.answer.strip() or not generated.claims:
            return self._abstention_response("GENERATOR_NO_ANSWER", results, assessment)
        citation_validation = validate_citations(generated, results)
        if not citation_validation.valid:
            response = self._abstention_response("INVALID_GENERATED_CITATIONS", results, assessment)
            response["citation_validation"] = citation_validation.to_dict()
            return response
        response = self._base_response(
            generated.answer, False, "ANSWER_GROUNDED", results, assessment, generated.claims
        )
        response["citation_validation"] = citation_validation.to_dict()
        return response

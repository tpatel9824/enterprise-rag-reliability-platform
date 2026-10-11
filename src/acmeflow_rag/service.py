from pathlib import Path
from datetime import date

from .audit import AuditSink, LoggingAuditSink, make_audit_event
from .auth import AuthenticationError, Principal
from .config import settings
from .evidence import EvidenceAssessor
from .generator import GroundedGenerator, build_generator
from .grounding import GeneratedAnswer, validate_citations
from .lifecycle import PolicySelectionError, select_policies, validate_chunk_lifecycle
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
        audit_sink: AuditSink | None = None,
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
        validate_chunk_lifecycle(self.retriever.chunks)
        self.retriever_type = "lexical" if selected_type == "tfidf" else selected_type
        self.assessor = assessor or EvidenceAssessor()
        self.generator = generator or build_generator(settings.use_gemini, settings.gemini_model)
        self.audit_sink = audit_sink or LoggingAuditSink()

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

    def ask(
        self,
        question: str,
        top_k: int,
        principal: Principal | None = None,
        department: str | None = None,
        as_of: date | None = None,
    ):
        if principal is None:
            raise AuthenticationError("Verified identity is required")
        effective_date = as_of or date.today()
        try:
            selection = select_policies(
                self.retriever.chunks,
                as_of=effective_date,
                customer_id=principal.customer_id,
            )
        except PolicySelectionError:
            assessment = self.assessor.assess(question, [], self.retriever_type)
            response = self._abstention_response(
                "AMBIGUOUS_POLICY_PRECEDENCE", [], assessment
            )
            self._audit(principal, response, effective_date)
            return response
        results = self.retriever.search(
            question,
            top_k=top_k,
            allowed_roles=set(principal.roles),
            department=department,
            eligible_document_ids=set(selection.eligible_document_ids),
        )
        assessment = self.assessor.assess(question, results, self.retriever_type)
        if not assessment.sufficient:
            has_accessible_document = any(
                chunk.document_id in selection.eligible_document_ids
                and principal.roles.intersection(chunk.access_roles)
                and (department is None or chunk.department == department)
                for chunk in self.retriever.chunks
            )
            reason = assessment.reason_code if has_accessible_document else "ACCESS_DENIED"
            response = self._abstention_response(reason, results, assessment)
            response["policy_selection"] = self._selection_dict(selection)
            self._audit(principal, response, effective_date)
            return response
        try:
            generated: GeneratedAnswer = self.generator.generate(question, results)
        except Exception as exc:
            response = self._abstention_response(
                "GENERATOR_FAILURE", results, assessment, f"{type(exc).__name__}: {exc}"
            )
            response["policy_selection"] = self._selection_dict(selection)
            self._audit(principal, response, effective_date)
            return response
        if not generated.answer.strip() or not generated.claims:
            response = self._abstention_response("GENERATOR_NO_ANSWER", results, assessment)
            response["policy_selection"] = self._selection_dict(selection)
            self._audit(principal, response, effective_date)
            return response
        citation_validation = validate_citations(generated, results)
        if not citation_validation.valid:
            response = self._abstention_response("INVALID_GENERATED_CITATIONS", results, assessment)
            response["citation_validation"] = citation_validation.to_dict()
            response["policy_selection"] = self._selection_dict(selection)
            self._audit(principal, response, effective_date)
            return response
        response = self._base_response(
            generated.answer, False, "ANSWER_GROUNDED", results, assessment, generated.claims
        )
        response["citation_validation"] = citation_validation.to_dict()
        response["policy_selection"] = self._selection_dict(selection)
        self._audit(principal, response, effective_date)
        return response

    @staticmethod
    def _selection_dict(selection):
        return {
            "as_of": selection.as_of.isoformat(),
            "eligible_document_ids": sorted(selection.eligible_document_ids),
            "precedence_decisions": list(selection.precedence_decisions),
        }

    def _audit(self, principal, response, as_of):
        self.audit_sink.record(make_audit_event(
            principal.subject,
            "abstained" if response["abstained"] else "answered",
            response["reason_code"],
            response["source_document_ids"],
            as_of,
        ))

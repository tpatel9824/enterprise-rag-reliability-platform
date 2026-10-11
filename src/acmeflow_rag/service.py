from pathlib import Path

from .config import settings
from .generator import build_generator
from .retrieval import Retriever, load_retriever


class RagService:
    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        retriever_type: str | None = None,
        artifacts_dir: Path | None = None,
        index_path: Path | None = None,
    ):
        selected_type = retriever_type or settings.retriever_type
        if retriever is None:
            if index_path is not None:  # Backward-compatible Phase 1 call site.
                selected_type = "tfidf"
                artifacts_dir = index_path.parent
            selected_dir = artifacts_dir or settings.artifacts_dir
            expected = selected_dir / ("index.joblib" if selected_type == "tfidf" else "index.faiss")
            if not expected.exists():
                raise FileNotFoundError(
                    f"{selected_type} index not found at {expected}. Run python -m acmeflow_rag.ingest"
                )
            retriever = load_retriever(selected_type, selected_dir)
        self.retriever = retriever
        self.retriever_type = selected_type
        self.generator = build_generator(settings.use_gemini, settings.gemini_model)

    def ask(
        self,
        question: str,
        top_k: int,
        roles: set[str] | None = None,
        department: str | None = None,
    ):
        results = self.retriever.search(
            question, top_k=top_k, allowed_roles=roles, department=department
        )
        best = results[0].score if results else 0.0
        abstained = best < settings.abstain_threshold
        answer = (
            "I could not find enough relevant evidence in the approved knowledge base to answer this question."
            if abstained
            else self.generator.generate(question, results)
        )
        return {
            "answer": answer,
            "abstained": abstained,
            "best_score": round(best, 4),
            "retrieval": [result.to_dict() for result in results],
        }

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .models import Chunk, RetrievalResult
from .retrieval import Retriever, load_retriever


DEFAULT_RRF_CONSTANT = 60
DEFAULT_CANDIDATE_K = 20
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RetrievalPipelineConfig:
    strategy: str = "lexical"
    candidate_k: int = DEFAULT_CANDIDATE_K
    rrf_constant: int = DEFAULT_RRF_CONSTANT
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_revision: str | None = None

    def __post_init__(self):
        if self.strategy not in {"lexical", "dense", "hybrid", "hybrid_rerank"}:
            raise ValueError(f"Unknown retrieval strategy: {self.strategy!r}")
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        if self.rrf_constant < 1:
            raise ValueError("rrf_constant must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RetrievalPipelineConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _deduplicate_results(results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
    output: list[RetrievalResult] = []
    seen: set[str] = set()
    for result in results:
        if result.chunk.chunk_id in seen:
            continue
        seen.add(result.chunk.chunk_id)
        output.append(result)
    return output


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Iterable[RetrievalResult]], rrf_constant: int = DEFAULT_RRF_CONSTANT
) -> list[RetrievalResult]:
    """Fuse ranked lists by chunk ID using sum(1 / (constant + rank))."""

    if rrf_constant < 1:
        raise ValueError("rrf_constant must be at least 1")
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    best_ranks: dict[str, int] = {}
    for results in ranked_lists:
        for rank, result in enumerate(_deduplicate_results(results), 1):
            chunk_id = result.chunk.chunk_id
            chunks.setdefault(chunk_id, result.chunk)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_constant + rank)
            best_ranks[chunk_id] = min(best_ranks.get(chunk_id, rank), rank)
    ordered_ids = sorted(scores, key=lambda value: (-scores[value], best_ranks[value], value))
    return [RetrievalResult(chunks[chunk_id], scores[chunk_id]) for chunk_id in ordered_ids]


def _validate_result_constraints(
    results: Iterable[RetrievalResult],
    allowed_roles: set[str] | None,
    department: str | None,
) -> None:
    for result in results:
        if allowed_roles is not None and not allowed_roles.intersection(result.chunk.access_roles):
            raise RuntimeError("Base retriever returned an unauthorized candidate")
        if department is not None and result.chunk.department != department:
            raise RuntimeError("Base retriever returned a candidate from a disallowed department")


class HybridRetriever:
    def __init__(
        self,
        lexical: Retriever,
        dense: Retriever,
        *,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
    ):
        if candidate_k < 1 or rrf_constant < 1:
            raise ValueError("candidate_k and rrf_constant must be positive")
        lexical_ids = [chunk.chunk_id for chunk in lexical.chunks]
        dense_ids = [chunk.chunk_id for chunk in dense.chunks]
        if lexical_ids != dense_ids:
            raise ValueError("Lexical and dense indexes must contain identical ordered chunks")
        self.lexical = lexical
        self.dense = dense
        self.chunks = lexical.chunks
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant
        self.index_metadata = {
            "retriever_type": "hybrid",
            "fusion": "reciprocal_rank_fusion",
            "rrf_constant": rrf_constant,
            "candidate_k_per_retriever": candidate_k,
            "chunk_count": len(self.chunks),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_roles: set[str] | None = None,
        department: str | None = None,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            return []
        request_k = max(top_k, self.candidate_k)
        lexical = self.lexical.search(query, request_k, allowed_roles, department)
        dense = self.dense.search(query, request_k, allowed_roles, department)
        _validate_result_constraints(lexical, allowed_roles, department)
        _validate_result_constraints(dense, allowed_roles, department)
        return reciprocal_rank_fusion((lexical, dense), self.rrf_constant)[:top_k]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model_revision: str | None = None,
        model: Any | None = None,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.model = model

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder

            kwargs = {"revision": self.model_revision} if self.model_revision else {}
            self.model = CrossEncoder(self.model_name, **kwargs)
        return self.model

    def rerank(
        self, query: str, candidates: Iterable[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        unique = _deduplicate_results(candidates)
        if not unique or top_k < 1:
            return []
        pairs = [(query, result.chunk.text) for result in unique]
        scores = np.asarray(self._get_model().predict(pairs, show_progress_bar=False)).reshape(-1)
        if len(scores) != len(unique):
            raise ValueError("Cross-encoder score count does not match candidate count")
        rescored = [
            RetrievalResult(result.chunk, float(score))
            for result, score in zip(unique, scores)
        ]
        rescored.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        return rescored[:top_k]


class RerankedRetriever:
    def __init__(
        self,
        base: Retriever,
        reranker: CrossEncoderReranker,
        *,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ):
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.chunks = base.chunks
        self.index_metadata = {
            **getattr(base, "index_metadata", {}),
            "retriever_type": "hybrid_rerank",
            "reranker_model": reranker.model_name,
            "reranker_revision": reranker.model_revision,
            "reranker_candidate_k": candidate_k,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_roles: set[str] | None = None,
        department: str | None = None,
    ) -> list[RetrievalResult]:
        candidates = self.base.search(
            query, max(top_k, self.candidate_k), allowed_roles, department
        )
        _validate_result_constraints(candidates, allowed_roles, department)
        return self.reranker.rerank(query, candidates, top_k)


def normalize_strategy(strategy: str) -> str:
    return "lexical" if strategy == "tfidf" else strategy


def load_pipeline(
    config: RetrievalPipelineConfig,
    *,
    lexical_artifacts_dir: Path,
    dense_artifacts_dir: Path,
    dense_model: Any | None = None,
    faiss_module: Any | None = None,
    reranker_model: Any | None = None,
) -> Retriever:
    strategy = normalize_strategy(config.strategy)
    if strategy == "lexical":
        return load_retriever("tfidf", lexical_artifacts_dir)
    dense = load_retriever(
        "dense", dense_artifacts_dir, model=dense_model, faiss_module=faiss_module
    )
    if strategy == "dense":
        return dense
    lexical = load_retriever("tfidf", lexical_artifacts_dir)
    hybrid = HybridRetriever(
        lexical,
        dense,
        candidate_k=config.candidate_k,
        rrf_constant=config.rrf_constant,
    )
    if strategy == "hybrid":
        return hybrid
    reranker = CrossEncoderReranker(
        config.reranker_model, config.reranker_revision, reranker_model
    )
    return RerankedRetriever(hybrid, reranker, candidate_k=config.candidate_k)

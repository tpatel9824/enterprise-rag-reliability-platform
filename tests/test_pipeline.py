from pathlib import Path

import numpy as np
import pytest

from acmeflow_rag.models import Chunk, RetrievalResult
from acmeflow_rag.pipeline import (
    CrossEncoderReranker,
    HybridRetriever,
    RerankedRetriever,
    RetrievalPipelineConfig,
    reciprocal_rank_fusion,
)


def make_chunk(chunk_id, roles=("employee",), department="Product"):
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        title=chunk_id,
        department=department,
        version="1.0",
        effective_date="2026-01-01",
        access_roles=roles,
        section="Rules",
        tags=("test",),
        text=f"Text for {chunk_id}",
    )


def result(chunk, score=1.0):
    return RetrievalResult(chunk, score)


class StubRetriever:
    def __init__(self, chunks, ranking, enforce_filters=True):
        self.chunks = chunks
        self.ranking = ranking
        self.enforce_filters = enforce_filters
        self.calls = []

    def search(self, query, top_k=5, allowed_roles=None, department=None):
        self.calls.append((query, top_k, allowed_roles, department))
        output = []
        for item in self.ranking:
            if self.enforce_filters:
                if allowed_roles is not None and not allowed_roles.intersection(item.chunk.access_roles):
                    continue
                if department is not None and item.chunk.department != department:
                    continue
            output.append(item)
        return output[:top_k]


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.seen_pairs = []

    def predict(self, pairs, show_progress_bar=False):
        self.seen_pairs = list(pairs)
        return np.asarray(self.scores[:len(pairs)], dtype=np.float32)


def test_rrf_ranking_deduplicates_repeated_chunks():
    a, b, c = (make_chunk(value) for value in ("a::1", "b::1", "c::1"))

    fused = reciprocal_rank_fusion(
        ([result(a), result(b), result(a)], [result(b), result(c)]), rrf_constant=60
    )

    assert [item.chunk.chunk_id for item in fused] == ["b::1", "a::1", "c::1"]
    assert len({item.chunk.chunk_id for item in fused}) == len(fused)


def test_rrf_ties_are_deterministic_and_missing_results_are_allowed():
    a, b = make_chunk("a::1"), make_chunk("b::1")

    fused = reciprocal_rank_fusion(([result(b)], [], [result(a)]), rrf_constant=10)

    assert [item.chunk.chunk_id for item in fused] == ["a::1", "b::1"]


def test_hybrid_passes_filters_before_fusion_and_returns_only_authorized_candidates():
    public = make_chunk("public::1", ("employee",), "Product")
    secret = make_chunk("secret::1", ("security",), "Security")
    chunks = [public, secret]
    lexical = StubRetriever(chunks, [result(secret), result(public)])
    dense = StubRetriever(chunks, [result(secret), result(public)])
    hybrid = HybridRetriever(lexical, dense, candidate_k=10, rrf_constant=20)

    found = hybrid.search(
        "query", top_k=5, allowed_roles={"employee"}, department="Product"
    )

    assert [item.chunk.chunk_id for item in found] == ["public::1"]
    assert lexical.calls[0][2:] == ({"employee"}, "Product")
    assert dense.calls[0][2:] == ({"employee"}, "Product")


def test_hybrid_rejects_a_base_retriever_that_leaks_unauthorized_candidates():
    public = make_chunk("public::1", ("employee",))
    secret = make_chunk("secret::1", ("security",))
    chunks = [public, secret]
    leaking = StubRetriever(chunks, [result(secret)], enforce_filters=False)
    safe = StubRetriever(chunks, [result(public)])
    hybrid = HybridRetriever(leaking, safe)

    with pytest.raises(RuntimeError, match="unauthorized"):
        hybrid.search("query", allowed_roles={"employee"})


def test_hybrid_requires_identical_ordered_chunk_alignment():
    a, b = make_chunk("a::1"), make_chunk("b::1")

    with pytest.raises(ValueError, match="identical ordered chunks"):
        HybridRetriever(StubRetriever([a, b], []), StubRetriever([b, a], []))


def test_cross_encoder_reranking_deduplicates_and_orders_by_score():
    a, b = make_chunk("a::1"), make_chunk("b::1")
    model = FakeCrossEncoder([0.1, 0.9])
    reranker = CrossEncoderReranker(model_name="fake", model=model)

    reranked = reranker.rerank("question", [result(a), result(a), result(b)], top_k=2)

    assert [item.chunk.chunk_id for item in reranked] == ["b::1", "a::1"]
    assert model.seen_pairs == [("question", a.text), ("question", b.text)]


def test_cross_encoder_ties_use_chunk_id_for_deterministic_order():
    a, b = make_chunk("a::1"), make_chunk("b::1")
    reranker = CrossEncoderReranker(model_name="fake", model=FakeCrossEncoder([0.5, 0.5]))

    reranked = reranker.rerank("question", [result(b), result(a)], top_k=2)

    assert [item.chunk.chunk_id for item in reranked] == ["a::1", "b::1"]


def test_reranker_never_receives_unauthorized_candidates():
    public = make_chunk("public::1", ("employee",), "Product")
    secret = make_chunk("secret::1", ("security",), "Security")
    base = StubRetriever([public, secret], [result(secret), result(public)])
    model = FakeCrossEncoder([0.8])
    pipeline = RerankedRetriever(
        base, CrossEncoderReranker(model_name="fake", model=model), candidate_k=10
    )

    found = pipeline.search(
        "question", top_k=5, allowed_roles={"employee"}, department="Product"
    )

    assert [item.chunk.chunk_id for item in found] == ["public::1"]
    assert model.seen_pairs == [("question", public.text)]


def test_pipeline_config_serialization_round_trip(tmp_path):
    config = RetrievalPipelineConfig(
        strategy="hybrid_rerank",
        candidate_k=12,
        rrf_constant=42,
        reranker_model="fake/reranker",
        reranker_revision="abc123",
    )
    path = tmp_path / "pipeline.json"

    config.save(path)

    assert RetrievalPipelineConfig.load(path) == config


def test_hybrid_reranker_integration_changes_candidate_order():
    a, b, c = (make_chunk(value) for value in ("a::1", "b::1", "c::1"))
    lexical = StubRetriever([a, b, c], [result(a), result(b), result(c)])
    dense = StubRetriever([a, b, c], [result(b), result(a), result(c)])
    hybrid = HybridRetriever(lexical, dense, candidate_k=3, rrf_constant=60)
    model = FakeCrossEncoder([0.1, 0.2, 0.9])
    pipeline = RerankedRetriever(
        hybrid, CrossEncoderReranker(model_name="fake", model=model), candidate_k=3
    )

    found = pipeline.search("question", top_k=3)

    assert found[0].chunk.chunk_id == "c::1"
    assert len(found) == 3

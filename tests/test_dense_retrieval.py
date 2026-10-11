import json
from pathlib import Path

import numpy as np
import pytest

from acmeflow_rag.dense_retrieval import DenseRetriever, normalize_embeddings
from acmeflow_rag.models import Chunk
from acmeflow_rag.retrieval import TfidfRetriever, create_retriever


class FakeIndexFlatIP:
    def __init__(self, dimension):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        assert vectors.dtype == np.float32
        assert vectors.shape[1] == self.d
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, queries, k):
        scores = queries @ self.vectors.T
        order = np.argsort(scores, axis=1)[:, ::-1][:, :k]
        ranked = np.take_along_axis(scores, order, axis=1)
        return ranked.astype(np.float32), order.astype(np.int64)


class FakeFaiss:
    IndexFlatIP = FakeIndexFlatIP
    stored = {}

    @classmethod
    def write_index(cls, index, path):
        cls.stored[path] = (index.d, index.vectors.copy())
        Path(path).write_bytes(b"fake-faiss-index")

    @classmethod
    def read_index(cls, path):
        dimension, vectors = cls.stored[path]
        index = FakeIndexFlatIP(dimension)
        index.add(vectors)
        return index


class FakeModel:
    revision = "fake-revision"

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def make_chunk(document_id, text, roles=("employee",), department="Test"):
    return Chunk(
        chunk_id=f"{document_id}::stable",
        document_id=document_id,
        title=document_id,
        department=department,
        version="1.0",
        effective_date="2026-01-01",
        access_roles=roles,
        section="Rules",
        tags=("test",),
        text=text,
        heading_path=("Policy", "Rules"),
        source_path=f"/{document_id}.md",
        start_char=0,
        end_char=len(text),
    )


def build_retriever():
    chunks = [
        make_chunk("alpha", "alpha text", ("employee",), "Product"),
        make_chunk("beta", "beta text", ("engineering",), "Engineering"),
        make_chunk("secret", "secret text", ("security",), "Security"),
    ]
    model = FakeModel({
        "alpha text": [3.0, 0.0],
        "beta text": [0.0, 4.0],
        "secret text": [2.0, 2.0],
        "alpha query": [5.0, 0.0],
        "secret query": [1.0, 1.0],
    })
    retriever = DenseRetriever(
        model_name="fake/model", model=model, faiss_module=FakeFaiss
    )
    retriever.fit(
        chunks,
        corpus_version="test-v2",
        chunking_config={"max_words": 40, "overlap_words": 10},
    )
    return retriever


def test_embedding_normalization_and_zero_vector_rejection():
    normalized = normalize_embeddings(np.array([[3.0, 4.0], [0.0, 2.0]]))

    assert normalized.dtype == np.float32
    assert np.allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])
    assert np.allclose(normalized[0], [0.6, 0.8])
    with pytest.raises(ValueError, match="zero-length"):
        normalize_embeddings(np.array([[0.0, 0.0]]))


def test_dense_dimensions_ranking_and_index_metadata():
    retriever = build_retriever()

    results = retriever.search("alpha query", top_k=3)

    assert retriever.embedding_dimension == 2
    assert retriever.index.d == 2
    assert retriever.index.ntotal == 3
    assert [result.chunk.document_id for result in results][0] == "alpha"
    assert retriever.index_metadata["index_type"] == "IndexFlatIP"
    assert retriever.index_metadata["model_name"] == "fake/model"
    assert retriever.index_metadata["resolved_revision"] == "fake-revision"
    assert retriever.index_metadata["corpus_version"] == "test-v2"
    assert retriever.index_metadata["chunking_config"] == {
        "max_words": 40,
        "overlap_words": 10,
    }
    assert np.allclose(np.linalg.norm(retriever.index.vectors, axis=1), 1.0)


def test_dense_filtering_never_returns_unauthorized_or_wrong_department_chunks():
    retriever = build_retriever()

    employee_results = retriever.search(
        "secret query", top_k=5, allowed_roles={"employee"}
    )
    engineering_results = retriever.search(
        "secret query", top_k=5, allowed_roles={"engineering"}, department="Engineering"
    )

    assert [result.chunk.document_id for result in employee_results] == ["alpha"]
    assert [result.chunk.document_id for result in engineering_results] == ["beta"]
    assert all("security" not in result.chunk.access_roles for result in employee_results)


def test_dense_persistence_preserves_chunk_alignment_and_ranking(tmp_path):
    retriever = build_retriever()
    retriever.save(tmp_path)

    loaded = DenseRetriever.load(tmp_path, model=retriever.model, faiss_module=FakeFaiss)

    assert [chunk.chunk_id for chunk in loaded.chunks] == [
        chunk.chunk_id for chunk in retriever.chunks
    ]
    assert loaded.embedding_dimension == retriever.embedding_dimension
    assert loaded.search("alpha query", 3)[0].chunk.document_id == "alpha"


def test_dense_load_rejects_chunk_order_misalignment(tmp_path):
    retriever = build_retriever()
    retriever.save(tmp_path)
    chunks_path = tmp_path / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks.reverse()
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    with pytest.raises(ValueError, match="chunk order"):
        DenseRetriever.load(tmp_path, model=retriever.model, faiss_module=FakeFaiss)


def test_dense_query_dimension_must_match_index():
    retriever = build_retriever()
    retriever.model.vectors["wrong query"] = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="does not match index dimension"):
        retriever.search("wrong query")


def test_common_factory_preserves_tfidf_baseline():
    assert isinstance(create_retriever("tfidf"), TfidfRetriever)
    with pytest.raises(ValueError, match="Unknown retriever"):
        create_retriever("unknown")

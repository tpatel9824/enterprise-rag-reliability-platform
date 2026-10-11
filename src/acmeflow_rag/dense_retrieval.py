from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import Chunk, RetrievalResult


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
METADATA_FILE = "index_metadata.json"


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """Return float32 row vectors normalized to unit L2 length."""

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D matrix, received shape {matrix.shape}")
    if matrix.shape[1] < 1:
        raise ValueError("Embedding dimension must be positive")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero-length embedding")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Dense retrieval requires the 'faiss-cpu' package") from exc
    return faiss


def _load_sentence_transformer(model_name: str, revision: str | None = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Dense retrieval requires the 'sentence-transformers' package") from exc
    kwargs = {"revision": revision} if revision else {}
    return SentenceTransformer(model_name, **kwargs)


def _resolved_revision(model: Any) -> str | None:
    configured = getattr(model, "revision", None)
    if configured:
        return str(configured)
    try:
        commit_hash = model[0].auto_model.config._commit_hash
    except (AttributeError, IndexError, KeyError, TypeError):
        commit_hash = None
    return str(commit_hash) if commit_hash else None


def _chunk_from_dict(value: dict[str, Any]) -> Chunk:
    return Chunk(**{
        **value,
        "access_roles": tuple(value["access_roles"]),
        "tags": tuple(value["tags"]),
        "heading_path": tuple(value.get("heading_path", ())),
    })


class DenseRetriever:
    """Exact cosine-similarity retrieval using normalized vectors and FAISS inner product."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str | None = None,
        model: Any | None = None,
        faiss_module: Any | None = None,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.model = model
        self._faiss = faiss_module
        self.index = None
        self.chunks: list[Chunk] = []
        self.embedding_dimension: int | None = None
        self.index_metadata: dict[str, Any] = {}

    def _get_model(self):
        if self.model is None:
            self.model = _load_sentence_transformer(self.model_name, self.model_revision)
        return self.model

    def _get_faiss(self):
        if self._faiss is None:
            self._faiss = _load_faiss()
        return self._faiss

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._get_model().encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return normalize_embeddings(vectors)

    def fit(
        self,
        chunks: list[Chunk],
        *,
        corpus_version: str,
        chunking_config: dict[str, int],
    ) -> None:
        if not chunks:
            raise ValueError("Cannot fit with zero chunks")
        vectors = self._encode([chunk.text for chunk in chunks])
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding row count {vectors.shape[0]} does not match chunk count {len(chunks)}"
            )
        dimension = int(vectors.shape[1])
        index = self._get_faiss().IndexFlatIP(dimension)
        index.add(vectors)
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index size does not match chunk count after indexing")
        self.index = index
        self.chunks = list(chunks)
        self.embedding_dimension = dimension
        self.index_metadata = {
            "retriever_type": "dense",
            "model_name": self.model_name,
            "requested_revision": self.model_revision,
            "resolved_revision": _resolved_revision(self._get_model()) or self.model_revision,
            "embedding_dimension": dimension,
            "index_type": "IndexFlatIP",
            "similarity": "cosine via L2-normalized inner product",
            "corpus_version": corpus_version,
            "chunking_config": chunking_config,
            "chunk_count": len(chunks),
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_roles: set[str] | None = None,
        department: str | None = None,
    ) -> list[RetrievalResult]:
        if self.index is None or self.embedding_dimension is None:
            raise RuntimeError("Retriever has not been fitted or loaded")
        if top_k < 1:
            return []
        query_vector = self._encode([query])
        if query_vector.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension {query_vector.shape[1]} does not match index dimension "
                f"{self.embedding_dimension}"
            )

        # Search the complete exact index, then filter. This prevents an unauthorized candidate
        # from being returned and still allows filling top_k from lower-ranked authorized chunks.
        scores, indices = self.index.search(query_vector, self.index.ntotal)
        output: list[RetrievalResult] = []
        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue
            chunk = self.chunks[int(index_position)]
            if allowed_roles is not None and not allowed_roles.intersection(chunk.access_roles):
                continue
            if department is not None and chunk.department != department:
                continue
            output.append(RetrievalResult(chunk=chunk, score=float(score)))
            if len(output) >= top_k:
                break
        return output

    def save(self, artifacts_dir: Path) -> None:
        if self.index is None or self.embedding_dimension is None:
            raise RuntimeError("Cannot save an unfitted dense retriever")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._get_faiss().write_index(self.index, str(artifacts_dir / INDEX_FILE))
        (artifacts_dir / CHUNKS_FILE).write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], indent=2), encoding="utf-8"
        )
        (artifacts_dir / METADATA_FILE).write_text(
            json.dumps(self.index_metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        artifacts_dir: Path,
        *,
        model: Any | None = None,
        faiss_module: Any | None = None,
    ) -> "DenseRetriever":
        metadata = json.loads((artifacts_dir / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata.get("retriever_type") != "dense":
            raise ValueError("Index metadata does not describe a dense retriever")
        revision = metadata.get("resolved_revision") or metadata.get("requested_revision")
        obj = cls(
            model_name=metadata["model_name"],
            model_revision=revision,
            model=model,
            faiss_module=faiss_module,
        )
        chunks_data = json.loads((artifacts_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        obj.chunks = [_chunk_from_dict(value) for value in chunks_data]
        obj.index = obj._get_faiss().read_index(str(artifacts_dir / INDEX_FILE))
        obj.embedding_dimension = int(metadata["embedding_dimension"])
        obj.index_metadata = metadata
        obj._validate_alignment()
        return obj

    def _validate_alignment(self) -> None:
        expected_ids = self.index_metadata.get("chunk_ids", [])
        actual_ids = [chunk.chunk_id for chunk in self.chunks]
        expected_count = int(self.index_metadata.get("chunk_count", -1))
        if expected_count != len(self.chunks) or self.index.ntotal != len(self.chunks):
            raise ValueError("Dense index, metadata, and chunk counts are not aligned")
        if expected_ids != actual_ids:
            raise ValueError("Dense index metadata and chunk order are not aligned")
        if self.index.d != self.embedding_dimension:
            raise ValueError("Dense index dimension does not match metadata")

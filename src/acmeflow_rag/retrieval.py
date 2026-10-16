from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk, RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    """Common contract implemented by sparse and dense retrievers."""

    chunks: list[Chunk]

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_roles: set[str] | None = None,
        department: str | None = None,
        eligible_document_ids: set[str] | None = None,
    ) -> list[RetrievalResult]: ...


class TfidfRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
        )
        self.matrix = None
        self.chunks: list[Chunk] = []

    def fit(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot fit with zero chunks")
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

    def search(
        self,
        query: str,
        top_k=5,
        allowed_roles: set[str] | None = None,
        department: str | None = None,
        eligible_document_ids: set[str] | None = None,
    ):
        if self.matrix is None:
            raise RuntimeError("Retriever has not been fitted")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = np.argsort(scores)[::-1]
        output = []
        for index in order:
            chunk = self.chunks[int(index)]
            if allowed_roles is not None and not allowed_roles.intersection(chunk.access_roles):
                continue
            if department is not None and chunk.department != department:
                continue
            if eligible_document_ids is not None and chunk.document_id not in eligible_document_ids:
                continue
            score = float(scores[index])
            if score <= 0:
                continue
            output.append(RetrievalResult(chunk, score))
            if len(output) >= top_k:
                break
        return output

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "chunks": [chunk.to_dict() for chunk in self.chunks],
            },
            path,
        )

    @classmethod
    def load(cls, path: Path):
        payload = joblib.load(path)
        obj = cls()
        obj.vectorizer = payload["vectorizer"]
        obj.matrix = payload["matrix"]
        obj.chunks = [
            Chunk(**{
                **value,
                "access_roles": tuple(value["access_roles"]),
                "tags": tuple(value["tags"]),
                "heading_path": tuple(value.get("heading_path", ())),
                "supersedes": tuple(value.get("supersedes", ())),
            })
            for value in payload["chunks"]
        ]
        return obj


def create_retriever(retriever_type: str, **kwargs) -> Retriever:
    if retriever_type == "tfidf":
        return TfidfRetriever()
    if retriever_type == "dense":
        from .dense_retrieval import DenseRetriever

        return DenseRetriever(**kwargs)
    raise ValueError(f"Unknown retriever type: {retriever_type!r}")


def load_retriever(retriever_type: str, artifacts_dir: Path, **kwargs) -> Retriever:
    if retriever_type == "tfidf":
        return TfidfRetriever.load(artifacts_dir / "index.joblib")
    if retriever_type == "dense":
        from .dense_retrieval import DenseRetriever

        return DenseRetriever.load(artifacts_dir, **kwargs)
    raise ValueError(f"Unknown retriever type: {retriever_type!r}")

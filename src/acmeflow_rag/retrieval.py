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
    ) -> list[RetrievalResult]: ...

class TfidfRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.98, sublinear_tf=True)
        self.matrix = None
        self.chunks: list[Chunk] = []
    def fit(self, chunks: list[Chunk]):
        if not chunks: raise ValueError("Cannot fit with zero chunks")
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])
    def search(self, query: str, top_k=5, allowed_roles: set[str] | None=None, department: str|None=None):
        if self.matrix is None: raise RuntimeError("Retriever has not been fitted")
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(scores)[::-1]
        out=[]
        for idx in order:
            c=self.chunks[int(idx)]
            if allowed_roles is not None and not allowed_roles.intersection(set(c.access_roles)): continue
            if department is not None and c.department != department: continue
            score=float(scores[idx])
            if score <= 0: continue
            out.append(RetrievalResult(c, score))
            if len(out)>=top_k: break
        return out
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer":self.vectorizer,"matrix":self.matrix,"chunks":[c.to_dict() for c in self.chunks]}, path)
    @classmethod
    def load(cls, path: Path):
        p=joblib.load(path); obj=cls(); obj.vectorizer=p["vectorizer"]; obj.matrix=p["matrix"]
        obj.chunks=[Chunk(**{**x,"access_roles":tuple(x["access_roles"]),"tags":tuple(x["tags"]),
                              "heading_path":tuple(x.get("heading_path", ()))}) for x in p["chunks"]]
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

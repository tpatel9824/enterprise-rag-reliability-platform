from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Chunk, RetrievalResult

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

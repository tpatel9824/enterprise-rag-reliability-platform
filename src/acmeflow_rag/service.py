from .config import settings
from .generator import build_generator
from .retrieval import TfidfRetriever

class RagService:
    def __init__(self, index_path=None):
        path=index_path or settings.artifacts_dir/"index.joblib"
        if not path.exists(): raise FileNotFoundError(f"Index not found at {path}. Run python -m acmeflow_rag.ingest")
        self.retriever=TfidfRetriever.load(path)
        self.generator=build_generator(settings.use_gemini, settings.gemini_model)
    def ask(self, question: str, top_k: int, roles: set[str]|None=None, department: str|None=None):
        results=self.retriever.search(question, top_k=top_k, allowed_roles=roles, department=department)
        best=results[0].score if results else 0.0
        abstained=best < settings.abstain_threshold
        answer=("I could not find enough relevant evidence in the approved knowledge base to answer this question."
                if abstained else self.generator.generate(question, results))
        return {"answer":answer,"abstained":abstained,"best_score":round(best,4),"retrieval":[r.to_dict() for r in results]}

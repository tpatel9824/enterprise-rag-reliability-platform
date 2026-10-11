from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .pipeline import RetrievalPipelineConfig, load_pipeline, normalize_strategy
from .service import RagService


app = FastAPI(title="AcmeFlow Enterprise RAG", version="0.3.0")
strategy = normalize_strategy(settings.retrieval_strategy)
pipeline_config = RetrievalPipelineConfig(
    strategy=strategy,
    candidate_k=settings.retrieval_candidate_k,
    rrf_constant=settings.rrf_constant,
    reranker_model=settings.reranker_model,
    reranker_revision=settings.reranker_revision,
)
retriever = load_pipeline(
    pipeline_config,
    lexical_artifacts_dir=settings.lexical_artifacts_dir,
    dense_artifacts_dir=settings.dense_artifacts_dir,
)
service = RagService(retriever=retriever, retriever_type=strategy)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=settings.top_k, ge=1, le=10)
    roles: list[str] = Field(default_factory=list)
    department: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "retriever": service.retriever_type,
        "documents": len({chunk.document_id for chunk in service.retriever.chunks}),
        "chunks": len(service.retriever.chunks),
    }


@app.post("/ask")
def ask(request: AskRequest):
    return service.ask(
        request.question,
        request.top_k,
        set(request.roles) if request.roles else None,
        request.department,
    )

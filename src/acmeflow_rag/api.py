from datetime import date

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .artifact_store import GCSArtifactStore, LocalArtifactStore
from .auth import AuthenticationError, build_identity_provider
from .config import settings
from .observability import RequestMetricsMiddleware, ServiceMetrics, configure_logging
from .pipeline import RetrievalPipelineConfig, load_pipeline, normalize_strategy
from .service import RagService


configure_logging(settings.log_level)
app = FastAPI(title="AcmeFlow Enterprise RAG", version="0.6.0")
metrics = ServiceMetrics()
app.add_middleware(RequestMetricsMiddleware, metrics=metrics)
strategy = normalize_strategy(settings.retrieval_strategy)
pipeline_config = RetrievalPipelineConfig(
    strategy=strategy,
    candidate_k=settings.retrieval_candidate_k,
    rrf_constant=settings.rrf_constant,
    reranker_model=settings.reranker_model,
    reranker_revision=settings.reranker_revision,
)
if settings.artifact_storage_mode == "local":
    lexical_dir = settings.lexical_artifacts_dir
    dense_dir = settings.dense_artifacts_dir
    if strategy in {"lexical", "hybrid", "hybrid_rerank"}:
        lexical_dir = LocalArtifactStore(lexical_dir).materialize("tfidf")
    if strategy in {"dense", "hybrid", "hybrid_rerank"}:
        dense_dir = LocalArtifactStore(dense_dir).materialize("dense")
elif settings.artifact_storage_mode == "gcs":
    lexical_dir = settings.artifact_cache_dir / "tfidf"
    dense_dir = settings.artifact_cache_dir / "dense"
    if strategy in {"lexical", "hybrid", "hybrid_rerank"}:
        lexical_dir = GCSArtifactStore(
            settings.gcs_artifact_bucket, settings.gcs_lexical_prefix,
            settings.artifact_cache_dir,
        ).materialize("tfidf")
    if strategy in {"dense", "hybrid", "hybrid_rerank"}:
        dense_dir = GCSArtifactStore(
            settings.gcs_artifact_bucket, settings.gcs_dense_prefix,
            settings.artifact_cache_dir,
        ).materialize("dense")
else:
    raise ValueError(f"Unsupported ARTIFACT_STORAGE_MODE: {settings.artifact_storage_mode!r}")

retriever = load_pipeline(
    pipeline_config,
    lexical_artifacts_dir=lexical_dir,
    dense_artifacts_dir=dense_dir,
)
service = RagService(retriever=retriever, retriever_type=strategy)
identity_provider = build_identity_provider(settings)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3)
    top_k: int = Field(default=settings.top_k, ge=1, le=10)
    department: str | None = None
    as_of: date | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def readiness():
    return {
        "status": "ready",
        "retriever": service.retriever_type,
        "documents": len({chunk.document_id for chunk in service.retriever.chunks}),
        "chunks": len(service.retriever.chunks),
        "artifact_storage": settings.artifact_storage_mode,
    }


@app.get("/metrics")
def operational_metrics():
    return metrics.snapshot()


@app.post("/ask")
def ask(request: AskRequest, authorization: str | None = Header(default=None)):
    credential = None
    if authorization and authorization.startswith("Bearer "):
        credential = authorization.removeprefix("Bearer ").strip()
    try:
        principal = identity_provider.authenticate(credential)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    response = service.ask(
        request.question, request.top_k, principal, request.department, request.as_of
    )
    if response["reason_code"] == "ACCESS_DENIED":
        raise HTTPException(status_code=403, detail="No accessible evidence for this identity")
    return response

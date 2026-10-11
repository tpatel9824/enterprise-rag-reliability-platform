from datetime import date

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .auth import AuthenticationError, LocalIdentityProvider
from .config import settings
from .pipeline import RetrievalPipelineConfig, load_pipeline, normalize_strategy
from .service import RagService


app = FastAPI(title="AcmeFlow Enterprise RAG", version="0.5.0")
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
identity_provider = LocalIdentityProvider.from_json(settings.local_identities_path)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3)
    top_k: int = Field(default=settings.top_k, ge=1, le=10)
    department: str | None = None
    as_of: date | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "retriever": service.retriever_type,
        "documents": len({chunk.document_id for chunk in service.retriever.chunks}),
        "chunks": len(service.retriever.chunks),
    }


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

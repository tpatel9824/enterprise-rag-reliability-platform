from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    app_environment: str = os.getenv("APP_ENV", "development")
    documents_dir: Path = ROOT / "data" / "documents"
    benchmark_path: Path = ROOT / "data" / "evaluation" / "benchmark.json"
    artifacts_dir: Path = Path(os.getenv("RAG_ARTIFACTS_DIR", str(ROOT / "artifacts")))
    retriever_type: str = os.getenv("RETRIEVER_TYPE", "tfidf")
    retrieval_strategy: str = os.getenv("RETRIEVAL_STRATEGY", os.getenv("RETRIEVER_TYPE", "lexical"))
    lexical_artifacts_dir: Path = Path(
        os.getenv("LEXICAL_ARTIFACTS_DIR", str(ROOT / "artifacts"))
    )
    dense_artifacts_dir: Path = Path(
        os.getenv("DENSE_ARTIFACTS_DIR", str(ROOT / "artifacts"))
    )
    retrieval_candidate_k: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "20"))
    rrf_constant: int = int(os.getenv("RRF_CONSTANT", "60"))
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker_revision: str | None = os.getenv("RERANKER_REVISION") or None
    dense_model_name: str = os.getenv(
        "DENSE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k: int = int(os.getenv("TOP_K", "5"))
    use_gemini: bool = os.getenv("RAG_USE_GEMINI", "false").lower() == "true"
    gemini_model: str = os.getenv("GENAI_MODEL", "")
    local_identities_path: Path = Path(
        os.getenv("LOCAL_IDENTITIES_PATH", str(ROOT / "data" / "identity" / "local_identities.json"))
    )
    auth_mode: str = os.getenv("AUTH_MODE", "local")
    oidc_issuer: str = os.getenv("OIDC_ISSUER", "")
    oidc_audience: str = os.getenv("OIDC_AUDIENCE", "")
    oidc_jwks_url: str = os.getenv("OIDC_JWKS_URL", "")
    oidc_roles_claim: str = os.getenv("OIDC_ROLES_CLAIM", "roles")
    oidc_customer_claim: str = os.getenv("OIDC_CUSTOMER_CLAIM", "customer_id")
    artifact_storage_mode: str = os.getenv("ARTIFACT_STORAGE_MODE", "local")
    gcs_artifact_bucket: str = os.getenv("GCS_ARTIFACT_BUCKET", "")
    gcs_lexical_prefix: str = os.getenv("GCS_LEXICAL_PREFIX", "indexes/lexical")
    gcs_dense_prefix: str = os.getenv("GCS_DENSE_PREFIX", "indexes/dense")
    artifact_cache_dir: Path = Path(os.getenv("ARTIFACT_CACHE_DIR", "/tmp/acmeflow-artifacts"))
    google_cloud_project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    google_cloud_location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

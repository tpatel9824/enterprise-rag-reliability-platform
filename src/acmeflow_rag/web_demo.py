"""Reusable backend adapter and presentation data for the local Streamlit demo."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .auth import LocalIdentityProvider
from .config import ROOT, settings

if TYPE_CHECKING:
    from .service import RagService


STRATEGY_LABELS = {
    "Lexical (TF-IDF)": "lexical",
    "Dense (MiniLM + FAISS)": "dense",
    "Hybrid (RRF)": "hybrid",
    "Hybrid + reranker": "hybrid_rerank",
}

IDENTITY_SUBJECTS = {
    "Employee": "local-employee",
    "Support agent": "local-support-agent",
    "Security engineer": "local-security-engineer",
    "Acme account manager": "local-acme-account-manager",
}


@dataclass(frozen=True)
class ExampleQuestion:
    key: str
    label: str
    description: str
    question: str
    identity_label: str
    as_of: date
    department: str | None = None


EXAMPLES = (
    ExampleQuestion(
        "multi_document",
        "Multi-document reasoning",
        "Connect subscription behavior with the Support intake procedure.",
        "Does cancellation automatically refund the last charge, and what must Support verify for a refund request?",
        "Support agent",
        date(2026, 6, 15),
    ),
    ExampleQuestion(
        "historical_policy",
        "Historical policy",
        "Use an explicit as-of date to select the superseded 2025 policy.",
        "As of June 2025, how long did a customer have to request a refund?",
        "Support agent",
        date(2025, 6, 15),
    ),
    ExampleQuestion(
        "customer_precedence",
        "Customer-specific precedence",
        "Apply signed Acme terms instead of the standard refund policy.",
        "For Acme in June 2026, what refund window and initiation time apply?",
        "Acme account manager",
        date(2026, 6, 15),
    ),
    ExampleQuestion(
        "authorization_denial",
        "Authorization restriction",
        "Show that an employee cannot retrieve Security-only evidence.",
        "What approvals are required for production access?",
        "Employee",
        date(2026, 6, 15),
        "Security",
    ),
    ExampleQuestion(
        "abstention",
        "Evidence-based abstention",
        "Ask for a policy fact that is absent from the synthetic corpus.",
        "Which antivirus app must employees install on personal phones?",
        "Employee",
        date(2026, 6, 15),
    ),
)


def load_verified_metrics(
    path: Path = ROOT / "data" / "evaluation" / "verified_metrics_v3.json",
) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_tokens(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_subject = {
        identity["subject"]: token for token, identity in payload.get("tokens", {}).items()
    }
    missing = set(IDENTITY_SUBJECTS.values()).difference(by_subject)
    if missing:
        raise ValueError(f"Development identity fixture is missing subjects: {sorted(missing)}")
    return {label: by_subject[subject] for label, subject in IDENTITY_SUBJECTS.items()}


class DemoBackend:
    """Thin adapter over the production retrieval/service code used by the UI."""

    def __init__(
        self,
        *,
        lexical_artifacts_dir: Path = ROOT / "artifacts" / "v3" / "tfidf",
        dense_artifacts_dir: Path = ROOT / "artifacts" / "v3" / "dense",
        identities_path: Path = settings.local_identities_path,
        service_loader: Callable[[str], "RagService"] | None = None,
    ):
        if settings.app_environment == "production":
            raise RuntimeError("The development-identity demo is disabled in production")
        self.lexical_artifacts_dir = lexical_artifacts_dir
        self.dense_artifacts_dir = dense_artifacts_dir
        self.identities_path = identities_path
        self.identity_provider = LocalIdentityProvider.from_json(identities_path)
        self.identity_tokens = _identity_tokens(identities_path)
        self._services: dict[str, "RagService"] = {}
        self._service_loader = service_loader or self._load_service

    def _load_service(self, strategy: str) -> "RagService":
        from .pipeline import RetrievalPipelineConfig, load_pipeline
        from .service import RagService

        retriever = load_pipeline(
            RetrievalPipelineConfig(
                strategy=strategy,
                candidate_k=settings.retrieval_candidate_k,
                rrf_constant=settings.rrf_constant,
                reranker_model=settings.reranker_model,
                reranker_revision=settings.reranker_revision,
            ),
            lexical_artifacts_dir=self.lexical_artifacts_dir,
            dense_artifacts_dir=self.dense_artifacts_dir,
        )
        return RagService(retriever=retriever, retriever_type=strategy)

    def _service(self, strategy: str) -> "RagService":
        if strategy not in STRATEGY_LABELS.values():
            raise ValueError(f"Unsupported demo strategy: {strategy!r}")
        if strategy not in self._services:
            self._services[strategy] = self._service_loader(strategy)
        return self._services[strategy]

    def ask(
        self,
        *,
        question: str,
        strategy: str,
        identity_label: str,
        as_of: date,
        department: str | None,
        top_k: int,
    ) -> dict:
        if identity_label not in self.identity_tokens:
            raise ValueError(f"Unknown development identity: {identity_label!r}")
        principal = self.identity_provider.authenticate(self.identity_tokens[identity_label])
        started = time.perf_counter()
        response = self._service(strategy).ask(
            question.strip(), top_k, principal, department or None, as_of
        )
        output = dict(response)
        output["demo_metadata"] = {
            "strategy": strategy,
            "identity_label": identity_label,
            "end_to_end_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "latency_scope": "Single local end-to-end request; model loading is cached after first use.",
        }
        return output


def build_demo_backend() -> DemoBackend:
    return DemoBackend()

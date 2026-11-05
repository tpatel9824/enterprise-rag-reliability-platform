from datetime import date
import json

from acmeflow_rag.auth import Principal
from acmeflow_rag.web_demo import DemoBackend, EXAMPLES, load_verified_metrics


class FakeService:
    def __init__(self):
        self.calls = []

    def ask(self, question, top_k, principal, department, as_of):
        self.calls.append((question, top_k, principal, department, as_of))
        return {
            "answer": "Grounded fixture answer.",
            "answerable": True,
            "abstained": False,
            "reason_code": "ANSWER_GROUNDED",
            "source_document_ids": ["doc-1"],
            "source_chunk_ids": ["chunk-1"],
            "evidence_passages": [],
            "claims": [],
            "claim_to_evidence": {},
            "evidence_assessment": {},
            "retrieval": [],
            "policy_selection": {},
        }


def test_demo_backend_uses_existing_service_and_server_side_identity(tmp_path):
    identities = tmp_path / "identities.json"
    identities.write_text(json.dumps({"tokens": {
        "employee-secret": {"subject": "local-employee", "roles": ["employee"]},
        "support-secret": {"subject": "local-support-agent", "roles": ["support"]},
        "security-secret": {"subject": "local-security-engineer", "roles": ["security"]},
        "acme-secret": {"subject": "local-acme-account-manager", "roles": ["support"], "customer_id": "customer-acme"},
    }}), encoding="utf-8")
    service = FakeService()
    backend = DemoBackend(identities_path=identities, service_loader=lambda strategy: service)

    result = backend.ask(
        question="What is the policy?",
        strategy="lexical",
        identity_label="Support agent",
        as_of=date(2026, 6, 15),
        department="Customer Support",
        top_k=5,
    )

    assert result["reason_code"] == "ANSWER_GROUNDED"
    assert result["demo_metadata"]["strategy"] == "lexical"
    assert result["demo_metadata"]["identity_label"] == "Support agent"
    assert result["demo_metadata"]["end_to_end_latency_ms"] >= 0
    _, _, principal, department, as_of = service.calls[0]
    assert principal == Principal("local-support-agent", frozenset({"support"}))
    assert department == "Customer Support"
    assert as_of == date(2026, 6, 15)


def test_examples_cover_required_governance_behaviors():
    assert {example.key for example in EXAMPLES} == {
        "multi_document", "historical_policy", "customer_precedence",
        "authorization_denial", "abstention",
    }
    assert all(example.question and example.identity_label for example in EXAMPLES)


def test_verified_metrics_snapshot_has_all_strategies_and_scope():
    metrics = load_verified_metrics()
    assert metrics["case_count"] == 16
    assert [row["strategy"] for row in metrics["strategies"]] == [
        "Lexical", "Dense", "Hybrid", "Hybrid + reranker",
    ]
    assert "retrieval-only" in metrics["latency_scope"]
    assert "independent review pending" in metrics["review_status"]

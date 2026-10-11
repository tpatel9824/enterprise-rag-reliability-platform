from datetime import date

import pytest
from pydantic import ValidationError

from acmeflow_rag.audit import MemoryAuditSink
from acmeflow_rag.auth import AuthenticationError, LocalIdentityProvider, Principal
from acmeflow_rag.evidence import EvidenceAssessor, EvidencePolicy
from acmeflow_rag.lifecycle import PolicySelectionError, select_policies, validate_chunk_lifecycle
from acmeflow_rag.loader import MetadataValidationError, load_corpus
from acmeflow_rag.models import Chunk, RetrievalResult
from acmeflow_rag.service import RagService


def chunk(
    document_id,
    *,
    family="refund",
    start="2026-01-01",
    end=None,
    status="active",
    roles=("employee",),
    scope="standard",
    customer=None,
    signed=False,
    text="Refund requests are allowed within fourteen days.",
):
    return Chunk(
        chunk_id=f"{document_id}::1", document_id=document_id, title=document_id,
        department="Product", version="1", effective_date=start, access_roles=roles,
        section="Policy", tags=("test",), text=text, effective_from=start,
        effective_until=end, status=status, policy_family=family, policy_scope=scope,
        applies_to_customer=customer, signed=signed,
    )


class FilteringRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.seen = []

    def search(self, query, top_k=5, allowed_roles=None, department=None, eligible_document_ids=None):
        self.seen.append((allowed_roles, eligible_document_ids))
        return [
            RetrievalResult(item, 0.9) for item in self.chunks
            if (allowed_roles is None or allowed_roles.intersection(item.access_roles))
            and (eligible_document_ids is None or item.document_id in eligible_document_ids)
        ][:top_k]


def permissive():
    return EvidenceAssessor({name: EvidencePolicy(name, -100, 0) for name in (
        "lexical", "dense", "hybrid", "hybrid_rerank"
    )})


def test_local_identity_provider_rejects_missing_and_unknown_credentials():
    provider = LocalIdentityProvider({"valid": Principal("user", frozenset({"employee"}))})
    with pytest.raises(AuthenticationError):
        provider.authenticate(None)
    with pytest.raises(AuthenticationError):
        provider.authenticate("forged")
    assert provider.authenticate("valid").roles == frozenset({"employee"})


def test_api_request_schema_rejects_client_supplied_role_escalation():
    from acmeflow_rag.api import AskRequest

    with pytest.raises(ValidationError):
        AskRequest.model_validate({"question": "production access rules", "roles": ["security"]})


def test_service_rejects_missing_verified_identity():
    service = RagService(retriever=FilteringRetriever([chunk("current")]), retriever_type="lexical")
    with pytest.raises(AuthenticationError):
        service.ask("refund window", 5)


def test_current_and_historical_version_selection():
    old = chunk("old", start="2025-01-01", end="2025-12-31", status="superseded")
    current = chunk("current", start="2026-01-01")
    assert select_policies([old, current], as_of=date(2025, 6, 1)).eligible_document_ids == {"old"}
    assert select_policies([old, current], as_of=date(2026, 6, 1)).eligible_document_ids == {"current"}


def test_signed_customer_terms_override_only_for_matching_customer_and_date():
    standard = chunk("standard")
    contract = chunk(
        "contract", start="2026-04-01", end="2026-12-31", scope="customer",
        customer="customer-acme", signed=True, roles=("support",),
    )
    employee = select_policies([standard, contract], as_of=date(2026, 6, 1))
    customer = select_policies(
        [standard, contract], as_of=date(2026, 6, 1), customer_id="customer-acme"
    )
    before = select_policies(
        [standard, contract], as_of=date(2026, 2, 1), customer_id="customer-acme"
    )
    assert employee.eligible_document_ids == {"standard"}
    assert customer.eligible_document_ids == {"contract"}
    assert before.eligible_document_ids == {"standard"}


def test_ambiguous_customer_precedence_fails_closed():
    contracts = [
        chunk(f"contract-{value}", scope="customer", customer="customer-acme", signed=True)
        for value in (1, 2)
    ]
    with pytest.raises(PolicySelectionError, match="Ambiguous"):
        select_policies(contracts, as_of=date(2026, 6, 1), customer_id="customer-acme")


def test_cross_role_access_is_filtered_before_results_and_audit_has_no_text():
    public = chunk("public", roles=("employee",), text="Public refund guidance secret-marker.")
    protected = chunk("protected", family="security", roles=("security",), text="Sensitive secret-marker.")
    retriever = FilteringRetriever([protected, public])
    audit = MemoryAuditSink()
    service = RagService(
        retriever=retriever, retriever_type="lexical", assessor=permissive(), audit_sink=audit
    )
    response = service.ask("refund guidance", 5, Principal("employee-1", frozenset({"employee"})))
    assert "protected" not in response["source_document_ids"]
    assert retriever.seen[0][0] == {"employee"}
    assert audit.events[0].document_ids == ("public",)
    assert "secret-marker" not in repr(audit.events[0])


def test_lifecycle_metadata_tampering_is_rejected(tmp_path):
    source = tmp_path / "bad.md"
    source.write_text("""---
document_id: bad_policy
title: Bad Policy
department: Test
version: "2"
effective_date: "2026-01-01"
effective_from: "2026-01-01"
effective_until: "2025-01-01"
supersedes: [missing_policy]
status: active
policy_family: bad
access_roles: [employee]
tags: [test]
---
# Bad
Text.
""", encoding="utf-8")
    with pytest.raises(MetadataValidationError):
        load_corpus(tmp_path, require_lifecycle=True)


def test_tampered_index_chunk_metadata_is_rejected():
    damaged = chunk("damaged", start="not-a-date")
    with pytest.raises(PolicySelectionError, match="Malformed"):
        validate_chunk_lifecycle([damaged])

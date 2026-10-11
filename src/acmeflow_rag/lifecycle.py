"""Effective-date and explicit customer-policy precedence selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .models import Chunk


class PolicySelectionError(ValueError):
    pass


def validate_chunk_lifecycle(chunks: list[Chunk]) -> None:
    documents: dict[str, Chunk] = {}
    signatures: dict[str, tuple] = {}
    for chunk in chunks:
        try:
            start = date.fromisoformat(chunk.effective_from or chunk.effective_date)
            end = date.fromisoformat(chunk.effective_until) if chunk.effective_until else None
        except ValueError as exc:
            raise PolicySelectionError(
                f"Malformed lifecycle date for document {chunk.document_id!r}"
            ) from exc
        if end is not None and end < start:
            raise PolicySelectionError(
                f"Invalid lifecycle range for document {chunk.document_id!r}"
            )
        if chunk.status not in {"draft", "active", "superseded", "retired"}:
            raise PolicySelectionError(f"Invalid lifecycle status for {chunk.document_id!r}")
        signature = (
            chunk.effective_from, chunk.effective_until, chunk.supersedes, chunk.status,
            chunk.policy_family, chunk.policy_scope, chunk.applies_to_customer, chunk.signed,
        )
        if chunk.document_id in signatures and signatures[chunk.document_id] != signature:
            raise PolicySelectionError(
                f"Inconsistent lifecycle metadata across chunks for {chunk.document_id!r}"
            )
        signatures[chunk.document_id] = signature
        documents[chunk.document_id] = chunk
    for chunk in documents.values():
        for reference in chunk.supersedes:
            if reference not in documents:
                raise PolicySelectionError(
                    f"Unknown supersedes reference {reference!r} in indexed metadata"
                )


@dataclass(frozen=True)
class PolicySelection:
    as_of: date
    eligible_document_ids: frozenset[str]
    excluded_document_ids: frozenset[str]
    precedence_decisions: tuple[dict[str, str], ...]


def select_policies(
    chunks: list[Chunk], *, as_of: date, customer_id: str | None = None
) -> PolicySelection:
    validate_chunk_lifecycle(chunks)
    documents = {chunk.document_id: chunk for chunk in chunks}
    effective: dict[str, Chunk] = {}
    for document_id, chunk in documents.items():
        start = date.fromisoformat(chunk.effective_from or chunk.effective_date)
        end = date.fromisoformat(chunk.effective_until) if chunk.effective_until else None
        if chunk.status == "draft" or start > as_of or (end is not None and as_of > end):
            continue
        if chunk.policy_scope == "customer":
            if not customer_id or chunk.applies_to_customer != customer_id or not chunk.signed:
                continue
        effective[document_id] = chunk

    decisions: list[dict[str, str]] = []
    by_family: dict[str, list[Chunk]] = {}
    for chunk in effective.values():
        by_family.setdefault(chunk.policy_family or chunk.document_id, []).append(chunk)

    eligible: set[str] = set()
    for family, family_documents in by_family.items():
        customer_documents = [item for item in family_documents if item.policy_scope == "customer"]
        if len(customer_documents) > 1:
            raise PolicySelectionError(
                f"Ambiguous policy precedence for customer {customer_id!r} and family {family!r}"
            )
        if customer_documents:
            selected = customer_documents[0]
            eligible.add(selected.document_id)
            decisions.append({
                "policy_family": family,
                "decision": "customer_terms_override_standard",
                "selected_document_id": selected.document_id,
            })
            continue
        standard = [item for item in family_documents if item.policy_scope == "standard"]
        if not standard:
            continue
        latest_start = max(date.fromisoformat(item.effective_from or item.effective_date) for item in standard)
        latest = [
            item for item in standard
            if date.fromisoformat(item.effective_from or item.effective_date) == latest_start
        ]
        if len(latest) != 1:
            raise PolicySelectionError(f"Ambiguous effective policy for family {family!r}")
        eligible.add(latest[0].document_id)

    return PolicySelection(
        as_of=as_of,
        eligible_document_ids=frozenset(eligible),
        excluded_document_ids=frozenset(documents).difference(eligible),
        precedence_decisions=tuple(decisions),
    )

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Any

import yaml

from .models import Document


META_RE = re.compile(r"^- \*\*(.+?):\*\* (.+)$")
DOCUMENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
REQUIRED_METADATA = {
    "document_id", "title", "department", "version", "effective_date", "access_roles", "tags"
}
LIFECYCLE_FIELDS = {"effective_from", "effective_until", "supersedes", "status", "policy_family"}
VALID_STATUSES = {"draft", "active", "superseded", "retired"}


class MetadataValidationError(ValueError):
    """Raised when document metadata is missing or does not satisfy the corpus schema."""


def _front_matter(text: str, path: Path) -> tuple[dict[str, Any], str] | None:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None
    closing = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if closing is None:
        raise MetadataValidationError(f"{path}: YAML front matter has no closing '---'")
    try:
        metadata = yaml.safe_load("".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise MetadataValidationError(f"{path}: invalid YAML front matter: {exc}") from exc
    if not isinstance(metadata, dict):
        raise MetadataValidationError(f"{path}: YAML front matter must be a mapping")
    return metadata, "".join(lines[closing + 1:]).strip()


def _legacy_metadata(text: str) -> tuple[dict[str, Any], str]:
    metadata: dict[str, Any] = {}
    for line in text.splitlines():
        match = META_RE.match(line)
        if match:
            metadata[match.group(1).lower().replace(" ", "_")] = match.group(2)
    marker = "## Policy / Guidance"
    body = text.split(marker, 1)[1].strip() if marker in text else text.strip()
    for field in ("access_roles", "tags"):
        raw = metadata.get(field, "")
        metadata[field] = [
            item.strip() for item in raw.strip("[]").replace("'", "").split(",") if item.strip()
        ]
    return metadata, body


def _validate_metadata(metadata: dict[str, Any], path: Path, require_lifecycle: bool = False) -> dict[str, Any]:
    missing = sorted(REQUIRED_METADATA - metadata.keys())
    if missing:
        raise MetadataValidationError(f"{path}: missing required metadata: {', '.join(missing)}")

    for field in ("document_id", "title", "department", "version", "effective_date"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise MetadataValidationError(f"{path}: {field} must be a non-empty string")
    if not DOCUMENT_ID_RE.fullmatch(metadata["document_id"]):
        raise MetadataValidationError(
            f"{path}: document_id must contain only lowercase letters, digits, underscores, or hyphens"
        )
    try:
        date.fromisoformat(metadata["effective_date"])
    except ValueError as exc:
        raise MetadataValidationError(f"{path}: effective_date must be an ISO date (YYYY-MM-DD)") from exc

    for field in ("access_roles", "tags"):
        value = metadata[field]
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise MetadataValidationError(f"{path}: {field} must be a non-empty list of strings")
        if len(value) != len(set(value)):
            raise MetadataValidationError(f"{path}: {field} must not contain duplicates")
    lifecycle_present = bool(LIFECYCLE_FIELDS.intersection(metadata))
    if require_lifecycle and not LIFECYCLE_FIELDS.issubset(metadata):
        missing_lifecycle = sorted(LIFECYCLE_FIELDS - metadata.keys())
        raise MetadataValidationError(
            f"{path}: missing required lifecycle metadata: {', '.join(missing_lifecycle)}"
        )
    if lifecycle_present:
        missing_lifecycle = sorted(LIFECYCLE_FIELDS - metadata.keys())
        if missing_lifecycle:
            raise MetadataValidationError(
                f"{path}: partial lifecycle metadata; missing: {', '.join(missing_lifecycle)}"
            )
        for field in ("effective_from", "policy_family"):
            if not isinstance(metadata[field], str) or not metadata[field].strip():
                raise MetadataValidationError(f"{path}: {field} must be a non-empty string")
        try:
            effective_from = date.fromisoformat(metadata["effective_from"])
            effective_until = (
                date.fromisoformat(metadata["effective_until"])
                if metadata["effective_until"] is not None else None
            )
        except (TypeError, ValueError) as exc:
            raise MetadataValidationError(
                f"{path}: lifecycle dates must be ISO dates or effective_until: null"
            ) from exc
        if effective_until is not None and effective_until < effective_from:
            raise MetadataValidationError(f"{path}: effective_until precedes effective_from")
        if metadata["status"] not in VALID_STATUSES:
            raise MetadataValidationError(f"{path}: invalid status {metadata['status']!r}")
        supersedes = metadata["supersedes"]
        if not isinstance(supersedes, list) or not all(isinstance(item, str) for item in supersedes):
            raise MetadataValidationError(f"{path}: supersedes must be a list of document IDs")
        if len(supersedes) != len(set(supersedes)) or metadata["document_id"] in supersedes:
            raise MetadataValidationError(f"{path}: supersedes contains duplicate or self references")
        scope = metadata.get("policy_scope", "standard")
        if scope not in {"standard", "customer"}:
            raise MetadataValidationError(f"{path}: policy_scope must be standard or customer")
        customer = metadata.get("applies_to_customer")
        signed = metadata.get("signed", False)
        if scope == "customer" and (not isinstance(customer, str) or not customer or signed is not True):
            raise MetadataValidationError(
                f"{path}: customer policy requires applies_to_customer and signed: true"
            )
        if scope == "standard" and customer is not None:
            raise MetadataValidationError(f"{path}: standard policy cannot name a customer")
    return metadata


def load_document(path: Path, *, require_lifecycle: bool = False) -> Document:
    text = path.read_text(encoding="utf-8")
    parsed = _front_matter(text, path)
    metadata, body = parsed if parsed is not None else _legacy_metadata(text)
    metadata = _validate_metadata(metadata, path, require_lifecycle)
    if not body.strip():
        raise MetadataValidationError(f"{path}: document body must not be empty")
    return Document(
        document_id=metadata["document_id"],
        title=metadata["title"],
        department=metadata["department"],
        version=metadata["version"],
        effective_date=metadata["effective_date"],
        access_roles=tuple(metadata["access_roles"]),
        tags=tuple(metadata["tags"]),
        path=str(path.resolve()),
        text=body,
        effective_from=metadata.get("effective_from", metadata["effective_date"]),
        effective_until=metadata.get("effective_until"),
        supersedes=tuple(metadata.get("supersedes", [])),
        status=metadata.get("status", "active"),
        policy_family=metadata.get("policy_family", metadata["document_id"]),
        policy_scope=metadata.get("policy_scope", "standard"),
        applies_to_customer=metadata.get("applies_to_customer"),
        signed=metadata.get("signed", False),
    )


def load_corpus(documents_dir: Path, *, require_lifecycle: bool = False) -> list[Document]:
    paths = sorted(documents_dir.rglob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No Markdown documents found under {documents_dir}")
    documents = [load_document(path, require_lifecycle=require_lifecycle) for path in paths]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for document in documents:
        if document.document_id in seen:
            duplicates.add(document.document_id)
        seen.add(document.document_id)
    if duplicates:
        raise MetadataValidationError(
            f"Duplicate document_id values: {', '.join(sorted(duplicates))}"
        )
    by_id = {document.document_id: document for document in documents}
    for document in documents:
        for reference in document.supersedes:
            target = by_id.get(reference)
            if target is None:
                raise MetadataValidationError(
                    f"{document.path}: supersedes references unknown document {reference!r}"
                )
            if target.policy_family != document.policy_family:
                raise MetadataValidationError(
                    f"{document.path}: superseded document must share policy_family"
                )
            if date.fromisoformat(target.effective_from) >= date.fromisoformat(document.effective_from):
                raise MetadataValidationError(
                    f"{document.path}: superseded document must start before its replacement"
                )
    def visit(document_id: str, path: set[str]) -> None:
        if document_id in path:
            raise MetadataValidationError(f"Lifecycle supersedes cycle includes {document_id}")
        for parent in by_id[document_id].supersedes:
            visit(parent, path | {document_id})
    for document_id in by_id:
        visit(document_id, set())
    return documents

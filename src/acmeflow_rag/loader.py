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


def _validate_metadata(metadata: dict[str, Any], path: Path) -> dict[str, Any]:
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
    return metadata


def load_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    parsed = _front_matter(text, path)
    metadata, body = parsed if parsed is not None else _legacy_metadata(text)
    metadata = _validate_metadata(metadata, path)
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
    )


def load_corpus(documents_dir: Path) -> list[Document]:
    paths = sorted(documents_dir.rglob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No Markdown documents found under {documents_dir}")
    documents = [load_document(path) for path in paths]
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
    return documents

from pathlib import Path
import json

import pytest

from acmeflow_rag.loader import MetadataValidationError, load_corpus, load_document


VALID_DOCUMENT = """---
document_id: test_policy_2026
title: Test Policy
department: Engineering
version: "1.0"
effective_date: "2026-01-01"
access_roles: [engineering, manager]
tags: [test, policy]
---
# Test Policy

## Rules

The validated body remains available for chunking.
"""


def test_yaml_front_matter_is_loaded_and_validated(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(VALID_DOCUMENT, encoding="utf-8")

    document = load_document(path)

    assert document.document_id == "test_policy_2026"
    assert document.version == "1.0"
    assert document.access_roles == ("engineering", "manager")
    assert document.tags == ("test", "policy")
    assert document.text.startswith("# Test Policy")
    assert document.path == str(path.resolve())


@pytest.mark.parametrize(
    "broken",
    [
        VALID_DOCUMENT.replace("title: Test Policy\n", ""),
        VALID_DOCUMENT.replace('effective_date: "2026-01-01"', 'effective_date: "January 1"'),
        VALID_DOCUMENT.replace("access_roles: [engineering, manager]", "access_roles: engineering"),
        VALID_DOCUMENT.replace("document_id: test_policy_2026", "document_id: Invalid ID"),
        VALID_DOCUMENT.replace("tags: [test, policy]", "tags: []"),
    ],
)
def test_invalid_metadata_is_rejected(tmp_path, broken):
    path = tmp_path / "invalid.md"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(MetadataValidationError):
        load_document(path)


def test_unclosed_front_matter_is_rejected(tmp_path):
    path = tmp_path / "invalid.md"
    path.write_text("---\ndocument_id: broken\n", encoding="utf-8")

    with pytest.raises(MetadataValidationError, match="no closing"):
        load_document(path)


def test_duplicate_document_ids_are_rejected(tmp_path):
    (tmp_path / "one.md").write_text(VALID_DOCUMENT, encoding="utf-8")
    (tmp_path / "two.md").write_text(VALID_DOCUMENT, encoding="utf-8")

    with pytest.raises(MetadataValidationError, match="Duplicate document_id"):
        load_corpus(tmp_path)


def test_v2_corpus_has_validated_front_matter():
    documents = load_corpus(Path("data/documents_v2"))

    assert len(documents) == 10
    assert len({document.document_id for document in documents}) == 10


def test_v2_benchmark_references_existing_documents_and_traceable_facts():
    documents = load_corpus(Path("data/documents_v2"))
    by_id = {document.document_id: document for document in documents}
    benchmark = json.loads(
        Path("data/evaluation/benchmark_v2.json").read_text(encoding="utf-8")
    )

    assert benchmark["case_count"] == len(benchmark["cases"])
    assert len({case["id"] for case in benchmark["cases"]}) == len(benchmark["cases"])
    for case in benchmark["cases"]:
        referenced = case.get("expected_documents", []) + case.get("excluded_documents", [])
        assert set(referenced).issubset(by_id)
        expected_text = " ".join(
            by_id[document_id].text for document_id in case.get("expected_documents", [])
        ).lower()
        for fact in case.get("expected_facts", []):
            assert fact.lower() in expected_text

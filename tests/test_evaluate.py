from dataclasses import replace

import pytest

from acmeflow_rag.evaluate import (
    calculate_abstention_metrics,
    calculate_grounding_metrics,
    calculate_per_category_metrics,
    calculate_retrieval_metrics,
    calculate_temporal_metrics,
    deduplicate_document_ids,
    evaluate_authorization,
    reciprocal_rank,
    wilson_interval,
)
from acmeflow_rag.models import Chunk, RetrievalResult


def make_result(document_id: str, roles=("employee",), score=1.0) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=f"{document_id}::chunk-0",
        document_id=document_id,
        title=document_id,
        department="Test",
        version="1.0",
        effective_date="2026-01-01",
        access_roles=tuple(roles),
        section="General",
        tags=("test",),
        text=f"Evidence from {document_id}",
    )
    return RetrievalResult(chunk=chunk, score=score)


def test_document_ids_are_deduplicated_before_ranking():
    first = make_result("doc-a")
    repeated_chunk = replace(first, chunk=replace(first.chunk, chunk_id="doc-a::chunk-1"))
    results = [first, repeated_chunk, make_result("doc-b")]

    assert deduplicate_document_ids(results, top_k=2) == ["doc-a", "doc-b"]
    assert reciprocal_rank(["doc-a", "doc-b"], ["doc-b"]) == 0.5


def test_retrieval_metrics_cover_empty_and_multiple_required_documents():
    cases = [
        {"id": "empty", "answerable": True, "expected_documents": ["doc-a"], "category": "direct"},
        {"id": "multi", "answerable": True, "expected_documents": ["doc-a", "doc-b"], "category": "multi"},
        {"id": "unanswerable", "answerable": False, "expected_documents": [], "category": "unanswerable"},
    ]
    results = {
        "empty": [],
        "multi": [make_result("doc-a"), make_result("doc-a"), make_result("doc-b")],
        "unanswerable": [make_result("noise")],
    }

    metrics = calculate_retrieval_metrics(cases, results)

    assert metrics["answerable_cases"] == 2
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["full_evidence_success_at_5"] == 0.5
    assert metrics["mean_required_document_coverage_at_5"] == 0.5


def test_abstention_confusion_matrix_and_rates():
    cases = [
        {"id": "a1", "answerable": True},
        {"id": "a2", "answerable": True},
        {"id": "u1", "answerable": False},
        {"id": "u2", "answerable": False},
    ]
    metrics = calculate_abstention_metrics(
        cases, {"a1": False, "a2": True, "u1": True, "u2": False}
    )

    assert metrics["confusion_matrix"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["false_abstention_rate"] == 0.5
    assert metrics["missed_abstention_rate"] == 0.5


def test_abstention_metrics_handle_empty_class_denominators():
    cases = [{"id": "a", "answerable": True}]
    metrics = calculate_abstention_metrics(cases, {"a": False})

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["missed_abstention_rate"] == 0.0


def test_grounding_metrics_separate_citation_structure_from_correctness():
    cases = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    responses = {
        "a": {"abstained": False, "citation_validation": {"valid": True},
              "claims": [{"cited_chunk_ids": ["x"]}]},
        "b": {"abstained": False, "citation_validation": {"valid": False},
              "claims": [{"cited_chunk_ids": []}]},
        "c": {"abstained": True, "claims": []},
    }
    metrics = calculate_grounding_metrics(cases, responses)
    assert metrics["citation_validity_rate"] == 0.5
    assert metrics["claim_evidence_coverage"] == 0.5
    assert "correctness" in metrics["scope"]
    low, high = wilson_interval(1, 2)
    assert low < 0.5 < high


def test_temporal_metrics_separate_current_and_historical_questions():
    cases = [
        {
            "id": "current",
            "category": "temporal_conflict",
            "answerable": True,
            "question": "What is the current policy?",
            "expected_documents": ["policy-2026"],
            "excluded_documents": ["policy-2025"],
        },
        {
            "id": "historical",
            "category": "temporal_conflict",
            "answerable": True,
            "question": "What was the policy before 2026?",
            "expected_documents": ["policy-2025"],
        },
    ]
    results = {
        "current": [make_result("policy-2026"), make_result("policy-2025")],
        "historical": [make_result("policy-2025")],
    }

    metrics = calculate_temporal_metrics(cases, results)

    assert metrics["current_policy_cases"] == 1
    assert metrics["current_policy_cases_with_excluded_document_at_5"] == 1
    assert metrics["current_policy_excluded_document_case_rate_at_5"] == 1.0
    assert metrics["explicit_historical_cases"] == 1
    assert metrics["historical_full_evidence_success_at_5"] == 1.0


def test_per_category_metrics_include_unanswerable_cases_without_retrieval_division_errors():
    cases = [
        {"id": "a", "category": "lookup", "answerable": True, "expected_documents": ["doc-a"]},
        {"id": "u", "category": "unanswerable", "answerable": False, "expected_documents": []},
    ]
    metrics = calculate_per_category_metrics(
        cases,
        {"a": [make_result("doc-a")], "u": []},
        {"a": False, "u": True},
    )

    assert metrics["lookup"]["retrieval"]["recall_at_5"] == 1.0
    assert metrics["unanswerable"]["retrieval"]["answerable_cases"] == 0
    assert metrics["unanswerable"]["abstention"]["recall"] == 1.0


class FilteringRetriever:
    def __init__(self):
        self.chunks = [
            make_result("support-private", ("support",)).chunk,
            make_result("employee-public", ("employee",)).chunk,
        ]
        self.seen_roles = []

    def search(self, query, top_k=5, allowed_roles=None, department=None):
        self.seen_roles.append(set(allowed_roles or set()))
        return [
            RetrievalResult(chunk, 1.0)
            for chunk in self.chunks
            if allowed_roles is None or set(chunk.access_roles).intersection(allowed_roles)
        ][:top_k]


def test_authorization_maps_symbolic_label_and_excludes_unauthorized_document():
    retriever = FilteringRetriever()
    cases = [{
        "id": "auth",
        "category": "authorization",
        "answerable": True,
        "question": "support question",
        "expected_documents": ["support-private"],
        "required_role": "support",
    }]

    metrics = evaluate_authorization(cases, retriever)

    assert {"support"} in retriever.seen_roles
    assert {"employee"} in retriever.seen_roles
    assert {"support"} != {"support-agent"}
    assert metrics["authorized_full_accessible_evidence_success_at_5"] == 1.0
    assert metrics["unauthorized_expected_document_leak_trials"] == 0
    assert metrics["unauthorized_expected_document_exclusion_rate"] == 1.0
    assert metrics["role_filter_result_violations"] == 0


def test_authorization_reports_unmapped_symbolic_labels():
    metrics = evaluate_authorization(
        [{
            "id": "auth",
            "question": "question",
            "expected_documents": [],
            "required_role": "unknown_symbolic_label",
        }],
        FilteringRetriever(),
    )

    assert metrics["authorized_trials"] == 0
    assert metrics["unauthorized_expected_document_exclusion_rate"] == 0.0
    assert metrics["unmapped_required_role_labels"] == ["unknown_symbolic_label"]

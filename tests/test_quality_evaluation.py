import json
from pathlib import Path

import pytest

from acmeflow_rag.demo import format_response
from acmeflow_rag.loader import load_corpus
from acmeflow_rag.quality_metrics import (
    fact_matches,
    failure_categories,
    score_claim_entailment,
    score_factual_correctness,
    validate_quality_benchmark,
)


def test_fact_matching_requires_one_complete_explicit_token_set():
    fact = {"acceptable_token_sets": [["14", "calendar", "days"], ["fourteen", "days"]]}
    assert fact_matches("The policy allows 14 calendar days.", fact)
    assert fact_matches("It allows fourteen days.", fact)
    assert not fact_matches("It allows 14 days.", fact)


def test_factual_correctness_is_fact_coverage_not_citation_validity():
    cases = [{
        "id": "a", "answerable": True,
        "expected_facts": [
            {"id": "window", "acceptable_token_sets": [["14", "days"]]},
            {"id": "timing", "acceptable_token_sets": [["3", "business", "days"]]},
        ],
    }]
    responses = {"a": {"answer": "Requests are allowed for 14 days.", "abstained": False}}
    score = score_factual_correctness(cases, responses)
    assert score["matched_facts"] == 1
    assert score["expected_fact_coverage"] == 0.5
    assert score["complete_fact_case_rate"] == 0.0


def test_unanswerable_case_is_excluded_from_fact_denominator():
    cases = [{"id": "u", "answerable": False, "expected_facts": []}]
    score = score_factual_correctness(cases, {"u": {"answer": "", "abstained": True}})
    assert score["scored_cases"] == 0
    assert score["expected_fact_coverage"] == 0.0


def test_claim_support_requires_claim_text_inside_a_cited_passage():
    responses = {
        "case": {
            "abstained": False,
            "evidence_passages": [
                {"chunk_id": "c1", "text": "Refunds are available for 14 calendar days."},
                {"chunk_id": "c2", "text": "Support verifies the account."},
            ],
            "claims": [
                {"claim_id": "ok", "text": "Refunds are available for 14 calendar days.", "cited_chunk_ids": ["c1"]},
                {"claim_id": "wrong-citation", "text": "Support verifies the account.", "cited_chunk_ids": ["c1"]},
                {"claim_id": "uncited", "text": "Anything", "cited_chunk_ids": []},
            ],
        }
    }
    score = score_claim_entailment(responses)
    assert score["supported_claims"] == 1
    assert score["claims"] == 3
    assert score["support_rate"] == pytest.approx(1 / 3)


def test_abstentions_have_no_claims_in_entailment_denominator():
    score = score_claim_entailment({"u": {"abstained": True, "claims": []}})
    assert score["claims"] == 0
    assert score["support_rate"] == 1.0


def test_benchmark_review_claim_requires_reviewer_and_date():
    payload = {
        "review_provenance": {"independent_review_status": "complete"},
        "cases": [{
            "id": "x", "question": "q", "category": "c", "answerable": False,
            "identity_token": "token", "as_of": "2026-01-01",
        }],
    }
    with pytest.raises(ValueError, match="reviewer and date"):
        validate_quality_benchmark(payload)
    payload["review_provenance"]["independent_review_status"] = "pending"
    validate_quality_benchmark(payload)


def test_benchmark_rejects_duplicate_ids_and_answerable_without_documents():
    base = {
        "review_provenance": {"independent_review_status": "pending"},
        "cases": [{
            "id": "x", "question": "q", "category": "c", "answerable": True,
            "identity_token": "token", "as_of": "2026-01-01", "expected_documents": [],
        }],
    }
    with pytest.raises(ValueError, match="expected documents"):
        validate_quality_benchmark(base)
    base["cases"][0]["answerable"] = False
    base["cases"].append(dict(base["cases"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_quality_benchmark(base)


def test_failure_categories_separate_quality_and_security_failures():
    cases = [{
        "id": "x", "answerable": True, "forbidden_documents": ["secret"]
    }]
    responses = {"x": {
        "abstained": False, "source_document_ids": ["secret"],
        "citation_validation": {"valid": False},
    }}
    factual = {"per_case": {"x": {"coverage": 0.5}}}
    entailment = {"per_case": {"x": {"claim": False}}}
    failures = failure_categories(cases, responses, factual, entailment)
    assert failures == {
        "forbidden_document_leakage": 1,
        "incomplete_expected_facts": 1,
        "invalid_citation": 1,
        "unsupported_extractive_claim": 1,
    }


def test_demo_format_shows_abstention_and_empty_sources():
    output = format_response("Unknown?", {
        "abstained": True,
        "reason_code": "INSUFFICIENT_EVIDENCE",
        "answer": "No answer.",
        "evidence_passages": [],
        "claims": [],
    })
    assert "Status: ABSTAINED" in output
    assert "Reason: INSUFFICIENT_EVIDENCE" in output
    assert "Sources:\n  (none)" in output


def test_quality_benchmark_is_valid_and_covers_required_scenarios():
    path = Path(__file__).parents[1] / "data/evaluation/quality_v3.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_quality_benchmark(payload)
    categories = {case["category"] for case in payload["cases"]}
    assert {"multi_document", "historical_policy", "authorization_denial", "unanswerable"} <= categories
    assert payload["review_provenance"]["independent_review_status"] == "pending"
    corpus_ids = {
        document.document_id
        for document in load_corpus(path.parents[1] / "documents_v3", require_lifecycle=True)
    }
    referenced_ids = {
        document_id
        for case in payload["cases"]
        for document_id in case.get("expected_documents", []) + case.get("forbidden_documents", [])
    }
    assert referenced_ids <= corpus_ids

"""Transparent, deterministic answer-quality proxies.

These metrics deliberately avoid calling an LLM.  They are useful regression signals,
not substitutes for independent human review or semantic entailment judging.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from typing import Any


_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize_text(value: str) -> str:
    return _SPACE.sub(" ", value.casefold()).strip()


def _tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(value.casefold()))


def fact_matches(answer: str, fact: dict[str, Any]) -> bool:
    """Match one manually authored fact using explicit acceptable token groups."""

    answer_tokens = _tokens(answer)
    alternatives = fact.get("acceptable_token_sets", [])
    return bool(alternatives) and any(
        set(map(str.casefold, alternative)).issubset(answer_tokens)
        for alternative in alternatives
    )


def score_factual_correctness(
    cases: list[dict[str, Any]], responses: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Calculate lexical expected-fact coverage for answerable benchmark cases."""

    matched = total = complete = 0
    details: dict[str, Any] = {}
    scored_cases = 0
    for case in cases:
        facts = case.get("expected_facts", []) if case.get("answerable") else []
        if not facts:
            continue
        scored_cases += 1
        answer = responses[case["id"]].get("answer", "")
        fact_results = [fact_matches(answer, fact) for fact in facts]
        hit_count = sum(fact_results)
        matched += hit_count
        total += len(facts)
        complete += int(hit_count == len(facts))
        details[case["id"]] = {
            "matched_facts": hit_count,
            "expected_facts": len(facts),
            "coverage": hit_count / len(facts),
            "matches": {
                fact["id"]: result for fact, result in zip(facts, fact_results)
            },
        }
    return {
        "method": "automated lexical expected-fact coverage",
        "judgment_source": "project-authored reference facts; independent review pending",
        "scored_cases": scored_cases,
        "matched_facts": matched,
        "expected_facts": total,
        "expected_fact_coverage": matched / total if total else 0.0,
        "complete_fact_case_rate": complete / scored_cases if scored_cases else 0.0,
        "per_case": details,
        "limitation": (
            "Token presence is neither a semantic correctness nor contradiction check; "
            "paraphrases can be missed and misleading answers can score well."
        ),
    }


def score_claim_entailment(responses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check whether each claim is an extractive substring of its cited evidence."""

    supported = total = 0
    per_case: dict[str, Any] = {}
    for case_id, response in responses.items():
        if response.get("abstained"):
            continue
        evidence = {
            item["chunk_id"]: normalize_text(item.get("text", ""))
            for item in response.get("evidence_passages", [])
        }
        outcomes: dict[str, bool] = {}
        for claim in response.get("claims", []):
            total += 1
            claim_text = normalize_text(claim.get("text", ""))
            citations = claim.get("cited_chunk_ids", [])
            outcome = bool(claim_text) and any(
                claim_text in evidence.get(chunk_id, "") for chunk_id in citations
            )
            supported += int(outcome)
            outcomes[claim.get("claim_id", f"claim-{total}")] = outcome
        per_case[case_id] = outcomes
    return {
        "method": "automated extractive claim-support proxy",
        "supported_claims": supported,
        "claims": total,
        "support_rate": supported / total if total else 1.0,
        "per_case": per_case,
        "limitation": (
            "Substring support confirms an extractive claim occurs in cited context; it does "
            "not establish semantic entailment, truth, completeness, or absence of contradiction."
        ),
    }


def validate_quality_benchmark(payload: dict[str, Any]) -> None:
    review = payload.get("review_provenance", {})
    if review.get("independent_review_status") not in {"pending", "complete"}:
        raise ValueError("Benchmark must declare independent_review_status")
    if review.get("independent_review_status") == "complete" and not all(
        review.get(key) for key in ("independent_reviewer", "reviewed_at")
    ):
        raise ValueError("A completed independent review requires reviewer and date")
    ids = [case.get("id") for case in payload.get("cases", [])]
    if not ids or None in ids or len(ids) != len(set(ids)):
        raise ValueError("Benchmark cases require unique non-empty IDs")
    for case in payload["cases"]:
        required = {"question", "category", "answerable", "identity_token", "as_of"}
        if required.difference(case):
            raise ValueError(f"Case {case['id']} is missing required fields")
        if case["answerable"] and not case.get("expected_documents"):
            raise ValueError(f"Answerable case {case['id']} needs expected documents")
        try:
            date.fromisoformat(case["as_of"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Case {case['id']} has an invalid as_of date") from exc
        for fact in case.get("expected_facts", []):
            if (
                not fact.get("id")
                or not fact.get("acceptable_token_sets")
                or fact.get("evidence_document") not in case.get("expected_documents", [])
            ):
                raise ValueError(f"Case {case['id']} has an invalid expected fact")


def failure_categories(
    cases: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    factual: dict[str, Any],
    entailment: dict[str, Any],
) -> dict[str, int]:
    failures: Counter[str] = Counter()
    for case in cases:
        response = responses[case["id"]]
        if case["answerable"] and response.get("abstained"):
            failures["false_abstention"] += 1
        if not case["answerable"] and not response.get("abstained"):
            failures["missed_abstention"] += 1
        detail = factual.get("per_case", {}).get(case["id"])
        if detail and detail["coverage"] < 1:
            failures["incomplete_expected_facts"] += 1
        if not response.get("abstained") and not response.get("citation_validation", {}).get("valid", False):
            failures["invalid_citation"] += 1
        if any(not value for value in entailment.get("per_case", {}).get(case["id"], {}).values()):
            failures["unsupported_extractive_claim"] += 1
        forbidden = set(case.get("forbidden_documents", []))
        if forbidden.intersection(response.get("source_document_ids", [])):
            failures["forbidden_document_leakage"] += 1
    return dict(sorted(failures.items()))

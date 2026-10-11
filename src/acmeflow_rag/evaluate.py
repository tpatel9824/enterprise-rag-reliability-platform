"""Offline evaluation for the Phase 1 retrieval and abstention baseline.

The module keeps metric calculation separate from loading the production artifacts so the
calculations can be tested with small, synthetic fixtures.
"""

from __future__ import annotations

import json
import re
import argparse
import time
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import settings
from .auth import Principal as AuthPrincipal
from .pipeline import RetrievalPipelineConfig, load_pipeline, normalize_strategy
from .retrieval import Retriever
from .service import RagService


DEFAULT_TOP_K = 5
HISTORICAL_TERMS = re.compile(r"\b(before|previous|historical|prior|old|formerly)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Principal:
    """A named evaluation fixture, not an authenticated application identity."""

    name: str
    roles: frozenset[str]


# Benchmark labels describe scenarios, while these sets are roles understood by the retriever.
# Keeping the mapping explicit prevents symbolic labels such as "security_or_engineering" from
# accidentally being treated as real access roles.
AUTHORIZATION_PRINCIPALS: dict[str, dict[str, tuple[Principal, ...]]] = {
    "support": {
        "authorized": (Principal("support-agent", frozenset({"support"})),),
        "unauthorized": (Principal("ordinary-employee", frozenset({"employee"})),),
    },
    "security_or_engineering": {
        "authorized": (
            Principal("security-engineer", frozenset({"security"})),
            Principal("software-engineer", frozenset({"engineering"})),
        ),
        "unauthorized": (Principal("sales-user", frozenset({"sales"})),),
    },
}

EVALUATION_PRINCIPAL = AuthPrincipal(
    "offline-evaluator",
    frozenset({"employee", "support", "sales", "finance", "product", "manager", "executive", "security", "engineering", "it"}),
)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def deduplicate_document_ids(results: Iterable[Any], top_k: int | None = None) -> list[str]:
    """Return ranked, unique document IDs; repeated chunks do not consume document K."""

    ids: list[str] = []
    seen: set[str] = set()
    for result in results:
        document_id = result.chunk.document_id
        if document_id in seen:
            continue
        seen.add(document_id)
        ids.append(document_id)
        if top_k is not None and len(ids) >= top_k:
            break
    return ids


def reciprocal_rank(document_ids: Iterable[str], expected_documents: Iterable[str]) -> float:
    wanted = set(expected_documents)
    for rank, document_id in enumerate(document_ids, 1):
        if document_id in wanted:
            return 1.0 / rank
    return 0.0


def calculate_retrieval_metrics(
    cases: list[dict[str, Any]],
    results_by_case: dict[str, Iterable[Any]],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Calculate document-level retrieval metrics for answerable cases.

    Recall@K retains the original "any expected document" definition. Full-evidence success
    requires every distinct expected document, and coverage is the fraction retrieved.
    """

    answerable = [case for case in cases if case["answerable"]]
    recall_at_1 = recall_at_k = full_success = 0
    mrr = coverage_sum = 0.0

    for case in answerable:
        ids = deduplicate_document_ids(results_by_case.get(case["id"], []), top_k)
        expected = set(case.get("expected_documents", []))
        found = expected.intersection(ids)
        recall_at_1 += int(bool(expected.intersection(ids[:1])))
        recall_at_k += int(bool(found))
        full_success += int(bool(expected) and expected.issubset(ids))
        mrr += reciprocal_rank(ids, expected)
        coverage_sum += safe_divide(len(found), len(expected))

    count = len(answerable)
    return {
        "answerable_cases": count,
        "recall_at_1": safe_divide(recall_at_1, count),
        "recall_at_5": safe_divide(recall_at_k, count),
        "mrr": safe_divide(mrr, count),
        "any_document_success_at_5": safe_divide(recall_at_k, count),
        "full_evidence_success_at_5": safe_divide(full_success, count),
        "mean_required_document_coverage_at_5": safe_divide(coverage_sum, count),
    }


def calculate_abstention_metrics(
    cases: list[dict[str, Any]], decisions_by_case: dict[str, bool]
) -> dict[str, Any]:
    """Treat 'should abstain' as the positive class and calculate its confusion matrix."""

    tp = fp = tn = fn = 0
    for case in cases:
        should_abstain = not case["answerable"]
        did_abstain = decisions_by_case[case["id"]]
        if should_abstain and did_abstain:
            tp += 1
        elif not should_abstain and did_abstain:
            fp += 1
        elif should_abstain and not did_abstain:
            fn += 1
        else:
            tn += 1

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "positive_class": "should_abstain (benchmark answerable=false)",
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "precision": precision,
        "precision_wilson_95": wilson_interval(tp, tp + fp),
        "recall": recall,
        "recall_wilson_95": wilson_interval(tp, tp + fn),
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "false_abstention_rate": safe_divide(fp, fp + tn),
        "false_abstention_rate_wilson_95": wilson_interval(fp, fp + tn),
        "missed_abstention_rate": safe_divide(fn, fn + tp),
        "missed_abstention_rate_wilson_95": wilson_interval(fn, fn + tp),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Approximate 95% Wilson interval for a binomial proportion."""

    if not total:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def calculate_grounding_metrics(
    cases: list[dict[str, Any]], responses_by_case: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Measure citation structure only; this does not score correctness or entailment."""

    answered = [responses_by_case[case["id"]] for case in cases if not responses_by_case[case["id"]]["abstained"]]
    valid = sum(bool(response.get("citation_validation", {}).get("valid")) for response in answered)
    total_claims = sum(len(response.get("claims", [])) for response in answered)
    cited_claims = sum(
        bool(claim.get("cited_chunk_ids"))
        for response in answered
        for claim in response.get("claims", [])
    )
    return {
        "answered_responses": len(answered),
        "responses_with_valid_citations": valid,
        "citation_validity_rate": safe_divide(valid, len(answered)),
        "citation_validity_wilson_95": wilson_interval(valid, len(answered)),
        "factual_claims": total_claims,
        "claims_with_evidence": cited_claims,
        "claim_evidence_coverage": safe_divide(cited_claims, total_claims),
        "scope": "Checks citation existence and authorized-context chunk IDs only; factual correctness and entailment are not measured.",
    }


def is_explicit_historical_question(case: dict[str, Any]) -> bool:
    return case.get("category") == "temporal_conflict" and bool(
        HISTORICAL_TERMS.search(case["question"])
    )


def calculate_temporal_metrics(
    cases: list[dict[str, Any]],
    results_by_case: dict[str, Iterable[Any]],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Separate current-policy exclusion checks from explicit historical lookups."""

    current_cases = [
        case for case in cases if case.get("excluded_documents") and not is_explicit_historical_question(case)
    ]
    historical_cases = [case for case in cases if is_explicit_historical_question(case)]
    conflict_cases = conflict_documents = historical_any = historical_full = 0

    for case in current_cases:
        ids = set(deduplicate_document_ids(results_by_case.get(case["id"], []), top_k))
        conflicts = ids.intersection(case["excluded_documents"])
        conflict_cases += int(bool(conflicts))
        conflict_documents += len(conflicts)

    for case in historical_cases:
        ids = set(deduplicate_document_ids(results_by_case.get(case["id"], []), top_k))
        expected = set(case.get("expected_documents", []))
        historical_any += int(bool(ids.intersection(expected)))
        historical_full += int(bool(expected) and expected.issubset(ids))

    return {
        "current_policy_cases": len(current_cases),
        "current_policy_cases_with_excluded_document_at_5": conflict_cases,
        "current_policy_excluded_document_case_rate_at_5": safe_divide(conflict_cases, len(current_cases)),
        "distinct_excluded_document_hits_at_5": conflict_documents,
        "explicit_historical_cases": len(historical_cases),
        "historical_any_document_success_at_5": safe_divide(historical_any, len(historical_cases)),
        "historical_full_evidence_success_at_5": safe_divide(historical_full, len(historical_cases)),
    }


def calculate_per_category_metrics(
    cases: list[dict[str, Any]],
    results_by_case: dict[str, Iterable[Any]],
    decisions_by_case: dict[str, bool],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["category"]].append(case)

    output: dict[str, Any] = {}
    for category, category_cases in sorted(grouped.items()):
        category_results = {case["id"]: results_by_case.get(case["id"], []) for case in category_cases}
        category_decisions = {case["id"]: decisions_by_case[case["id"]] for case in category_cases}
        output[category] = {
            "case_count": len(category_cases),
            "retrieval": calculate_retrieval_metrics(category_cases, category_results, top_k),
            "abstention": calculate_abstention_metrics(category_cases, category_decisions),
        }
    return output


def evaluate_authorization(
    cases: list[dict[str, Any]], retriever: Retriever, top_k: int = DEFAULT_TOP_K
) -> dict[str, Any]:
    """Evaluate retrieval filters with declared role fixtures, not authentication or RBAC."""

    acl_by_document: dict[str, set[str]] = defaultdict(set)
    for chunk in retriever.chunks:
        acl_by_document[chunk.document_id].update(chunk.access_roles)

    authorized_trials = authorized_any = authorized_full = 0
    unauthorized_trials = unauthorized_leaks = filter_violations = 0
    unmapped_labels: set[str] = set()
    trial_details: list[dict[str, Any]] = []

    for case in (case for case in cases if case.get("required_role")):
        label = case["required_role"]
        fixtures = AUTHORIZATION_PRINCIPALS.get(label)
        if fixtures is None:
            unmapped_labels.add(label)
            continue

        for principal in fixtures["authorized"]:
            results = retriever.search(case["question"], top_k, allowed_roles=set(principal.roles))
            ids = set(deduplicate_document_ids(results, top_k))
            accessible_expected = {
                document_id
                for document_id in case["expected_documents"]
                if principal.roles.intersection(acl_by_document.get(document_id, set()))
            }
            authorized_trials += 1
            authorized_any += int(bool(ids.intersection(accessible_expected)))
            authorized_full += int(bool(accessible_expected) and accessible_expected.issubset(ids))
            trial_details.append({
                "case_id": case["id"], "principal": principal.name, "kind": "authorized",
                "roles": sorted(principal.roles), "accessible_expected_documents": sorted(accessible_expected),
            })

        for principal in fixtures["unauthorized"]:
            results = retriever.search(case["question"], top_k, allowed_roles=set(principal.roles))
            ids = set(deduplicate_document_ids(results, top_k))
            protected_expected = {
                document_id
                for document_id in case["expected_documents"]
                if not principal.roles.intersection(acl_by_document.get(document_id, set()))
            }
            leaked = ids.intersection(protected_expected)
            violations = [
                result.chunk.document_id
                for result in results
                if not principal.roles.intersection(result.chunk.access_roles)
            ]
            unauthorized_trials += 1
            unauthorized_leaks += int(bool(leaked))
            filter_violations += len(violations)
            trial_details.append({
                "case_id": case["id"], "principal": principal.name, "kind": "unauthorized",
                "roles": sorted(principal.roles), "protected_expected_documents": sorted(protected_expected),
                "leaked_expected_documents": sorted(leaked),
            })

    return {
        "scope": "retrieval filtering with synthetic principals; not authentication or secure RBAC",
        "authorized_trials": authorized_trials,
        "authorized_any_accessible_document_success_at_5": safe_divide(authorized_any, authorized_trials),
        "authorized_full_accessible_evidence_success_at_5": safe_divide(authorized_full, authorized_trials),
        "unauthorized_trials": unauthorized_trials,
        "unauthorized_expected_document_leak_trials": unauthorized_leaks,
        "unauthorized_expected_document_exclusion_rate": safe_divide(
            unauthorized_trials - unauthorized_leaks, unauthorized_trials
        ),
        "role_filter_result_violations": filter_violations,
        "unmapped_required_role_labels": sorted(unmapped_labels),
        "trials": trial_details,
    }


def evaluate_benchmark(
    cases: list[dict[str, Any]],
    retriever: Retriever,
    service: RagService,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    # One unmeasured query removes one-time lazy model initialization from per-query latency.
    retriever.search("evaluation warmup", top_k=1)
    results_by_case: dict[str, Iterable[Any]] = {}
    latencies_ms: list[float] = []
    for case in cases:
        started = time.perf_counter()
        results_by_case[case["id"]] = retriever.search(case["question"], top_k=top_k)
        latencies_ms.append((time.perf_counter() - started) * 1000)
    responses_by_case = {
        case["id"]: service.ask(case["question"], top_k=top_k, principal=EVALUATION_PRINCIPAL)
        for case in cases
    }
    decisions_by_case = {
        case_id: bool(response["abstained"])
        for case_id, response in responses_by_case.items()
    }
    retrieval = calculate_retrieval_metrics(cases, results_by_case, top_k)
    unanswerable = [case for case in cases if not case["answerable"]]
    unanswerable_nonempty = sum(
        bool(deduplicate_document_ids(results_by_case[case["id"]], top_k)) for case in unanswerable
    )

    ordered_latency = sorted(latencies_ms)
    percentile = lambda fraction: ordered_latency[round((len(ordered_latency) - 1) * fraction)]
    multi_document_cases = [case for case in cases if case["category"] == "multi_document"]
    multi_document_results = {
        case["id"]: results_by_case[case["id"]] for case in multi_document_cases
    }
    index_metadata = getattr(retriever, "index_metadata", None)
    retriever_metadata = index_metadata or {
        "retriever_type": "tfidf",
        "index_type": "TfidfVectorizer with exact cosine similarity",
    }

    return {
        "retriever_metadata": retriever_metadata,
        "benchmark_cases": len(cases),
        "answerable_cases": retrieval["answerable_cases"],
        "unanswerable_cases": len(unanswerable),
        # Original keys are retained for before/after comparisons.
        "recall_at_1": retrieval["recall_at_1"],
        "recall_at_5": retrieval["recall_at_5"],
        "mrr": retrieval["mrr"],
        "unanswerable_retrieval_rate": safe_divide(unanswerable_nonempty, len(unanswerable)),
        "retrieval": retrieval,
        "multi_document_retrieval": calculate_retrieval_metrics(
            multi_document_cases, multi_document_results, top_k
        ),
        "latency_ms": {
            "scope": "warm-query retrieval only; excludes index loading, model loading, generation, and authorization trials",
            "query_count": len(latencies_ms),
            "mean": safe_divide(sum(latencies_ms), len(latencies_ms)),
            "min": ordered_latency[0] if ordered_latency else 0.0,
            "p50": percentile(0.50) if ordered_latency else 0.0,
            "p95": percentile(0.95) if ordered_latency else 0.0,
            "max": ordered_latency[-1] if ordered_latency else 0.0,
        },
        "temporal": calculate_temporal_metrics(cases, results_by_case, top_k),
        "abstention": calculate_abstention_metrics(cases, decisions_by_case),
        "grounding": calculate_grounding_metrics(cases, responses_by_case),
        "authorization": evaluate_authorization(cases, retriever, top_k),
        "per_category": calculate_per_category_metrics(cases, results_by_case, decisions_by_case, top_k),
        "definitions": {
            "document_ranking": "Ranked chunks are deduplicated by document ID before top-K metrics.",
            "recall_at_k": "Fraction of answerable cases with any expected document in the first K distinct documents.",
            "full_evidence_success_at_k": "Fraction of answerable cases with every expected document in the first K distinct documents.",
            "mean_required_document_coverage_at_k": "Mean fraction of distinct expected documents found in the first K distinct documents.",
            "abstention_positive_class": "A benchmark case marked answerable=false.",
            "false_abstention_rate": "Answerable cases incorrectly abstained / all answerable cases.",
            "missed_abstention_rate": "Unanswerable cases incorrectly answered / all unanswerable cases.",
            "citation_validity": "Answered responses whose claims all cite chunk IDs present in their authorized retrieved context.",
            "claim_evidence_coverage": "Claims containing at least one citation / all generated claims; not an entailment metric.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a configured retrieval pipeline")
    parser.add_argument("--benchmark", type=Path, default=settings.benchmark_path)
    parser.add_argument("--calibration-set", type=Path, default=None,
                        help="Optional labeled development set used to document policy calibration")
    parser.add_argument("--strategy", "--retriever", dest="strategy",
                        choices=("lexical", "tfidf", "dense", "hybrid", "hybrid_rerank"),
                        default="lexical")
    parser.add_argument("--artifacts-dir", type=Path, default=None,
                        help="Backward-compatible component artifact directory")
    parser.add_argument("--lexical-artifacts-dir", type=Path, default=settings.lexical_artifacts_dir)
    parser.add_argument("--dense-artifacts-dir", type=Path, default=settings.dense_artifacts_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-k", type=int, default=settings.retrieval_candidate_k)
    parser.add_argument("--rrf-constant", type=int, default=settings.rrf_constant)
    parser.add_argument("--reranker-model", default=settings.reranker_model)
    parser.add_argument("--reranker-revision", default=settings.reranker_revision)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    strategy = normalize_strategy(args.strategy)
    lexical_dir = args.lexical_artifacts_dir
    dense_dir = args.dense_artifacts_dir
    if args.artifacts_dir is not None:
        if strategy == "lexical":
            lexical_dir = args.artifacts_dir
        elif strategy == "dense":
            dense_dir = args.artifacts_dir
    config = RetrievalPipelineConfig(
        strategy=strategy,
        candidate_k=args.candidate_k,
        rrf_constant=args.rrf_constant,
        reranker_model=args.reranker_model,
        reranker_revision=args.reranker_revision,
    )
    retriever = load_pipeline(
        config, lexical_artifacts_dir=lexical_dir, dense_artifacts_dir=dense_dir
    )
    service = RagService(retriever=retriever, retriever_type=strategy)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    metrics = evaluate_benchmark(benchmark["cases"], retriever, service)
    if args.calibration_set:
        calibration = json.loads(args.calibration_set.read_text(encoding="utf-8"))
        evaluation_ids = {case["id"] for case in benchmark["cases"]}
        calibration_ids = {case["id"] for case in calibration["cases"]}
        overlap = sorted(evaluation_ids.intersection(calibration_ids))
        if overlap:
            raise ValueError(f"Calibration and evaluation case IDs overlap: {overlap}")
        metrics["dataset_split"] = {
            "calibration_path": str(args.calibration_set),
            "calibration_cases": len(calibration["cases"]),
            "held_out_evaluation_path": str(args.benchmark),
            "held_out_evaluation_cases": len(benchmark["cases"]),
            "overlapping_case_ids": overlap,
            "policy_selection": "Evidence policy thresholds were chosen using the development calibration set, not held-out labels.",
        }
    output_dir = args.output_dir or args.artifacts_dir or settings.artifacts_dir / "evaluation" / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "pipeline_config.json")
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

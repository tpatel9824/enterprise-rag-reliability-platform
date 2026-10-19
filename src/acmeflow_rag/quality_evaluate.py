"""Offline end-to-end quality evaluation and comparison reporting."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .auth import LocalIdentityProvider
from .config import ROOT, settings
from .evaluate import (
    calculate_abstention_metrics,
    calculate_grounding_metrics,
    calculate_retrieval_metrics,
)
from .lifecycle import select_policies
from .pipeline import RetrievalPipelineConfig, load_pipeline
from .quality_metrics import (
    failure_categories,
    score_claim_entailment,
    score_factual_correctness,
    validate_quality_benchmark,
)
from .service import RagService


STRATEGIES = ("lexical", "dense", "hybrid", "hybrid_rerank")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(values: list[float], scope: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "samples": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def evaluate_strategy(
    strategy: str,
    cases: list[dict[str, Any]],
    retriever,
    identity_provider: LocalIdentityProvider,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval and full offline responses with real policy/auth filtering."""

    service = RagService(retriever=retriever, retriever_type=strategy)
    retrievals: dict[str, Any] = {}
    responses: dict[str, dict[str, Any]] = {}
    retrieval_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    decisions: dict[str, bool] = {}
    cases_output: dict[str, Any] = {}

    # One unmeasured query initializes lazy model components and backend kernels. This keeps
    # model construction/startup out of every strategy's reported warm-query latency.
    if cases:
        warmup = cases[0]
        warmup_principal = identity_provider.authenticate(warmup["identity_token"])
        warmup_selection = select_policies(
            retriever.chunks,
            as_of=date.fromisoformat(warmup["as_of"]),
            customer_id=warmup_principal.customer_id,
        )
        retriever.search(
            warmup["question"],
            top_k=top_k,
            allowed_roles=set(warmup_principal.roles),
            department=warmup.get("department"),
            eligible_document_ids=set(warmup_selection.eligible_document_ids),
        )

    for case in cases:
        principal = identity_provider.authenticate(case["identity_token"])
        as_of = date.fromisoformat(case["as_of"])
        selection = select_policies(
            retriever.chunks, as_of=as_of, customer_id=principal.customer_id
        )
        started = time.perf_counter()
        results = retriever.search(
            case["question"],
            top_k=top_k,
            allowed_roles=set(principal.roles),
            department=case.get("department"),
            eligible_document_ids=set(selection.eligible_document_ids),
        )
        retrieval_latencies.append((time.perf_counter() - started) * 1000)
        retrievals[case["id"]] = results

        started = time.perf_counter()
        response = service.ask(
            case["question"], top_k, principal, case.get("department"), as_of
        )
        end_to_end_latencies.append((time.perf_counter() - started) * 1000)
        responses[case["id"]] = response
        decisions[case["id"]] = bool(response["abstained"])
        cases_output[case["id"]] = {
            "category": case["category"],
            "expected_answerable": case["answerable"],
            "actual_abstained": response["abstained"],
            "reason_code": response["reason_code"],
            "source_document_ids": response["source_document_ids"],
            "answer": response["answer"],
        }

    factual = score_factual_correctness(cases, responses)
    entailment = score_claim_entailment(responses)
    grounding = calculate_grounding_metrics(cases, responses)
    forbidden_hits = sum(
        bool(set(case.get("forbidden_documents", [])).intersection(
            responses[case["id"]].get("source_document_ids", [])
        ))
        for case in cases
    )
    per_category: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["category"]].append(case)
    for category, category_cases in sorted(grouped.items()):
        per_category[category] = {
            "cases": len(category_cases),
            "retrieval": calculate_retrieval_metrics(category_cases, retrievals, top_k),
            "abstention": calculate_abstention_metrics(category_cases, decisions),
        }

    return {
        "strategy": strategy,
        "cases": len(cases),
        "retrieval": calculate_retrieval_metrics(cases, retrievals, top_k),
        "abstention": calculate_abstention_metrics(cases, decisions),
        "citation_structure": grounding,
        "factual_correctness_proxy": factual,
        "claim_evidence_entailment_proxy": entailment,
        "authorization_and_versioning": {
            "forbidden_document_hits": forbidden_hits,
            "violation_rate": forbidden_hits / len(cases) if cases else 0.0,
            "definition": "A forbidden/superseded/unauthorized document appeared in returned sources.",
        },
        "latency_ms": {
            "retrieval": latency_summary(
                retrieval_latencies,
                "warm retrieval call including query embedding/reranking; excludes one unmeasured warmup and model/index startup",
            ),
            "end_to_end": latency_summary(
                end_to_end_latencies,
                "warm local service call including policy selection, retrieval, assessment, and offline generation",
            ),
        },
        "failure_categories": failure_categories(cases, responses, factual, entailment),
        "per_category": per_category,
        "case_results": cases_output,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    review = payload["benchmark"]["review_provenance"]
    lines = [
        "# Offline quality evaluation",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "> **Review status:** Independent benchmark review is "
        f"**{review['independent_review_status']}**. Reference labels are project-authored; "
        "automated scores are regression proxies, not human correctness judgments.",
        "",
        "## Strategy comparison",
        "",
        "| Strategy | R@1 | R@5 | MRR | Full evidence@5 | Fact coverage* | Claim support* | Abstention F1 | Forbidden hits | Retrieval mean/P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, result in payload["strategies"].items():
        retrieval = result["retrieval"]
        latency = result["latency_ms"]["retrieval"]
        lines.append(
            f"| {strategy} | {retrieval['recall_at_1']:.3f} | {retrieval['recall_at_5']:.3f} | "
            f"{retrieval['mrr']:.3f} | {retrieval['full_evidence_success_at_5']:.3f} | "
            f"{result['factual_correctness_proxy']['expected_fact_coverage']:.3f} | "
            f"{result['claim_evidence_entailment_proxy']['support_rate']:.3f} | "
            f"{result['abstention']['f1']:.3f} | "
            f"{result['authorization_and_versioning']['forbidden_document_hits']} | "
            f"{latency['mean']:.2f}/{latency['p95']:.2f} |"
        )
    lines += [
        "",
        "\\* Automated heuristic. Fact coverage is explicit token-set presence against manually authored references. "
        "Claim support requires the extractive claim to occur in cited evidence. Neither is a semantic judge.",
        "",
        "## Abstention and end-to-end latency",
        "",
        "| Strategy | Precision | Recall | F1 | False-abstention rate | Missed-abstention rate | End-to-end mean/P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, result in payload["strategies"].items():
        abstention = result["abstention"]
        latency = result["latency_ms"]["end_to_end"]
        lines.append(
            f"| {strategy} | {abstention['precision']:.3f} | {abstention['recall']:.3f} | "
            f"{abstention['f1']:.3f} | {abstention['false_abstention_rate']:.3f} | "
            f"{abstention['missed_abstention_rate']:.3f} | "
            f"{latency['mean']:.2f}/{latency['p95']:.2f} |"
        )
    lines += [
        "",
        "The abstention positive class is a benchmark case marked answerable=false.",
        "",
        "## Failure categories",
        "",
    ]
    for strategy, result in payload["strategies"].items():
        failures = result["failure_categories"]
        formatted = ", ".join(f"{key}={value}" for key, value in failures.items()) or "none"
        lines.append(f"- **{strategy}:** {formatted}")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- The corpus and benchmark are small, synthetic, and project-authored.",
        "- Independent review is pending; this report must not be described as independently validated.",
        "- Latency is machine- and cache-dependent. Model/index startup is excluded.",
        "- Citation validity, extractive support, factual correctness, and answer usefulness are distinct properties.",
        "- Static local tokens demonstrate an authorization boundary but are not production authentication.",
        "- No paid API or cloud resource is used by this evaluation.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline quality benchmark")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/evaluation/quality_v3.json")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--lexical-artifacts-dir", type=Path, default=ROOT / "artifacts/v3/tfidf")
    parser.add_argument("--dense-artifacts-dir", type=Path, default=ROOT / "artifacts/v3/dense")
    parser.add_argument("--identities", type=Path, default=settings.local_identities_path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/evaluation/quality")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=settings.retrieval_candidate_k)
    parser.add_argument("--rrf-constant", type=int, default=settings.rrf_constant)
    parser.add_argument("--reranker-model", default=settings.reranker_model)
    parser.add_argument("--reranker-revision", default=settings.reranker_revision)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Evaluation is offline by default: cached models are required and accidental network
    # lookups cannot turn a reproducibility run into an external dependency.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    validate_quality_benchmark(benchmark)
    identities = LocalIdentityProvider.from_json(args.identities)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "benchmark": {
            "path": str(args.benchmark),
            "corpus": benchmark["corpus"],
            "case_count": len(benchmark["cases"]),
            "review_provenance": benchmark["review_provenance"],
            "evaluation_contract": benchmark["evaluation_contract"],
        },
        "strategies": {},
    }
    for strategy in args.strategies:
        config = RetrievalPipelineConfig(
            strategy=strategy,
            candidate_k=args.candidate_k,
            rrf_constant=args.rrf_constant,
            reranker_model=args.reranker_model,
            reranker_revision=args.reranker_revision,
        )
        retriever = load_pipeline(
            config,
            lexical_artifacts_dir=args.lexical_artifacts_dir,
            dense_artifacts_dir=args.dense_artifacts_dir,
        )
        report["strategies"][strategy] = evaluate_strategy(
            strategy, benchmark["cases"], retriever, identities, top_k=args.top_k
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(report)
    (args.output_dir / "quality_report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

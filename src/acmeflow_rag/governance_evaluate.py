"""Lifecycle, precedence, and authorization evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path

from .auth import LocalIdentityProvider
from .config import settings
from .evaluate import calculate_abstention_metrics, safe_divide
from .pipeline import RetrievalPipelineConfig, load_pipeline
from .service import RagService


def evaluate_governance(cases, service, identity_provider, top_k=5):
    correct_versions = retrieval_hits = leaks = 0
    decisions = {}
    latencies = []
    details = []
    for case in cases:
        principal = identity_provider.authenticate(case["identity_token"])
        started = time.perf_counter()
        response = service.ask(
            case["question"], top_k, principal, department=case.get("department"),
            as_of=date.fromisoformat(case["as_of"])
        )
        latencies.append((time.perf_counter() - started) * 1000)
        returned = set(response["source_document_ids"])
        expected = set(case["expected_documents"])
        forbidden = set(case["forbidden_documents"])
        correct_versions += int(expected.issubset(returned) and not returned.intersection(forbidden))
        retrieval_hits += int(bool(expected.intersection(returned)))
        leaks += len(returned.intersection(forbidden))
        decisions[case["id"]] = response["abstained"]
        details.append({
            "case_id": case["id"],
            "reason_code": response["reason_code"],
            "returned_document_ids": sorted(returned),
            "forbidden_leaks": sorted(returned.intersection(forbidden)),
        })
    ordered = sorted(latencies)
    def percentile(fraction):
        return ordered[round((len(ordered) - 1) * fraction)]
    answerable_count = sum(case["answerable"] for case in cases)
    return {
        "case_count": len(cases),
        "version_selection_accuracy": safe_divide(correct_versions, len(cases)),
        "expected_document_retrieval_rate": safe_divide(retrieval_hits, answerable_count),
        "unauthorized_or_ineligible_document_leaks": leaks,
        "abstention": calculate_abstention_metrics(cases, decisions),
        "latency_ms": {
            "scope": "warm end-to-end local service decision, including retrieval, evidence assessment, and offline generation; excludes model/index startup",
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(0.50) if ordered else 0.0,
            "p95": percentile(0.95) if ordered else 0.0,
        },
        "details": details,
        "limitations": "Small synthetic governance fixture; citation existence is not entailment and local tokens are not production authentication.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("data/evaluation/governance_v3.json"))
    parser.add_argument("--strategy", choices=("lexical", "dense", "hybrid", "hybrid_rerank"), default="hybrid")
    parser.add_argument("--lexical-artifacts-dir", type=Path, required=True)
    parser.add_argument("--dense-artifacts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reranker-revision", default=settings.reranker_revision)
    args = parser.parse_args()
    config = RetrievalPipelineConfig(strategy=args.strategy, reranker_revision=args.reranker_revision)
    retriever = load_pipeline(
        config,
        lexical_artifacts_dir=args.lexical_artifacts_dir,
        dense_artifacts_dir=args.dense_artifacts_dir,
    )
    service = RagService(retriever=retriever, retriever_type=args.strategy)
    identities = LocalIdentityProvider.from_json(settings.local_identities_path)
    cases = json.loads(args.benchmark.read_text(encoding="utf-8"))["cases"]
    service.ask("warmup policy query", 1, identities.authenticate("dev-employee-token"))
    metrics = evaluate_governance(cases, service, identities)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "governance_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

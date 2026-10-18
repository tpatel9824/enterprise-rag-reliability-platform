"""Credential-free local local demonstration for structured RAG responses."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .auth import LocalIdentityProvider
from .config import ROOT, settings
from .pipeline import RetrievalPipelineConfig, load_pipeline
from .service import RagService


def format_response(question: str, response: dict) -> str:
    lines = [
        f"Question: {question}",
        f"Status: {'ABSTAINED' if response['abstained'] else 'ANSWERED'}",
        f"Reason: {response['reason_code']}",
        f"Answer: {response['answer']}",
        "",
        "Sources:",
    ]
    if not response["evidence_passages"]:
        lines.append("  (none)")
    for evidence in response["evidence_passages"]:
        heading = " > ".join(evidence.get("heading_path", [])) or "(document root)"
        lines.extend([
            f"  - {evidence['document_id']} / {evidence['chunk_id']}",
            f"    Heading: {heading}",
            f"    Evidence: {evidence['text']}",
        ])
    if response.get("claims"):
        lines += ["", "Claim-to-evidence map:"]
        for claim in response["claims"]:
            citations = ", ".join(claim["cited_chunk_ids"])
            lines.append(f"  - {claim['claim_id']}: {claim['text']} -> {citations}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline interactive AcmeFlow RAG demo")
    parser.add_argument("--strategy", choices=("lexical", "dense", "hybrid", "hybrid_rerank"), default="hybrid")
    parser.add_argument("--question", help="Ask once; omit for an interactive prompt")
    parser.add_argument("--token", default="dev-support-token", help="Development identity token")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 6, 15))
    parser.add_argument("--department")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lexical-artifacts-dir", type=Path, default=ROOT / "artifacts/v3/tfidf")
    parser.add_argument("--dense-artifacts-dir", type=Path, default=ROOT / "artifacts/v3/dense")
    parser.add_argument("--identities", type=Path, default=settings.local_identities_path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    retriever = load_pipeline(
        RetrievalPipelineConfig(strategy=args.strategy),
        lexical_artifacts_dir=args.lexical_artifacts_dir,
        dense_artifacts_dir=args.dense_artifacts_dir,
    )
    service = RagService(retriever=retriever, retriever_type=args.strategy)
    principal = LocalIdentityProvider.from_json(args.identities).authenticate(args.token)

    def ask(question: str) -> None:
        response = service.ask(question, args.top_k, principal, args.department, args.as_of)
        print(format_response(question, response))

    if args.question:
        ask(args.question)
        return
    print("AcmeFlow offline demo. Type a question, or 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            return
        if question.casefold() in {"quit", "exit"}:
            return
        if question:
            ask(question)


if __name__ == "__main__":
    main()

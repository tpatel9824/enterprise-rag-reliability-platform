import argparse
from pathlib import Path
from datetime import date

from .auth import LocalIdentityProvider
from .config import settings
from .pipeline import RetrievalPipelineConfig, load_pipeline, normalize_strategy
from .service import RagService


STRATEGIES = ("lexical", "tfidf", "dense", "hybrid", "hybrid_rerank")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--identity-token", required=True,
                        help="Development token resolved by the configured local identity provider")
    parser.add_argument("--department", default=None)
    parser.add_argument("--as-of", type=date.fromisoformat, default=None,
                        help="Policy effective date in YYYY-MM-DD format")
    parser.add_argument("--strategy", "--retriever", dest="strategy", choices=STRATEGIES,
                        default=settings.retrieval_strategy)
    parser.add_argument("--artifacts-dir", type=Path, default=None,
                        help="Backward-compatible component artifact directory")
    parser.add_argument("--lexical-artifacts-dir", type=Path, default=settings.lexical_artifacts_dir)
    parser.add_argument("--dense-artifacts-dir", type=Path, default=settings.dense_artifacts_dir)
    parser.add_argument("--candidate-k", type=int, default=settings.retrieval_candidate_k)
    parser.add_argument("--rrf-constant", type=int, default=settings.rrf_constant)
    parser.add_argument("--reranker-model", default=settings.reranker_model)
    parser.add_argument("--reranker-revision", default=settings.reranker_revision)
    args = parser.parse_args()

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
    principal = LocalIdentityProvider.from_json(settings.local_identities_path).authenticate(
        args.identity_token
    )
    result = service.ask(
        args.question, args.top_k, principal, args.department, args.as_of
    )
    print("\nANSWER\n" + result["answer"])
    print("ANSWERABLE:", result["answerable"])
    print("\nABSTAINED:", result["abstained"])
    print("REASON:", result["reason_code"])
    print("\nSOURCES")
    for index, item in enumerate(result["retrieval"], 1):
        chunk = item["chunk"]
        print(f'{index}. {chunk["document_id"]} | {chunk["section"]} | score={item["score"]:.4f}')
    if result["claims"]:
        print("\nCLAIMS")
        for claim in result["claims"]:
            print(f'{claim["claim_id"]}: {claim["text"]} -> {", ".join(claim["cited_chunk_ids"])}')


if __name__ == "__main__":
    main()

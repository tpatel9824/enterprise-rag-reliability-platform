import argparse
from pathlib import Path

from .config import settings
from .pipeline import RetrievalPipelineConfig, load_pipeline, normalize_strategy
from .service import RagService


STRATEGIES = ("lexical", "tfidf", "dense", "hybrid", "hybrid_rerank")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--department", default=None)
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
    result = service.ask(
        args.question, args.top_k, set(args.role) if args.role else None, args.department
    )
    print("\nANSWER\n" + result["answer"])
    print("\nABSTAINED:", result["abstained"])
    print("BEST SCORE:", result["best_score"])
    print("\nSOURCES")
    for index, item in enumerate(result["retrieval"], 1):
        chunk = item["chunk"]
        print(f'{index}. {chunk["document_id"]} | {chunk["section"]} | score={item["score"]:.4f}')


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from .config import settings
from .service import RagService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--department", default=None)
    parser.add_argument("--retriever", choices=("tfidf", "dense"), default=settings.retriever_type)
    parser.add_argument("--artifacts-dir", type=Path, default=settings.artifacts_dir)
    args = parser.parse_args()
    service = RagService(retriever_type=args.retriever, artifacts_dir=args.artifacts_dir)
    result = service.ask(
        args.question,
        args.top_k,
        set(args.role) if args.role else None,
        args.department,
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

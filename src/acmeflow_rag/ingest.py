from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .chunker import chunk_corpus
from .config import settings
from .dense_retrieval import DEFAULT_MODEL_NAME
from .loader import load_corpus
from .retrieval import TfidfRetriever
from .artifact_store import create_artifact_manifest


def ingest_corpus(
    corpus_path: Path,
    artifacts_dir: Path,
    max_words: int = 90,
    overlap_words: int = 20,
    *,
    retriever_type: str = "tfidf",
    model_name: str = DEFAULT_MODEL_NAME,
    model_revision: str | None = None,
    corpus_version: str | None = None,
    dense_model: Any | None = None,
    faiss_module: Any | None = None,
    require_lifecycle: bool = False,
) -> tuple[int, list]:
    documents = load_corpus(corpus_path, require_lifecycle=require_lifecycle)
    chunks = chunk_corpus(documents, max_words=max_words, overlap_words=overlap_words)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "chunks.json").write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], indent=2), encoding="utf-8"
    )

    if retriever_type == "tfidf":
        retriever = TfidfRetriever()
        retriever.fit(chunks)
        retriever.save(artifacts_dir / "index.joblib")
    elif retriever_type == "dense":
        from .dense_retrieval import DenseRetriever

        retriever = DenseRetriever(
            model_name=model_name,
            model_revision=model_revision,
            model=dense_model,
            faiss_module=faiss_module,
        )
        retriever.fit(
            chunks,
            corpus_version=corpus_version or corpus_path.name,
            chunking_config={"max_words": max_words, "overlap_words": overlap_words},
        )
        retriever.save(artifacts_dir)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type!r}")
    create_artifact_manifest(
        artifacts_dir,
        artifact_type=retriever_type,
        corpus_version=corpus_version or corpus_path.name,
        chunk_count=len(chunks),
    )
    return len(documents), chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a retrieval index for a Markdown corpus")
    parser.add_argument("--corpus", type=Path, default=settings.documents_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=settings.artifacts_dir)
    parser.add_argument("--max-words", type=int, default=90)
    parser.add_argument("--overlap-words", type=int, default=20)
    parser.add_argument("--retriever", choices=("tfidf", "dense"), default="tfidf")
    parser.add_argument("--model-name", default=settings.dense_model_name)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--corpus-version", default=None)
    parser.add_argument("--require-lifecycle", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    document_count, chunks = ingest_corpus(
        args.corpus,
        args.artifacts_dir,
        args.max_words,
        args.overlap_words,
        retriever_type=args.retriever,
        model_name=args.model_name,
        model_revision=args.model_revision,
        corpus_version=args.corpus_version,
        require_lifecycle=args.require_lifecycle,
    )
    lengths = [len(chunk.text.split()) for chunk in chunks]
    print(f"Retriever: {args.retriever}")
    print(f"Loaded documents: {document_count}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Chunk words: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")


if __name__ == "__main__":
    main()

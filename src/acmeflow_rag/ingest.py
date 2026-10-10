from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunker import chunk_corpus
from .config import settings
from .loader import load_corpus
from .retrieval import TfidfRetriever


def ingest_corpus(
    corpus_path: Path,
    artifacts_dir: Path,
    max_words: int = 90,
    overlap_words: int = 20,
) -> tuple[int, list]:
    documents = load_corpus(corpus_path)
    chunks = chunk_corpus(documents, max_words=max_words, overlap_words=overlap_words)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "chunks.json").write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], indent=2), encoding="utf-8"
    )
    retriever = TfidfRetriever()
    retriever.fit(chunks)
    retriever.save(artifacts_dir / "index.joblib")
    return len(documents), chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a TF-IDF index for a Markdown corpus")
    parser.add_argument("--corpus", type=Path, default=settings.documents_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=settings.artifacts_dir)
    parser.add_argument("--max-words", type=int, default=90)
    parser.add_argument("--overlap-words", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    document_count, chunks = ingest_corpus(
        args.corpus, args.artifacts_dir, args.max_words, args.overlap_words
    )
    lengths = [len(chunk.text.split()) for chunk in chunks]
    print(f"Loaded documents: {document_count}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Chunk words: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")


if __name__ == "__main__":
    main()

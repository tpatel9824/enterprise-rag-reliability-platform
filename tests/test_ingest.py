import json
from pathlib import Path

from acmeflow_rag.ingest import ingest_corpus


def test_ingestion_accepts_explicit_corpus_and_artifact_paths(tmp_path):
    artifact_path = tmp_path / "v2-artifacts"

    document_count, chunks = ingest_corpus(
        Path("data/documents_v2"), artifact_path, max_words=90, overlap_words=20
    )

    serialized = json.loads((artifact_path / "chunks.json").read_text(encoding="utf-8"))
    assert document_count == 10
    assert len(chunks) > document_count
    assert len(serialized) == len(chunks)
    assert (artifact_path / "index.joblib").exists()
    assert all(item["heading_path"] for item in serialized)

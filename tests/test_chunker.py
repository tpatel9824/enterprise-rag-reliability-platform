from pathlib import Path
from acmeflow_rag.loader import load_document
from acmeflow_rag.chunker import chunk_document

def test_chunk_metadata():
    doc=load_document(Path("data/documents/product/product_refund_policy_2026.md")); chunks=chunk_document(doc)
    assert chunks and all(c.document_id==doc.document_id for c in chunks) and all(c.section for c in chunks)

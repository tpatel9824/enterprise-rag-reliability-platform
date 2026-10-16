from pathlib import Path
from acmeflow_rag.loader import load_corpus
from acmeflow_rag.chunker import chunk_corpus
from acmeflow_rag.retrieval import TfidfRetriever

def test_refund_query():
    chunks = chunk_corpus(load_corpus(Path("data/documents")))
    r = TfidfRetriever()
    r.fit(chunks)
    ids=[x.chunk.document_id for x in r.search("How long does a card refund take?",5)]
    assert "product_refund_policy_2026" in ids

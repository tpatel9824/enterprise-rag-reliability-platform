from pathlib import Path
from acmeflow_rag.loader import load_corpus

def test_corpus_size():
    docs=load_corpus(Path("data/documents")); assert len(docs)==60

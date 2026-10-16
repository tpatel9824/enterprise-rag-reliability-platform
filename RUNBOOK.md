# Runbook

1. `pip install -e ".[dev,cloud]"`
2. `python -m acmeflow_rag.ingest`
3. `python -m acmeflow_rag.cli "How long does a card refund take?"`
4. `python -m acmeflow_rag.cli "What is AcmeFlow's cryptocurrency refund policy?"`
5. `python -m acmeflow_rag.evaluate`
6. `uvicorn acmeflow_rag.api:app --reload`

The next development step is dense retrieval using sentence-transformer embeddings and FAISS.

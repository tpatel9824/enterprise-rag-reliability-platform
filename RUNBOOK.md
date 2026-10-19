# Local operations runbook

## Install

```bash
python3.12 -m venv .venv
python -m pip install --requirement requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
```

## Build local artifacts

```bash
python -m acmeflow_rag.ingest --retriever tfidf --corpus data/documents_v3 --artifacts-dir artifacts/v3/tfidf --max-words 40 --overlap-words 10 --corpus-version v3 --require-lifecycle
python -m acmeflow_rag.ingest --retriever dense --corpus data/documents_v3 --artifacts-dir artifacts/v3/dense --max-words 40 --overlap-words 10 --corpus-version v3 --require-lifecycle --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

## Validate

```bash
python -m pytest
python -m ruff check src tests
python -m pip check
python -m acmeflow_rag.quality_evaluate
```

## Run

```bash
python -m acmeflow_rag.demo --strategy lexical --token dev-support-token
```

For the API, set `RETRIEVAL_STRATEGY`, `LEXICAL_ARTIFACTS_DIR`, and
`DENSE_ARTIFACTS_DIR` as required, then run:

```bash
uvicorn acmeflow_rag.api:app --host 127.0.0.1 --port 8000
```

Check `/health`, `/ready`, and `/metrics`. Development tokens must not be used in production.

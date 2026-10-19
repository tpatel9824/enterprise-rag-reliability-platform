# Local demonstration

This demonstration uses the deterministic lexical pipeline and does not require model downloads,
cloud credentials, paid APIs, Docker, or external services after dependencies are installed.

## Build the index

```bash
python -m acmeflow_rag.ingest --retriever tfidf --corpus data/documents_v3 --artifacts-dir artifacts/v3/tfidf --max-words 40 --overlap-words 10 --corpus-version v3 --require-lifecycle
```

Expected summary:

```text
Retriever: tfidf
Loaded documents: 11
Created chunks: 45
Chunk words: min=13 median=35 max=40
```

## Grounded multi-document answer

```bash
python -m acmeflow_rag.demo --strategy lexical --token dev-support-token --as-of 2026-06-15 --question "Does cancellation automatically refund the last charge, and what must Support verify?"
```

The result is `ANSWERED` with reason `ANSWER_GROUNDED`. It cites the subscription policy,
support procedure, and refund policy, then prints the complete evidence passages and
claim-to-evidence mapping.

## Authorization denial

```bash
python -m acmeflow_rag.demo --strategy lexical --token dev-employee-token --department Security --as-of 2026-06-15 --question "What approvals are required for production access?"
```

Expected result:

```text
Status: ABSTAINED
Reason: ACCESS_DENIED
Sources:
  (none)
```

The development identity has no Security role, and protected chunks are excluded before ranking.

## Known abstention failure

```bash
python -m acmeflow_rag.demo --strategy lexical --token dev-employee-token --as-of 2026-06-15 --question "What is AcmeFlow's daily meal reimbursement cap?"
```

The current evidence policy may answer with a cited expense-submission sentence rather than
abstaining. The citation is valid and the claim is extractive, but the response is irrelevant.
This case demonstrates why citation validity, evidence support, answer correctness, and
abstention are evaluated separately.

Run all examples on PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\demo.ps1 -Strategy lexical
```

## Interactive mode

```bash
python -m acmeflow_rag.demo --strategy lexical --token dev-support-token
```

Use `quit` or `exit` to stop the session.

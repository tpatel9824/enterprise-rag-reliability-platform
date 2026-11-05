# Interactive local demonstration

The Streamlit interface is a presentation layer over the existing `RagService`, retrieval
pipelines, evidence assessor, lifecycle selector, authorization filters, generator, and citation
validator. It does not implement a second RAG path.

## Setup

From a clean checkout, create the environment and build both v3 indexes:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --requirement requirements.lock
python -m pip install --no-deps -e .

python -m acmeflow_rag.ingest --retriever tfidf --corpus data/documents_v3 --artifacts-dir artifacts/v3/tfidf --max-words 40 --overlap-words 10 --corpus-version v3 --require-lifecycle
python -m acmeflow_rag.ingest --retriever dense --corpus data/documents_v3 --artifacts-dir artifacts/v3/dense --max-words 40 --overlap-words 10 --corpus-version v3 --require-lifecycle --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

The dense build downloads `sentence-transformers/all-MiniLM-L6-v2` once if it is not cached.
Hybrid + reranker similarly requires the configured cross-encoder model. Lexical mode works
without either model download.

Run the interface:

```powershell
python -m streamlit run src/acmeflow_rag/streamlit_app.py
```

Streamlit opens `http://localhost:8501`. The app uses deterministic offline generation by default;
no paid API or cloud credential is required. Identity labels are mapped server-side to the
development-only fixture in `data/identity/local_identities.json`; credentials are not rendered in
the browser.

## Suggested walkthrough

1. Start with **Multi-document reasoning** and show the answer, claim citations, and evidence from
   both subscription and Support documents.
2. Load **Historical policy** and point out the explicit 2025 as-of date and selected policy ID.
3. Load **Customer-specific precedence** to show signed Acme terms overriding the standard policy.
4. Load **Authorization restriction** and show that Security evidence never reaches the answer
   context for the Employee identity.
5. Load **Evidence-based abstention** and explain the strategy-specific reason code.
6. Compare retrieval strategies and relate the live request latency to the verified benchmark table.

Model initialization can make the first dense or reranked request slower. The live number is an
end-to-end measurement; the benchmark table reports warm retrieval-only latency with startup
excluded.

## Recording a short GitHub video or GIF

Use a 1280×800 browser window and keep the recording to roughly 60–90 seconds:

1. Begin on the hero and query controls for two seconds.
2. Run the multi-document example and expand one evidence passage.
3. Switch to the authorization example and show `ACCESS_DENIED` with no returned evidence.
4. Run the abstention example and pause on its reason code.
5. End on the verified metrics table and its limitation note.

Record locally with the operating system screen recorder or OBS. For a lightweight GitHub asset,
convert the recording with a local tool such as `ffmpeg`:

```powershell
ffmpeg -i demo.mp4 -vf "fps=12,scale=1200:-1:flags=lanczos" -loop 0 demo.gif
```

Keep the source video outside Git. If a GIF is committed later, review its size, ensure it contains
no notifications or personal information, and add descriptive alt text in the README. The current
repository does not publish or host a recording.

## Boundaries

- The corpus, identities, customer terms, and benchmark cases are synthetic.
- Static local identities are a development fixture, not production authentication or RBAC.
- The benchmark is small and project-authored; independent review remains pending.
- Citation validity proves that cited chunk IDs exist in authorized context, not that every claim is
  semantically entailed.
- The meal-reimbursement-cap benchmark case is a known missed abstention for lexical, dense, and
  hybrid retrieval; it is retained in evaluation as a visible failure, not used as a success demo.
- No GCP deployment or paid Gemini call is part of this demonstration.

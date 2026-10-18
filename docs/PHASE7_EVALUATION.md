# Phase 7: end-to-end offline quality evaluation

This report records a real local run on 2026-10-18 against the 16-case
`quality_v3.json` held-out benchmark, the 11-document v3 corpus, top K = 5, and the
offline deterministic extractive generator. The benchmark labels are project-authored
from traceable corpus passages. **Independent review is pending.** These results must
not be described as independently validated.

## Verified strategy comparison

| Strategy | Recall@1 | Recall@5 | MRR | Full evidence@5 | Expected-fact coverage* | Extractive claim support* | Abstention F1 | Forbidden hits | Retrieval mean/P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lexical | 0.833 | 1.000 | 0.903 | 1.000 | 0.235 | 1.000 | 0.667 | 0 | 1.69 / 1.84 |
| dense | 0.917 | 1.000 | 0.958 | 1.000 | 0.353 | 1.000 | 0.571 | 0 | 15.76 / 18.40 |
| hybrid | 0.917 | 1.000 | 0.944 | 1.000 | 0.235 | 1.000 | 0.600 | 0 | 28.41 / 49.06 |
| hybrid + reranker | 0.917 | 1.000 | 0.958 | 1.000 | 0.353 | 1.000 | 0.667 | 0 | 240.74 / 370.87 |

The latency figures are warm retrieval calls on the local Windows CPU. They include
query embedding and reranking when applicable, but exclude model/index startup and one
unmeasured warm-up query per strategy. They
are not portable performance claims. The run used locally cached real MiniLM and
cross-encoder models; no paid API was called.

## Abstention and end-to-end behavior

Here, “positive” means the benchmark says the system should abstain.

| Strategy | Precision | Recall | F1 | False-abstention rate | Missed-abstention rate | End-to-end mean/P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| lexical | 0.600 | 0.750 | 0.667 | 0.167 | 0.250 | 2.13 / 2.55 |
| dense | 0.667 | 0.500 | 0.571 | 0.083 | 0.500 | 15.66 / 17.74 |
| hybrid | 0.500 | 0.750 | 0.600 | 0.250 | 0.250 | 29.86 / 49.77 |
| hybrid + reranker | 0.600 | 0.750 | 0.667 | 0.167 | 0.250 | 225.01 / 315.18 |

False-abstention rate is answerable cases incorrectly abstained divided by all answerable
cases. Missed-abstention rate is unanswerable cases incorrectly answered divided by all
unanswerable cases. End-to-end latency includes lifecycle selection, authorized retrieval,
evidence assessment, and offline generation, but still excludes startup and warm-up.

## What the quality columns mean

Expected-fact coverage is an **automated lexical proxy**. Each manually authored fact
has one or more explicit acceptable token sets. A fact matches when all tokens in one
set occur in the answer. This can miss valid paraphrases and can reward a misleading
answer that happens to contain the expected words. It is not a semantic correctness
judge.

Extractive claim support is an **automated entailment proxy**. A claim is counted as
supported when its normalized text appears inside at least one evidence passage named
by that claim's citation. This is useful for the extractive offline generator, but it
does not prove truth, completeness, or semantic entailment. Citation-ID validity is
reported separately by the existing grounding metrics.

Manual reference judgments are the benchmark's answerability, expected-document,
forbidden-document, and expected-fact labels. They were checked by the project author
against the versioned corpus. No Phase 7 generated answers have independent human
quality ratings yet.

## Observed failures

- Lexical: 2 false abstentions, 1 missed abstention, 9 cases with incomplete expected facts.
- Dense: 1 false abstention, 2 missed abstentions, 8 cases with incomplete expected facts.
- Hybrid: 3 false abstentions, 1 missed abstention, 9 cases with incomplete expected facts.
- Hybrid + reranker: 2 false abstentions, 1 missed abstention, 8 cases with incomplete expected facts.
- All strategies returned zero benchmark-forbidden documents.
- All generated extractive claims occurred in their cited passages; this does not imply
  that the answers were correct or responsive.

The low expected-fact coverage and missed abstentions expose a real limitation: the
offline generator selects locally overlapping sentences, and the conservative evidence
heuristics still accept some unrelated evidence. The demo transcript includes one such
failure rather than hiding it.

## Reproduce

```powershell
python -m acmeflow_rag.ingest --retriever tfidf --corpus data/documents_v3 --artifacts-dir artifacts/v3/tfidf --max-words 40 --overlap-words 10 --corpus-version v3 --require-lifecycle
python -m acmeflow_rag.ingest --retriever dense --corpus data/documents_v3 --artifacts-dir artifacts/v3/dense --max-words 40 --overlap-words 10 --corpus-version v3 --require-lifecycle
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m acmeflow_rag.quality_evaluate
```

Dense ingestion downloads the pinned model revision on its first run if it is not cached; that is
a free model download, not a paid API call. The evaluator itself defaults to offline mode and
therefore fails rather than accessing the network if model files are absent. It writes
machine-readable `artifacts/phase7/quality_report.json` and a Markdown report. The
`artifacts/` directory stays untracked because indexes and generated measurements are
local build outputs.

## Limitations

- Sixteen synthetic cases cannot establish real-world generalization or statistical superiority.
- Independent benchmark review is pending.
- Reference facts are not a complete list of every acceptable answer.
- There is no learned or LLM-based semantic judge in offline mode.
- The development identity tokens are fixtures, not production authentication.
- No GCP deployment, Vertex AI generation test, or cloud resource was involved.

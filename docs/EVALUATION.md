# Evaluation methodology and results

The quality evaluation compares lexical, dense, hybrid, and reranked retrieval on the same
16-case held-out fixture, 11-document corpus, chunking configuration, authorization rules, and
top K = 5. Cases cover current and historical policies, signed customer precedence,
multi-document evidence, authorization denial, and unanswerable questions.

The corpus and labels are synthetic and project-authored. Independent review is pending. No
threshold is selected from the held-out labels.

## Recorded results

These results are from a local Windows CPU run using the deterministic offline generator and
locally cached real retrieval models. Retrieval latency includes query embedding and reranking
where applicable, but excludes index/model startup and one unmeasured warm-up query.

| Strategy | Recall@1 | Recall@5 | MRR | Full evidence@5 | Fact coverage proxy | Claim support proxy | Abstention F1 | Forbidden hits | Mean/P95 retrieval latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lexical | 0.833 | 1.000 | 0.903 | 1.000 | 0.235 | 1.000 | 0.667 | 0 | 1.69 / 1.84 ms |
| Dense | 0.917 | 1.000 | 0.958 | 1.000 | 0.353 | 1.000 | 0.571 | 0 | 15.76 / 18.40 ms |
| Hybrid | 0.917 | 1.000 | 0.944 | 1.000 | 0.235 | 1.000 | 0.600 | 0 | 28.41 / 49.06 ms |
| Hybrid + reranker | 0.917 | 1.000 | 0.958 | 1.000 | 0.353 | 1.000 | 0.667 | 0 | 240.74 / 370.87 ms |

| Strategy | Abstention precision | Abstention recall | False-abstention rate | Missed-abstention rate |
|---|---:|---:|---:|---:|
| Lexical | 0.600 | 0.750 | 0.167 | 0.250 |
| Dense | 0.667 | 0.500 | 0.083 | 0.500 |
| Hybrid | 0.500 | 0.750 | 0.250 | 0.250 |
| Hybrid + reranker | 0.600 | 0.750 | 0.167 | 0.250 |

All strategies had zero benchmark-forbidden document hits. Every strategy also made abstention
errors, and expected-fact coverage remained low despite complete document retrieval at K=5.

## Metric definitions

- **Recall@K:** fraction of answerable cases with at least one required document among the first K
  distinct document IDs.
- **MRR:** mean reciprocal rank of the first required document.
- **Full evidence@5:** fraction of answerable cases containing every required document among the
  first five distinct documents.
- **Required-document coverage:** mean fraction of required documents retrieved per case.
- **Abstention precision/recall/F1:** treats a benchmark case marked unanswerable as the positive
  class.
- **False-abstention rate:** answerable cases incorrectly abstained divided by answerable cases.
- **Missed-abstention rate:** unanswerable cases incorrectly answered divided by unanswerable cases.
- **Citation validity:** every claim cites a chunk ID present in authorized retrieved context.
- **Expected-fact coverage proxy:** explicit acceptable token sets found in the generated answer.
- **Claim support proxy:** normalized extractive claim text occurs in a cited evidence passage.
- **Forbidden hit:** a document labeled superseded, inapplicable, or unauthorized appears in
  returned sources.

Citation validity does not establish entailment. Token presence does not establish correctness,
and substring support does not establish truth, completeness, or absence of contradiction.

## Reproduction

```bash
python -m acmeflow_rag.ingest --retriever tfidf --corpus data/documents_v3 --artifacts-dir artifacts/v3/tfidf --max-words 40 --overlap-words 10 --corpus-version v3 --require-lifecycle
python -m acmeflow_rag.ingest --retriever dense --corpus data/documents_v3 --artifacts-dir artifacts/v3/dense --max-words 40 --overlap-words 10 --corpus-version v3 --require-lifecycle --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
python -m acmeflow_rag.quality_evaluate
```

Dense ingestion downloads the public model on first use when it is not cached. The evaluator
defaults Hugging Face libraries to offline mode. It writes machine-readable JSON and Markdown to
`artifacts/evaluation/quality`.

## Limitations

- Sixteen synthetic cases cannot establish generalization or statistical superiority.
- Independent label and output review is pending.
- Reference facts are not exhaustive descriptions of every acceptable answer.
- The automated proxies do not perform semantic entailment or factuality judging.
- Latency is machine- and cache-dependent and excludes startup.
- Development identities do not prove secure authentication or RBAC.
- No cloud service participated in the recorded evaluation.

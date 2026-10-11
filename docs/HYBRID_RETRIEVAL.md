# Hybrid retrieval and reranking

## Pipeline

The common retrieval interface exposes `search(query, top_k, allowed_roles, department)` for all
four strategies:

1. `lexical` ranks eligible chunks using TF-IDF cosine similarity.
2. `dense` embeds the query with MiniLM and searches normalized embeddings with FAISS inner
   product.
3. `hybrid` retrieves an eligible candidate pool from each component, deduplicates by chunk ID,
   and combines ranks with reciprocal rank fusion (RRF).
4. `hybrid_rerank` sends only the already-authorized fused candidates to a cross encoder, which
   jointly scores each query/chunk pair and returns the highest scores.

Authorization roles and the optional department constraint are applied by each component before
candidate selection. A second pipeline check rejects a component result that violates the request
constraints. Unauthorized text therefore does not enter fusion or the reranker. This boundary is
not secure authentication or RBAC. Phase 5 derives roles through an identity-provider interface;
the included static local provider is development-only and production must integrate a trusted
identity and entitlement system.

## Reciprocal rank fusion

For a chunk `d`, RRF adds its reciprocal rank from each result list:

```text
RRF(d) = sum(1 / (k + rank_i(d)))
```

With `k = 60`, a chunk ranked first by lexical and third by dense receives
`1/61 + 1/63 = 0.03227`. A chunk ranked second in only one list receives `1/62 = 0.01613`.
The first chunk ranks higher without requiring lexical and dense scores to share a scale.

The fusion constant is configurable. A larger value makes rank differences less influential;
a smaller value rewards the top ranks more sharply. Each input list is deduplicated by stable
chunk ID before scoring, and the fused output is deduplicated again. Equal scores are resolved
deterministically by best source rank and then chunk ID. An empty component result is valid and
the available list still produces results.

RRF is simple, deterministic, and avoids score calibration. It can nevertheless promote a result
that appears moderately high in both lists over a uniquely strong result, and it cannot infer
relevance beyond the component rankings. Candidate-pool size limits the evidence available to
fusion and reranking.

## Cross-encoder reranking

The optional `cross-encoder/ms-marco-MiniLM-L-6-v2` model jointly reads the query and each fused
candidate. This can model interactions that independent query/document embeddings miss, but its
cost grows with the number of candidates. Model initialization is lazy so startup is excluded
from measured retrieval latency; inference for every measured query is included.

Cross-encoder outputs are raw ranking scores, not probabilities. RRF scores are also ranking
scores. The service's legacy fixed abstention threshold has not been recalibrated for either
scale, so Phase 3 retrieval comparisons should use ranking metrics—not cross-strategy abstention
comparisons.

## Persistence and reproducibility

Lexical and dense component indexes remain independently persisted Phase 2 artifacts. Phase 3
evaluation writes `pipeline_config.json` beside `retrieval_metrics.json`, recording the strategy,
candidate count, RRF constant, and reranker model/revision. Hybrid loading requires the component
indexes to contain the same ordered chunk IDs; a mismatch raises an error instead of silently
fusing different corpora or chunk configurations.

Historical and current-policy documents are left untouched. RRF and reranking do not suppress a
superseded policy, so temporal conflict remains visible in evaluation and requires a later,
explicit policy/versioning design.

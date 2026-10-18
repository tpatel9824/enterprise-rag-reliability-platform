# Architecture — Phase 5

```text
AcmeFlow documents
       |
       v
 validated document loader
       |
       v
 section-aware chunker
       |
       v
 verified identity -> server-derived roles/customer scope
       |
 lifecycle + policy precedence eligibility
       |
 authorization + department eligibility
       |
       +-----------------------+
       |                       |
       v                       v
 TF-IDF retrieval       MiniLM + FAISS retrieval
       |                       |
       +-----------+-----------+
                   |
                   v
       optional reciprocal rank fusion
                   |
                   v
       optional cross-encoder reranking
                   |
                   v
             top-k evidence
                   |
                   v
       strategy-specific evidence assessment
                   |
          insufficient +----+ sufficient
                   |              |
                   v              v
              no answer     grounded generator
                                  |
                                  v
                       claim/citation validation
                                  |
                         invalid +----+ valid
                                  |          |
                                  v          v
                             no answer   structured answer
       |
       +--> offline answer
       |
       +--> optional Gemini generation
```

Retrieval and generation are intentionally separate. This lets us determine whether a bad
answer came from retrieval or from generation/grounding.

The selectable retrieval strategies are `lexical`, `dense`, `hybrid`, and `hybrid_rerank`.
Authorization and department filters are passed into both component retrievers before candidates
can enter fusion or reranking. The pipeline validates returned candidates again and fails closed
if an underlying retriever violates those constraints. This is retrieval-time filtering and does
not by itself establish secure RBAC.

Phase 5 removes client-provided roles from the API. Its local static-token identity provider is a
development adapter, not production authentication. All eligibility constraints are applied before
candidate text can reach hybrid fusion, the cross-encoder, Gemini, citations, or audit events.

Evidence assessment is deliberately separate from retrieval and generation. Each strategy has its
own policy because TF-IDF similarity, embedding cosine similarity, RRF scores, and cross-encoder
logits have different meanings and scales. The policies consume relevance signals, not calibrated
probabilities. Only evidence that survived retrieval authorization filters can be supplied to a
generator or accepted by citation validation.

## Phase 7 evaluation boundary

The offline quality runner sends identical held-out cases through each selectable retrieval
pipeline and the real service boundary, including identity resolution, lifecycle selection,
authorization, evidence assessment, generation, and citation validation. It records retrieval-only
latency separately from end-to-end local service latency.

Three answer properties remain intentionally distinct:

1. citation validity checks that every cited chunk ID exists in authorized retrieved context;
2. extractive claim support checks that a generated claim occurs in its cited passage; and
3. expected-fact coverage checks explicit reference tokens in the final answer.

None is a semantic factuality guarantee. Human review provenance belongs to the benchmark rather
than the retrieval code, and the current Phase 7 benchmark truthfully records independent review
as pending.

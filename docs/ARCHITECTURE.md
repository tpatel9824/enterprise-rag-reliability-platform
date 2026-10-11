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

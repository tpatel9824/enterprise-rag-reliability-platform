# Architecture — Phase 3

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
       +--> offline answer
       |
       +--> optional Gemini generation
```

Retrieval and generation are intentionally separate. This lets us determine whether a bad
answer came from retrieval or from generation/grounding.

The selectable retrieval strategies are `lexical`, `dense`, `hybrid`, and `hybrid_rerank`.
Authorization and department filters are passed into both component retrievers before candidates
can enter fusion or reranking. The pipeline validates returned candidates again and fails closed
if an underlying retriever violates those constraints. This is retrieval-time filtering, not
authentication or secure RBAC; the current API accepts client-provided roles.

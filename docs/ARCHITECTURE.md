# Architecture — Phase 1

```text
AcmeFlow documents
       |
       v
 document loader
       |
       v
 section-aware chunker
       |
       v
 TF-IDF vectorizer
       |
       v
 cosine similarity
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

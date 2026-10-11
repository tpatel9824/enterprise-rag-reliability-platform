# Embeddings, cosine similarity, and FAISS

## What an embedding represents

A sentence-transformer converts text into a fixed-length vector of floating-point numbers. Texts
with similar learned meaning tend to point in similar directions, even when they do not share the
same keywords. AcmeFlow Phase 2 uses `sentence-transformers/all-MiniLM-L6-v2`, whose expected
output dimension is 384. The persisted metadata records the observed dimension rather than
assuming it.

Embeddings are useful similarity features, not factual knowledge or authorization decisions. A
high similarity score does not prove that a passage answers a question correctly.

## A small numerical example

Suppose a query embedding is:

```text
q = [3, 4]
```

Its L2 length is `sqrt(3² + 4²) = 5`, so its normalized form is:

```text
q_normalized = [0.6, 0.8]
```

Suppose a chunk embedding is `[4, 3]`. Its normalized form is `[0.8, 0.6]`. Their inner product
after normalization is:

```text
(0.6 × 0.8) + (0.8 × 0.6) = 0.96
```

That value is also their cosine similarity. The vectors point in nearly the same direction. By
contrast, the normalized vector `[0, 1]` has similarity `0.8` to the query, and `[-0.6, -0.8]`
has similarity `-1.0`.

Without normalization, the raw inner product depends on both direction and vector magnitude. For
example, `[3, 4] · [4, 3] = 24`, which is not a cosine similarity. L2 normalization makes every
nonzero vector have length one, allowing inner product to equal cosine similarity.

## Why `IndexFlatIP`

FAISS `IndexFlatIP` stores vectors and performs exact inner-product search. “Flat” means it scans
the stored vectors rather than using an approximate partition or graph. It requires no training
step and is appropriate for this small corpus and for establishing a reproducible dense baseline.

AcmeFlow normalizes both document and query embeddings before adding or searching. Therefore:

```text
IndexFlatIP score = normalized query · normalized chunk = cosine similarity
```

This exact search is computationally linear in the number of chunks. It is simple and reliable
for the current corpus but is not a scalability design for millions of chunks.

## Index construction and validation

The dense index directory contains:

- `index.faiss`: serialized FAISS vectors and their row order;
- `chunks.json`: chunk text and metadata in the same row order;
- `index_metadata.json`: model, revision, dimension, index type, corpus version, chunking
  configuration, chunk count, and ordered chunk IDs.

Loading rejects mismatched dimensions, counts, or ordered chunk IDs. This matters because FAISS
returns integer row positions; if chunk metadata is reordered independently, a correct vector
score could be attributed to the wrong source or access policy.

## Authorization filtering

Phase 2 searches the complete exact index and applies role and department checks before a result
can enter the returned top K. Searching more candidates is necessary because filtering only an
unfiltered top K could return too few authorized results. This is still application-level
filtering based on client-supplied roles, not authentication or production RBAC.

## Limitations

- MiniLM may miss company-specific terminology or rank semantically related but non-authoritative
  text highly.
- Dense retrieval does not resolve policy versions by itself.
- Chunk embeddings discard some document-wide and cross-document context.
- Exact `IndexFlatIP` search does not provide approximate-search scale advantages.
- Results depend on model weights, model revision, corpus text, and chunking configuration.
- Similarity scores from TF-IDF and dense retrieval are not calibrated to the same scale, so a
  shared abstention threshold is not automatically valid.

# System architecture

```text
versioned documents -> validation -> section-aware chunks -> TF-IDF / FAISS artifacts
                                                               |
verified identity -> lifecycle and precedence -> authorization -+
                                                               |
                                            lexical / dense / RRF
                                                               |
                                              optional reranker
                                                               |
                                             evidence assessment
                                                /          \
                                          abstention     generation
                                                            |
                                                citation validation
                                                            |
                                                structured response
```

## Ingestion

The loader validates document metadata before chunking. The chunker preserves heading paths,
document and lifecycle fields, access roles, source paths, and normalized source offsets. Stable
chunk IDs derive from document ID, version, heading path, and text.

TF-IDF and dense artifacts persist independently. Dense artifacts record model name and revision,
dimension, index type, corpus version, chunking settings, and ordered chunk IDs. Artifact manifests
record file size and SHA-256 checksums.

## Query processing

`RagService.ask` requires an authenticated `Principal` and an explicit or default as-of date.
Policy selection identifies currently eligible standard or signed customer-specific documents.
Roles, department, and eligible document IDs are passed into each component retriever before
fusion or reranking.

The selectable strategies are `lexical`, `dense`, `hybrid`, and `hybrid_rerank`. Reciprocal Rank
Fusion combines ranks instead of incomparable raw scores. The cross-encoder processes only the
authorized hybrid candidate set. The pipeline validates candidates again and fails closed if a
base retriever violates a constraint.

## Evidence and generation

Evidence assessment is separate from retrieval and generation. Each strategy uses a distinct
policy because TF-IDF similarity, embedding cosine similarity, RRF scores, and cross-encoder
logits have different meanings and scales. These relevance signals are not calibrated
probabilities.

Insufficient evidence produces a structured abstention before generation. Sufficient evidence is
passed to either the deterministic extractive generator or the optional Gemini adapter. Citation
validation requires every claim to cite a chunk in the authorized retrieved context. Citation
existence does not establish factual correctness or entailment.

## Evaluation boundary

The quality runner sends identical held-out cases through each retrieval pipeline and service
boundary, including identity resolution, policy selection, authorization, evidence assessment,
generation, and citation validation. It records retrieval-only and end-to-end latency separately.

Citation validity, extractive claim support, and expected-fact coverage remain distinct. None is a
semantic factuality guarantee. Benchmark provenance records that independent review is pending.

## Operations and external systems

FastAPI exposes query, health, readiness, and metrics endpoints. Audit events contain identifiers
and decisions without query, answer, or document text. Logs are structured for stdout ingestion.

Local artifacts and deterministic generation are fully usable without cloud services. GCS,
Gemini, and OIDC adapters have mock-based coverage only. Cloud Run and Workload Identity
Federation files are deployment configuration and have not been exercised in GCP.

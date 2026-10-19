# Engineering tradeoffs

The project keeps retrieval strategies interchangeable so their costs and failure modes can be
measured under identical corpus, chunking, authorization, and top-K settings. The numbers below
come from the recorded local Windows CPU run over 16 synthetic, project-authored cases.
They are regression evidence, not production benchmarks or proof of statistical superiority.

## Retrieval strategy comparison

| Strategy | Why use it | Main cost or failure mode | Recall@1 | Full evidence@5 | Warm mean latency |
|---|---|---|---:|---:|---:|
| TF-IDF | Fast, deterministic, inspectable baseline; strong on exact policy vocabulary | Misses paraphrases and semantic similarity | 0.833 | 1.000 | 1.69 ms |
| Dense MiniLM + FAISS | Retrieves semantically related wording; exact `IndexFlatIP` search | Model dependency, embedding cost, less interpretable matches | 0.917 | 1.000 | 15.76 ms |
| Hybrid RRF | Combines lexical and dense ranks without comparing incompatible raw scores | Runs both retrievers; fusion rank can dilute a strong single-system result | 0.917 | 1.000 | 28.41 ms |
| Hybrid + cross-encoder | Scores query/chunk interactions and improved ordering in some earlier fixtures | Highest CPU latency; logits are not probabilities; may reduce diversity | 0.917 | 1.000 | 240.74 ms |

RRF uses `1 / (constant + rank)` and deduplicates by chunk ID. It avoids pretending TF-IDF
similarity and embedding cosine similarity share a numeric scale. The cross-encoder sees only
already-authorized candidates; authorization cannot be deferred until after reranking.

## Evidence coverage versus answer quality

All four strategies retrieved every required document within five distinct documents on the
quality set, yet automated expected-fact coverage was only 0.235–0.353. Retrieval success did not
translate directly into a complete answer because the deterministic generator selects up to three
overlapping sentences. This separation is intentional:

- retrieval metrics ask whether evidence was available;
- citation validity asks whether cited IDs existed in authorized context;
- extractive support asks whether a claim occurred in its cited text;
- expected-fact coverage is a lexical answer-content proxy; and
- true semantic correctness still requires qualified human judgment or a separately validated
  judge.

## Abstention

Each retrieval strategy has its own evidence policy because TF-IDF similarities, cosine
similarities, RRF scores, and cross-encoder logits are not comparable or calibrated probabilities.
On the quality benchmark, abstention F1 ranged from 0.571 to 0.667. Every strategy made at least one error.

Conservative abstention lowers unsupported-answer risk but can reject answerable queries. Permissive
thresholds improve answer rate but can produce a cited, extractive, irrelevant response—as the
meal-cap demo shows. The next rigorous step is a larger independently reviewed calibration set,
not tuning on the held-out quality labels.

## Chunking

Heading paths preserve document structure, sentence packing avoids unnecessary sentence breaks,
and overlap carries complete trailing sentences when they fit the word budget. Overlap can retain
local context across boundaries, but it duplicates text, may reduce result diversity, and cannot
solve cross-document reasoning. Sizes are word-based, not tokenizer-based.

## Governance and authorization

Effective-date selection and signed customer-policy precedence happen before retrieval. Server-
derived roles, customer scope, department, and eligible document IDs are passed to each base
retriever before fusion or reranking. A second validation layer fails closed on an invalid
candidate.

This ordering reduces accidental leakage inside the application pipeline, but it is not a proof
of secure RBAC. Local bearer tokens are fixtures. Production use would require a live-tested IdP,
revocation strategy, entitlement ownership, audit review, penetration testing, and operational
controls beyond this repository.

## Persistence and cloud boundaries

Local FAISS and TF-IDF indexes are appropriate for this 45-chunk v3 corpus and keep the demo free.
The GCS adapter transports checksum-validated index artifacts; it is not a managed vector store.
GCS, Vertex AI Gemini, and OIDC integrations have mock-based tests only. Cloud Run and Workload
Identity Federation files are deployment preparation, not evidence of a deployment.

## What I would measure before a production decision

- independently reviewed answer correctness and entailment;
- adversarial authorization and policy-precedence cases;
- threshold calibration with held-out confidence intervals;
- cold-start, concurrency, tail latency, and memory under realistic load;
- corpus growth, update frequency, and index rebuild/rollback behavior;
- live IdP, artifact-store, logging, and generator failure modes; and
- cost per indexed document and answered query.

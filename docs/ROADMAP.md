# Project status and roadmap

Status labels are evidence-based: **implemented and verified**, **mock-tested**, **configuration
only**, or **future work**. Implemented does not mean production-ready.

## Implemented and locally/CI verified

- Validated Markdown ingestion, lifecycle metadata, section/sentence-aware word chunking, stable
  IDs, source offsets, and reproducible TF-IDF artifacts.
- MiniLM embeddings, normalized FAISS `IndexFlatIP`, RRF hybrid retrieval, and optional
  cross-encoder reranking using real locally cached models.
- Authorization and department filters before candidate fusion/reranking, plus explicit as-of
  selection, supersession, and signed customer-policy precedence.
- Strategy-specific evidence heuristics, pre-generation abstention, deterministic offline
  generation, structured claims, authorized citations, and safe generator-failure handling.
- Retrieval, governance, abstention, citation, heuristic quality, latency, and failure-category
  evaluation with machine-readable reports.
- FastAPI endpoints, metadata-only audit events, content-safe structured logs, local metrics,
  non-root Docker image, container ingestion/readiness smoke test, and CI history scanning.
- Latest verified CI: 85 tests, Ruff, dependency checks, Docker build/smoke test, and Gitleaks over
  12 commits and approximately 467.54 KB, all passing in run 39164410475.

## Implemented adapters with mock-based tests only

- Google Cloud Storage artifact download/upload and checksum verification.
- Vertex AI Gemini generation using the Google Gen AI SDK and ADC configuration.
- OIDC/JWT identity provider boundary and verified-claim-to-principal mapping.

These components have not been exercised against live GCS, Vertex AI, an external issuer, or a
trusted identity gateway. Mocks validate application behavior, not IAM, quotas, network policy,
real token semantics, or service availability.

## Deployment configuration only

- Private-by-default Cloud Run service configuration.
- Artifact Registry build/push and rollback instructions.
- Workload Identity Federation-based, manually gated CD workflow.
- Least-privilege IAM recommendations and cost-control guidance.

No GCP project was modified, no APIs were enabled, no cloud resources were created, no image was
pushed to Artifact Registry, and no service was deployed.

## Near-term future work

1. Independently review the benchmark and manually label answer correctness and claim entailment.
2. Expand adversarial authorization, policy ambiguity, temporal, and contradictory-evidence cases.
3. Recalibrate evidence policies on a larger development set without using held-out labels.
4. Improve offline answer synthesis and contradiction handling, then measure whether expected-fact
   coverage and missed-abstention rate improve.
5. Add repeated-run latency distributions, memory profiling, concurrency, and cold-start tests.

## Later work requiring explicit approval or external systems

1. Test OIDC against a real issuer and define entitlement ownership, revocation, and incident
   response with a security reviewer.
2. Validate GCS artifacts and Gemini on an isolated non-production GCP project with budgets,
   quotas, private access, and log review.
3. Run a private Cloud Run staging deployment and test IAM invocation, rollback, and cleanup.
4. Add managed vector infrastructure only if corpus scale and measured operational needs justify
   its cost; local FAISS is sufficient for the current 45-chunk corpus.
5. Conduct threat modeling, dependency governance, load testing, accessibility review, and an
   independent security assessment before any production-readiness claim.

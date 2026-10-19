# Google Cloud deployment preparation

This is a deployment runbook, not a record of a completed deployment. No APIs, IAM bindings,
buckets, repositories, secrets, or Cloud Run services were created while preparing these adapters.

## Architecture and trust boundaries

Cloud Run remains private and requires IAM invocation. A trusted gateway or service-to-service
caller sends its Google-signed Cloud Run invocation token in `X-Serverless-Authorization` and the
end-user OIDC token in `Authorization`. The application validates the latter against the configured
issuer, audience, and JWKS before deriving roles and customer scope. Do not route untrusted traffic
directly to headers asserted by a gateway.

The Cloud Run runtime service account uses ADC for Cloud Storage and optionally Vertex AI. Never
create or mount a service-account key. The development token file is not copied into the image and
`APP_ENV=production` rejects `AUTH_MODE=local`.

Artifacts remain local FAISS/TF-IDF files. GCS is transport and durable storage, not online vector
search. Each immutable prefix has an `artifact_manifest.json`; CRC32C protects transport and the
application verifies SHA-256, file size, type, corpus version, chunk count, and dense metadata.

## Manual prerequisites

Choose values for `PROJECT_ID`, `REGION`, `ARTIFACT_BUCKET`, and an immutable `IMAGE_TAG`.
The following commands are examples to review before running:

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  storage.googleapis.com aiplatform.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com

gcloud artifacts repositories create acmeflow \
  --repository-format=docker --location="$REGION"

gcloud storage buckets create "gs://$ARTIFACT_BUCKET" \
  --location="$REGION" --uniform-bucket-level-access

gcloud iam service-accounts create acmeflow-rag-runtime
gcloud iam service-accounts create acmeflow-rag-deployer
```

Enable Vertex AI only if Gemini will be used. Secret Manager is not required for ADC, GCS, public
OIDC discovery values, or JWKS URLs. Use it only for a genuine secret introduced by a chosen
gateway/IdP integration, and grant access on that individual secret rather than the project.

## Least-privilege IAM

Runtime service account:

- bucket-level `roles/storage.objectViewer` on the artifact bucket;
- project-level `roles/aiplatform.user` only when Vertex AI Gemini is enabled;
- `roles/secretmanager.secretAccessor` only on a specifically required secret, if any.

Artifact publisher identity:

- bucket-level `roles/storage.objectCreator` for immutable, new version prefixes;
- use a separate controlled cleanup identity with object deletion permission when necessary.

Deployment identity used through GitHub Workload Identity Federation:

- repository-level `roles/artifactregistry.writer`;
- `roles/run.developer` for the target service/project;
- `roles/iam.serviceAccountUser` only on the runtime service account.

Invoker identities receive `roles/run.invoker` on this service only. Never bind `allUsers` and do
not pass `--allow-unauthenticated`.

## Build and artifact publication

```bash
docker build -t "$REGION-docker.pkg.dev/$PROJECT_ID/acmeflow/acmeflow-rag:$IMAGE_TAG" .
docker push "$REGION-docker.pkg.dev/$PROJECT_ID/acmeflow/acmeflow-rag:$IMAGE_TAG"

python -m acmeflow_rag.artifact_store upload \
  --source artifacts/v3/tfidf --type tfidf \
  --bucket "$ARTIFACT_BUCKET" --prefix indexes/v3/lexical/$IMAGE_TAG

python -m acmeflow_rag.artifact_store upload \
  --source artifacts/v3/dense --type dense \
  --bucket "$ARTIFACT_BUCKET" --prefix indexes/v3/dense/$IMAGE_TAG
```

Existing pre-Phase-6 indexes must be rebuilt once so they contain manifests. GCS uploads use
`if_generation_match=0`; publish to a new prefix instead of overwriting an active version.

## Private deployment

Replace every placeholder in `deploy/cloud-run-service.yaml`, or deploy explicitly:

```bash
gcloud run deploy acmeflow-rag \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/acmeflow/acmeflow-rag:$IMAGE_TAG" \
  --region "$REGION" \
  --service-account "acmeflow-rag-runtime@$PROJECT_ID.iam.gserviceaccount.com" \
  --no-allow-unauthenticated \
  --ingress internal-and-cloud-load-balancing \
  --min 0 --max 3 --concurrency 4 --cpu 2 --memory 4Gi
```

Set `APP_ENV=production`, `AUTH_MODE=oidc`, all OIDC values, GCS bucket/versioned prefixes, and
`ARTIFACT_STORAGE_MODE=gcs`. Start with `RAG_USE_GEMINI=false`. Enable Gemini only after an ADC
smoke test with a permitted runtime identity and a supported regional model.

Grant an approved caller invocation access:

```bash
gcloud run services add-iam-policy-binding acmeflow-rag \
  --region "$REGION" \
  --member "serviceAccount:trusted-gateway@$PROJECT_ID.iam.gserviceaccount.com" \
  --role roles/run.invoker
```

## CI and gated CD

`.github/workflows/ci.yml` runs Ruff, pytest, dependency checks, Docker build/container smoke, and
Gitleaks. `.github/workflows/cd.yml` is manual, requires the GitHub `production` environment, and
uses GitHub OIDC plus Google Workload Identity Federation. Configure repository/environment
variables; do not store a service-account JSON key.

Restrict the workload identity provider with repository, branch/ref, and environment attribute
conditions. Require environment reviewers and deploy only an immutable tested Git SHA/image tag.

## Health, observability, and rollback

- `/health`: process liveness only.
- `/ready`: confirms artifacts and retriever loaded; use as startup/readiness signal.
- `/metrics`: basic in-process request/error/status/mean-latency counters. They reset on each
  instance and are not a replacement for Cloud Monitoring.
- JSON request events record method, path, status, request ID, and latency. Audit events contain
  subject, document IDs, decision, reason, and as-of date. Neither contains content or credentials.

Deploy with no traffic first if your release process supports it, test the tagged revision, then
shift traffic. Roll back by routing traffic to the previous healthy revision and restoring its
immutable artifact prefixes. Do not overwrite indexes in place.

## Cost drivers and controls

Main cost drivers are Cloud Run CPU/memory/request duration, minimum instances, Vertex AI input and
output tokens, GCS bytes/operations/egress, Artifact Registry storage, build minutes, and Logging
ingestion/retention. Model downloads also increase startup time and network use.

Controls used here: minimum instances zero, maximum three, concurrency four, one process per
container, local FAISS rather than managed vector search, Gemini disabled by default, immutable
artifact caching per instance, bounded output tokens, and content-free logs. Before production,
set budgets/alerts, review Cloud Run/Vertex quotas, cap model output, set log exclusions/retention,
and load-test memory and cold starts. Pricing varies by region/model and must be checked in current
Google Cloud pricing calculators rather than copied into this repository.

## Cleanup

After traffic is rolled back or the project is retired:

```bash
gcloud run services delete acmeflow-rag --region "$REGION"
gcloud artifacts repositories delete acmeflow --location "$REGION"
gcloud storage rm --recursive "gs://$ARTIFACT_BUCKET/**"
gcloud storage buckets delete "gs://$ARTIFACT_BUCKET"
gcloud iam service-accounts delete "acmeflow-rag-runtime@$PROJECT_ID.iam.gserviceaccount.com"
gcloud iam service-accounts delete "acmeflow-rag-deployer@$PROJECT_ID.iam.gserviceaccount.com"
```

Also remove Cloud Run Invoker bindings, WIF provider/pool bindings, unused secrets, log sinks, and
budget alerts. Destructive cleanup commands must be reviewed against the exact project and bucket.

## Verification still required in Google Cloud

1. Build the Linux image and scan it for vulnerabilities.
2. Upload rebuilt manifest-bearing artifacts to a non-production bucket prefix.
3. Confirm runtime ADC can read only that bucket prefix/bucket and cannot write.
4. Verify OIDC issuer/JWKS/audience and gateway header separation end to end.
5. Deploy a private no-traffic revision and test IAM denial plus approved invocation.
6. Run Gemini with the selected model/region and verify structured JSON, failure behavior, quotas,
   safety settings, latency, and billing.
7. Test cold start, memory, concurrency, shutdown, rollback, and audit/log redaction.

## Authoritative references

- [Cloud Run container runtime contract](https://cloud.google.com/run/docs/container-contract)
- [Cloud Run authentication overview](https://cloud.google.com/run/docs/authenticating/overview)
- [Cloud Run service-to-service authentication](https://cloud.google.com/run/docs/authenticating/service-to-service)
- [Cloud Run service identity and ADC](https://cloud.google.com/run/docs/securing/service-identity)
- [Google Gen AI SDK on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview)
- [Cloud Storage checksum validation](https://cloud.google.com/storage/docs/data-validation)
- [Cloud Storage IAM](https://cloud.google.com/storage/docs/access-control/iam)
- [Workload Identity Federation for deployment pipelines](https://cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines)
- [Secret Manager access control](https://cloud.google.com/secret-manager/docs/manage-access-to-secrets)

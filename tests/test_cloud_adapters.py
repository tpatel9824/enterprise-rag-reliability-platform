import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from acmeflow_rag.artifact_store import (
    ArtifactValidationError,
    GCSArtifactStore,
    create_artifact_manifest,
    validate_artifacts,
)
from acmeflow_rag.auth import AuthenticationError, OIDCIdentityProvider, build_identity_provider
from acmeflow_rag.generator import GeminiGenerator
from acmeflow_rag.models import Chunk, RetrievalResult


def make_tfidf_artifacts(path: Path):
    path.mkdir()
    (path / "index.joblib").write_bytes(b"index")
    (path / "chunks.json").write_text(json.dumps([{"chunk_id": "one"}]), encoding="utf-8")
    return create_artifact_manifest(
        path, artifact_type="tfidf", corpus_version="test", chunk_count=1
    )


def test_artifact_manifest_detects_checksum_and_compatibility_errors(tmp_path):
    source = tmp_path / "source"
    make_tfidf_artifacts(source)
    assert validate_artifacts(source, "tfidf")["corpus_version"] == "test"
    with pytest.raises(ArtifactValidationError, match="does not match"):
        validate_artifacts(source, "dense")
    (source / "index.joblib").write_bytes(b"tampered")
    with pytest.raises(ArtifactValidationError, match="mismatch"):
        validate_artifacts(source, "tfidf")


class FakeBlob:
    def __init__(self, source=None):
        self.source = source
        self.download_checksums = []
        self.uploads = []

    def download_to_filename(self, filename, checksum=None):
        self.download_checksums.append(checksum)
        shutil.copyfile(self.source, filename)

    def upload_from_filename(self, filename, checksum=None, if_generation_match=None):
        self.uploads.append((Path(filename).name, checksum, if_generation_match))


class FakeBucket:
    def __init__(self, blobs):
        self.blobs = blobs

    def blob(self, name):
        return self.blobs[name]


class FakeStorageClient:
    def __init__(self, blobs):
        self.blobs = blobs

    def bucket(self, name):
        assert name == "bucket"
        return FakeBucket(self.blobs)


def test_gcs_download_uses_crc32c_then_verifies_sha256(tmp_path):
    source = tmp_path / "source"
    make_tfidf_artifacts(source)
    blobs = {
        f"prefix/{name}": FakeBlob(source / name)
        for name in ("artifact_manifest.json", "index.joblib", "chunks.json")
    }
    store = GCSArtifactStore("bucket", "prefix", tmp_path / "cache", FakeStorageClient(blobs))
    materialized = store.materialize("tfidf")
    assert validate_artifacts(materialized, "tfidf")["chunk_count"] == 1
    assert all(blob.download_checksums == ["crc32c"] for blob in blobs.values())


def test_gcs_upload_is_create_only_and_manifest_is_last(tmp_path):
    source = tmp_path / "source"
    make_tfidf_artifacts(source)
    names = ("index.joblib", "chunks.json", "artifact_manifest.json")
    blobs = {f"prefix/{name}": FakeBlob() for name in names}
    store = GCSArtifactStore("bucket", "prefix", tmp_path / "cache", FakeStorageClient(blobs))
    store.upload(source, "tfidf")
    for blob in blobs.values():
        assert blob.uploads[0][1:] == ("crc32c", 0)


def test_oidc_provider_uses_only_verified_claims_and_rejects_invalid_tokens():
    provider = OIDCIdentityProvider(
        issuer="https://issuer.example", audience="acmeflow", jwks_url="https://issuer/jwks",
        decoder=lambda token: {"sub": "verified-user", "roles": ["security"], "customer_id": "c1"},
    )
    principal = provider.authenticate("signed-token")
    assert principal.subject == "verified-user"
    assert principal.roles == {"security"}

    failing = OIDCIdentityProvider(
        issuer="https://issuer.example", audience="acmeflow", jwks_url="https://issuer/jwks",
        decoder=lambda token: (_ for _ in ()).throw(ValueError("bad signature")),
    )
    with pytest.raises(AuthenticationError):
        failing.authenticate("forged")


def test_local_authentication_is_disabled_in_production():
    settings = SimpleNamespace(
        auth_mode="local", app_environment="production", local_identities_path=Path("unused")
    )
    with pytest.raises(RuntimeError, match="disabled"):
        build_identity_provider(settings)


class FakeModels:
    def generate_content(self, **kwargs):
        return SimpleNamespace(text=json.dumps({
            "answer": "Approved.",
            "claims": [{"claim_id": "c1", "text": "Approved.", "cited_chunk_ids": ["chunk-1"]}],
        }))


def test_vertex_generator_parses_structured_claims_with_mock_client():
    client = SimpleNamespace(models=FakeModels())
    generator = GeminiGenerator("gemini-test", "project", "us-central1", client=client)
    chunk = Chunk(
        chunk_id="chunk-1", document_id="doc", title="Doc", department="Test", version="1",
        effective_date="2026-01-01", access_roles=("employee",), section="Rules",
        tags=("test",), text="Approval is required.",
    )
    generated = generator.generate("What is required?", [RetrievalResult(chunk, 1.0)])
    assert generated.claims[0].cited_chunk_ids == ("chunk-1",)

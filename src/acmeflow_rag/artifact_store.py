"""Validated local and Google Cloud Storage artifact transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


MANIFEST_NAME = "artifact_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class ArtifactValidationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_artifact_manifest(
    artifacts_dir: Path,
    *,
    artifact_type: str,
    corpus_version: str,
    chunk_count: int,
) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in artifacts_dir.iterdir() if item.is_file() and item.name != MANIFEST_NAME):
        files[path.name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "corpus_version": corpus_version,
        "chunk_count": chunk_count,
        "files": files,
    }
    (artifacts_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_artifacts(artifacts_dir: Path, expected_type: str | None = None) -> dict[str, Any]:
    manifest_path = artifacts_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ArtifactValidationError(f"Artifact manifest missing at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactValidationError("Unsupported artifact manifest schema")
    if expected_type and manifest.get("artifact_type") != expected_type:
        raise ArtifactValidationError(
            f"Artifact type {manifest.get('artifact_type')!r} does not match {expected_type!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactValidationError("Artifact manifest has no files")
    for name, metadata in files.items():
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
            raise ArtifactValidationError(f"Unsafe artifact filename {name!r}")
        path = artifacts_dir / name
        if not path.is_file():
            raise ArtifactValidationError(f"Artifact file missing: {name}")
        if path.stat().st_size != metadata.get("size"):
            raise ArtifactValidationError(f"Artifact size mismatch: {name}")
        if sha256_file(path) != metadata.get("sha256"):
            raise ArtifactValidationError(f"Artifact SHA-256 mismatch: {name}")
    required = {
        "tfidf": {"index.joblib", "chunks.json"},
        "dense": {"index.faiss", "chunks.json", "index_metadata.json"},
    }.get(manifest.get("artifact_type"), set())
    if not required.issubset(files):
        raise ArtifactValidationError(f"Artifact set is missing required files: {sorted(required - set(files))}")
    chunks = json.loads((artifacts_dir / "chunks.json").read_text(encoding="utf-8"))
    if len(chunks) != manifest.get("chunk_count"):
        raise ArtifactValidationError("Manifest chunk count does not match chunks.json")
    if manifest.get("artifact_type") == "dense":
        index_metadata = json.loads((artifacts_dir / "index_metadata.json").read_text(encoding="utf-8"))
        if index_metadata.get("chunk_count") != manifest.get("chunk_count"):
            raise ArtifactValidationError("Dense index metadata is incompatible with artifact manifest")
        if index_metadata.get("corpus_version") != manifest.get("corpus_version"):
            raise ArtifactValidationError("Dense corpus version is incompatible with artifact manifest")
    return manifest


class ArtifactStore(Protocol):
    def materialize(self, expected_type: str) -> Path: ...


@dataclass
class LocalArtifactStore:
    path: Path
    require_manifest: bool = False

    def materialize(self, expected_type: str) -> Path:
        if self.require_manifest or (self.path / MANIFEST_NAME).exists():
            validate_artifacts(self.path, expected_type)
        return self.path


class GCSArtifactStore:
    def __init__(self, bucket: str, prefix: str, cache_dir: Path, client=None):
        if not bucket or not prefix:
            raise ValueError("GCS bucket and prefix are required")
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self.cache_dir = cache_dir
        if client is None:
            from google.cloud import storage

            client = storage.Client()
        self.client = client

    def _blob(self, name: str):
        return self.client.bucket(self.bucket_name).blob(f"{self.prefix}/{name}")

    def materialize(self, expected_type: str) -> Path:
        target = self.cache_dir / expected_type
        staging = self.cache_dir / f".{expected_type}-staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        self._blob(MANIFEST_NAME).download_to_filename(
            str(staging / MANIFEST_NAME), checksum="crc32c"
        )
        manifest = json.loads((staging / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ArtifactValidationError("Unsupported artifact manifest schema")
        if manifest.get("artifact_type") != expected_type:
            raise ArtifactValidationError("Downloaded artifact type is incompatible")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files or len(files) > 20:
            raise ArtifactValidationError("Downloaded artifact manifest has an invalid file set")
        for name in files:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
                raise ArtifactValidationError(f"Unsafe artifact filename {name!r}")
            self._blob(name).download_to_filename(str(staging / name), checksum="crc32c")
        validate_artifacts(staging, expected_type)
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
        return target

    def upload(self, source: Path, expected_type: str) -> None:
        manifest = validate_artifacts(source, expected_type)
        for name in manifest["files"]:
            self._blob(name).upload_from_filename(
                str(source / name), checksum="crc32c", if_generation_match=0
            )
        self._blob(MANIFEST_NAME).upload_from_filename(
            str(source / MANIFEST_NAME), checksum="crc32c", if_generation_match=0
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a validated artifact set to GCS")
    parser.add_argument("upload", nargs="?")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--type", choices=("tfidf", "dense"), required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    GCSArtifactStore(args.bucket, args.prefix, Path(".artifact-cache")).upload(args.source, args.type)


if __name__ == "__main__":
    main()

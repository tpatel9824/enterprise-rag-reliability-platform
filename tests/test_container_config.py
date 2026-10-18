import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTAINER_PATHS = {
    "RAG_ARTIFACTS_DIR": "/app/artifacts",
    "LEXICAL_ARTIFACTS_DIR": "/app/artifacts",
    "DENSE_ARTIFACTS_DIR": "/app/artifacts",
    "LOCAL_IDENTITIES_PATH": "/app/data/identity/local_identities.json",
}


def test_dockerfile_defines_explicit_runtime_paths():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    for name, value in CONTAINER_PATHS.items():
        assert f"{name}={value}" in dockerfile


def test_settings_read_container_runtime_path_environment():
    environment = os.environ.copy()
    environment.update(CONTAINER_PATHS)
    script = """
import json
from acmeflow_rag.config import settings
print(json.dumps({
    "RAG_ARTIFACTS_DIR": settings.artifacts_dir.as_posix(),
    "LEXICAL_ARTIFACTS_DIR": settings.lexical_artifacts_dir.as_posix(),
    "DENSE_ARTIFACTS_DIR": settings.dense_artifacts_dir.as_posix(),
    "LOCAL_IDENTITIES_PATH": settings.local_identities_path.as_posix(),
}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == CONTAINER_PATHS

import tomllib
from pathlib import Path

from acmeflow_rag import __version__
from acmeflow_rag.api import app


def test_package_project_and_api_versions_match():
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["version"] == __version__
    assert app.version == __version__

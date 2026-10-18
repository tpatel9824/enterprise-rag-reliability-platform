from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_service_smoke_failure_retains_diagnostics_before_cleanup():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    service_start = "docker run -d --name acmeflow-rag-smoke"
    assert service_start in workflow
    assert "docker run --rm -d --name acmeflow-rag-smoke" not in workflow
    assert "docker logs acmeflow-rag-smoke" in workflow
    assert "docker inspect acmeflow-rag-smoke" in workflow
    assert "{{.State.ExitCode}}" in workflow
    assert "trap cleanup EXIT" in workflow
    assert workflow.index(service_start) < workflow.index("diagnose\n          exit 1")

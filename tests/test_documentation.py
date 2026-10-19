import re
from pathlib import Path

from acmeflow_rag.demo import build_parser as build_demo_parser
from acmeflow_rag.quality_evaluate import build_parser as build_evaluation_parser


ROOT = Path(__file__).parents[1]


def test_readme_local_markdown_links_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    local_targets = [target.split("#", 1)[0] for target in targets if "://" not in target]
    assert local_targets
    missing = [target for target in local_targets if not (ROOT / target).exists()]
    assert missing == []


def test_demo_and_quality_runner_share_documented_v3_artifact_defaults():
    demo = build_demo_parser().parse_args([])
    evaluation = build_evaluation_parser().parse_args([])
    assert demo.lexical_artifacts_dir == ROOT / "artifacts/v3/tfidf"
    assert demo.dense_artifacts_dir == ROOT / "artifacts/v3/dense"
    assert evaluation.lexical_artifacts_dir == demo.lexical_artifacts_dir
    assert evaluation.dense_artifacts_dir == demo.dense_artifacts_dir
    assert evaluation.output_dir == ROOT / "artifacts/evaluation/quality"


def test_public_documents_state_cloud_and_evaluation_limitations():
    roadmap = (ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8")
    evaluation = (ROOT / "docs/EVALUATION.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "mock-based tests only" in roadmap
    assert "No GCP project was modified" in roadmap
    assert "Independent review is pending" in evaluation
    assert "does not claim production readiness" in readme

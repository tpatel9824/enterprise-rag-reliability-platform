from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).parents[1]


def test_streamlit_demo_renders_controls_examples_and_verified_metrics():
    app = AppTest.from_file(str(ROOT / "src" / "acmeflow_rag" / "streamlit_app.py"))
    app.run(timeout=20)

    assert not app.exception
    assert app.selectbox(key="strategy_label").value == "Hybrid (RRF)"
    assert app.selectbox(key="identity").value == "Support agent"
    assert len(app.button) == 6
    assert any("Verified benchmark snapshot" in heading.value for heading in app.subheader)
    assert any("small synthetic benchmark" in info.value for info in app.info)


def test_streamlit_example_updates_question_and_governance_controls():
    app = AppTest.from_file(str(ROOT / "src" / "acmeflow_rag" / "streamlit_app.py"))
    app.run(timeout=20)
    app.button(key="example_authorization_denial").click().run(timeout=20)

    assert not app.exception
    assert app.text_area(key="question").value == "What approvals are required for production access?"
    assert app.selectbox(key="identity").value == "Employee"
    assert app.selectbox(key="department").value == "Security"


def test_streamlit_renders_grounded_answer_citation_and_evidence():
    app = AppTest.from_file(str(ROOT / "src" / "acmeflow_rag" / "streamlit_app.py"))
    app.session_state["last_response"] = {
        "answer": "Support verifies the account. [chunk-1]",
        "abstained": False,
        "reason_code": "ANSWER_GROUNDED",
        "source_document_ids": ["support_operations_2026_v2"],
        "evidence_passages": [{
            "document_id": "support_operations_2026_v2",
            "chunk_id": "chunk-1",
            "heading_path": ["Refund intake"],
            "text": "Support verifies the requester, account, charge, contract, and reason.",
            "relevance_score": 0.82,
        }],
        "claims": [{
            "claim_id": "claim-1",
            "text": "Support verifies the account.",
            "cited_chunk_ids": ["chunk-1"],
        }],
        "evidence_assessment": {"sufficient": True},
        "policy_selection": {"as_of": "2026-06-15"},
        "demo_metadata": {
            "strategy": "hybrid",
            "identity_label": "Support agent",
            "end_to_end_latency_ms": 12.34,
        },
    }
    app.run(timeout=20)

    assert not app.exception
    assert any("Answer" == heading.value for heading in app.subheader)
    assert any("Support verifies the account" in markdown.value for markdown in app.markdown)
    assert any("chunk-1" in markdown.value for markdown in app.markdown)
    assert len(app.expander) >= 2

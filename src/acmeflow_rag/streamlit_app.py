"""Professional local Streamlit interface for the AcmeFlow RAG service."""

from __future__ import annotations

from datetime import date

import streamlit as st

from acmeflow_rag import web_demo


st.set_page_config(
    page_title="AcmeFlow RAG Reliability Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .block-container {max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem;}
  [data-testid="stSidebar"] {border-right: 1px solid #d8e3e0;}
  .hero {padding: 1.55rem 1.7rem; border: 1px solid #c9dcd7; border-radius: 18px;
         background: linear-gradient(135deg, #f7fbfa 0%, #e5f3ef 100%); margin-bottom: 1.2rem;}
  .eyebrow {font-size: .76rem; letter-spacing: .12em; text-transform: uppercase;
            color: #0f766e; font-weight: 700; margin-bottom: .45rem;}
  .hero h1 {font-size: 2.2rem; line-height: 1.12; margin: 0 0 .55rem; color: #132a27;}
  .hero p {font-size: 1rem; color: #46615c; max-width: 760px; margin: 0;}
  .status-ok, .status-stop {padding: .9rem 1rem; border-radius: 12px; font-weight: 650;}
  .status-ok {background: #e4f5ed; color: #14532d; border: 1px solid #b9e3ce;}
  .status-stop {background: #fff2df; color: #7c3f00; border: 1px solid #efd1a4;}
  .evidence-meta {font-size: .82rem; color: #526a65;}
  div[data-testid="stMetric"] {background: white; border: 1px solid #d9e5e2; padding: .8rem;
                                border-radius: 12px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def backend() -> web_demo.DemoBackend:
    return web_demo.build_demo_backend()


@st.cache_data(show_spinner=False)
def verified_metrics() -> dict:
    return web_demo.load_verified_metrics()


def apply_example(example: web_demo.ExampleQuestion) -> None:
    st.session_state.question = example.question
    st.session_state.identity = example.identity_label
    st.session_state.as_of = example.as_of
    st.session_state.department = example.department or "All accessible departments"


def render_response(response: dict) -> None:
    metadata = response["demo_metadata"]
    sources = response.get("source_document_ids", [])
    status = "Abstained" if response["abstained"] else "Grounded answer"
    status_class = "status-stop" if response["abstained"] else "status-ok"
    st.markdown(
        f'<div class="{status_class}">{status} · {response["reason_code"]}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Strategy", metadata["strategy"].replace("_", " + ").title())
    cols[1].metric("End-to-end", f'{metadata["end_to_end_latency_ms"]:.2f} ms')
    cols[2].metric("Documents", len(sources))
    cols[3].metric("Evidence chunks", len(response.get("evidence_passages", [])))

    st.subheader("Answer")
    st.markdown(response["answer"])
    if response["abstained"]:
        st.caption(
            "The reason code comes from strategy-specific evidence assessment or governance. "
            "Retrieval scores are relevance signals, not calibrated probabilities."
        )

    if response.get("claims"):
        st.subheader("Claim-to-evidence map")
        for claim in response["claims"]:
            cited = ", ".join(f'`{chunk}`' for chunk in claim["cited_chunk_ids"])
            st.markdown(f'**{claim["claim_id"]}** — {claim["text"]}  \nCitations: {cited}')

    st.subheader("Authorized retrieved evidence")
    evidence = response.get("evidence_passages", [])
    if not evidence:
        st.info("No evidence was returned to the answer context.")
    for rank, passage in enumerate(evidence, 1):
        heading = " › ".join(passage.get("heading_path", [])) or "Document root"
        label = f'{rank}. {passage["document_id"]} · {heading}'
        with st.expander(label, expanded=rank == 1):
            st.markdown(passage["text"])
            st.markdown(
                f'<div class="evidence-meta">Chunk <code>{passage["chunk_id"]}</code> · '
                f'Relevance signal {passage["relevance_score"]:.4f}</div>',
                unsafe_allow_html=True,
            )

    with st.expander("Evidence and policy diagnostics"):
        assessment = response.get("evidence_assessment", {})
        policy = response.get("policy_selection", {})
        st.markdown("**Evidence assessment**")
        st.json(assessment, expanded=False)
        st.markdown("**Version and precedence selection**")
        st.json(policy, expanded=False)


st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Enterprise RAG reliability console</div>
  <h1>Evidence first. Authorized by design.</h1>
  <p>Explore grounded answers, claim-level citations, version-aware policy selection,
  authorization boundaries, and conservative abstention over a synthetic AcmeFlow corpus.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.caption(
    "Local demonstration · offline deterministic generation · synthetic documents · "
    "development identities only"
)

with st.sidebar:
    st.header("Query controls")
    strategy_label = st.selectbox(
        "Retrieval strategy", list(web_demo.STRATEGY_LABELS), index=2, key="strategy_label"
    )
    identity = st.selectbox(
        "Development identity", list(web_demo.IDENTITY_SUBJECTS), index=1, key="identity"
    )
    as_of = st.date_input("Policy as-of date", value=date(2026, 6, 15), key="as_of")
    department = st.selectbox(
        "Department scope",
        ["All accessible departments", "Product", "Customer Support", "Engineering", "Security", "HR"],
        key="department",
    )
    top_k = st.slider("Evidence passages", 1, 10, 5, key="top_k")
    st.divider()
    st.markdown("**Trust boundary**")
    st.caption(
        "Identity labels resolve server-side to local synthetic fixtures. The interface never "
        "accepts roles or displays development credentials."
    )

st.subheader("Prepared scenarios")
example_columns = st.columns(len(web_demo.EXAMPLES))
for column, example in zip(example_columns, web_demo.EXAMPLES):
    with column:
        st.markdown(f"**{example.label}**")
        st.caption(example.description)
        st.button(
            "Load",
            key=f"example_{example.key}",
            use_container_width=True,
            on_click=apply_example,
            args=(example,),
        )

st.subheader("Ask the knowledge base")
if "question" not in st.session_state:
    st.session_state.question = web_demo.EXAMPLES[0].question
question = st.text_area(
    "Question",
    key="question",
    height=105,
    placeholder="Ask about a policy, procedure, or control…",
)
ask_clicked = st.button("Run grounded query", type="primary", key="run_query")

if ask_clicked:
    if not question.strip():
        st.warning("Enter a question before running retrieval.")
    else:
        try:
            with st.spinner("Selecting eligible policies and retrieving authorized evidence…"):
                result = backend().ask(
                    question=question,
                    strategy=web_demo.STRATEGY_LABELS[strategy_label],
                    identity_label=identity,
                    as_of=as_of,
                    department=None if department == "All accessible departments" else department,
                    top_k=top_k,
                )
            st.session_state.last_response = result
        except FileNotFoundError as exc:
            st.error(f"Required index artifacts are missing. Build the v3 indexes first. Details: {exc}")
        except Exception as exc:
            st.error(f"The local demo could not complete the query: {type(exc).__name__}: {exc}")

if "last_response" in st.session_state:
    st.divider()
    render_response(st.session_state.last_response)

st.divider()
st.subheader("Verified benchmark snapshot")
metrics = verified_metrics()
st.caption(
    f'{metrics["case_count"]} project-authored synthetic cases · corpus {metrics["corpus"]} · '
    f'top-{metrics["top_k"]} · {metrics["generator"]}'
)
rows = []
for item in metrics["strategies"]:
    rows.append({
        "Strategy": item["strategy"],
        "Recall@1": f'{item["recall_at_1"]:.3f}',
        "Recall@5": f'{item["recall_at_5"]:.3f}',
        "MRR": f'{item["mrr"]:.3f}',
        "Full evidence@5": f'{item["full_evidence_at_5"]:.3f}',
        "Abstention F1": f'{item["abstention_f1"]:.3f}',
        "Forbidden hits": item["forbidden_hits"],
        "Mean / P95 latency": f'{item["mean_latency_ms"]:.2f} / {item["p95_latency_ms"]:.2f} ms',
    })
st.dataframe(rows, hide_index=True, use_container_width=True)
st.caption(f'{metrics["latency_scope"]} {metrics["review_status"]}')
st.info(
    "This small synthetic benchmark supports regression testing, not broad performance claims. "
    "Citation validity confirms IDs are present in authorized context; it does not prove entailment."
)

"""
app.py
=======
ConstiGraph — a multi-agent, graph-RAG assistant for the Constitution of
India, built with LangGraph agents, the Groq Cloud LLM API, and a NetworkX
knowledge graph, presented through Streamlit.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import traceback

import streamlit as st
import streamlit.components.v1 as components

from src.agents.graph_workflow import build_workflow
from src.graph_store import ConstitutionGraphStore
from src.llm_client import DEFAULT_MODEL, GroqLLMClient
from src.viz import render_graph_html

st.set_page_config(page_title="ConstiGraph", page_icon="⚖️", layout="wide")

MODE_LABELS = {
    "Q&A (plain language)": "qa",
    "Legal research briefing": "research",
    "Debate (for vs. against)": "debate",
}

# When a shared deployment key is in use, cap how many questions a single
# browser session can ask, so one visitor can't run up unbounded cost on
# the app owner's key. Override via st.secrets["MAX_QUERIES_PER_SESSION"]
# or the same-named env var; 0 or unset disables the cap (unlimited).
def get_session_query_cap() -> int:
    try:
        raw = st.secrets.get("MAX_QUERIES_PER_SESSION", "")
    except Exception:
        raw = ""
    raw = raw or os.environ.get("MAX_QUERIES_PER_SESSION", "20")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 20

# Curated so at least one lands cleanly in each mode's sweet spot, and every
# one of them is guaranteed to hit real nodes in the current graph (Part I-III,
# Articles 1-35) — see the Limitations panel for why the graph stops there.
SUGGESTED_QUESTIONS = [
    "Can the government restrict freedom of speech, and under what conditions?",
    "What protections exist against arbitrary arrest and detention?",
    "Does the Constitution guarantee free education for children?",
    "What is the significance of the right to life under Article 21?",
    "How does someone become a citizen of India under the Constitution?",
    "Should reservations in public employment be considered discrimination?",
    "Can a religious minority run its own educational institution?",
    "Is untouchability legally abolished, and what enforcement exists?",
]


# ---------------------------------------------------------------------- #
# Cached resources
# ---------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_store() -> ConstitutionGraphStore:
    return ConstitutionGraphStore()


def get_deployment_api_key() -> str:
    """
    Resolve a Groq API key the app owner configured for this deployment,
    without ever putting it in a widget the user could read back out.

    Resolution order:
      1. Streamlit Cloud secrets (st.secrets["GROQ_API_KEY"]) — the
         recommended path for Streamlit Community Cloud: set this in your
         app's Settings -> Secrets, it's encrypted at rest and never shows
         up in the repo or the client.
      2. A GROQ_API_KEY environment variable (useful for other hosts, or
         local `.env` via python-dotenv).
      3. Empty string — in that case the sidebar falls back to asking the
         visitor for their own key, so the app still works when no
         deployment key is configured (e.g. running locally for dev).
    """
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        # st.secrets raises if no secrets.toml exists at all (e.g. local
        # dev with only a .env file) — that's fine, just fall through.
        secret_key = ""
    return secret_key or os.environ.get("GROQ_API_KEY", "")


def get_deployment_model() -> str:
    try:
        return st.secrets.get("GROQ_MODEL", "") or DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def get_llm(api_key: str, model: str) -> GroqLLMClient:
    return GroqLLMClient(api_key=api_key, model=model)


def run_question(question: str, mode: str, mode_label: str, api_key: str, model: str, is_shared_key: bool) -> None:
    if not api_key:
        st.error("Please provide a Groq API key in the sidebar (get one free at console.groq.com/keys).")
        return

    if is_shared_key:
        cap = get_session_query_cap()
        if cap and st.session_state.get("query_count", 0) >= cap:
            st.warning(
                f"You've reached this session's limit of {cap} questions on the shared demo key. "
                "Refresh the page to reset, or use your own free Groq API key in the sidebar for unlimited use."
            )
            return

    prior_question = st.session_state.history[0]["question"] if st.session_state.history else None
    with st.spinner(f"Running the {mode_label.lower()} pipeline..."):
        try:
            llm = get_llm(api_key, model)
            app = build_workflow(llm, get_store())
            result = app.invoke({"question": question, "mode": mode, "prior_question": prior_question})
            st.session_state.history.insert(0, {"question": question, "mode": mode_label, "result": result})
            if is_shared_key:
                st.session_state.query_count = st.session_state.get("query_count", 0) + 1
        except Exception as exc:  # surface a readable error in the UI
            st.error(f"Something went wrong: {exc}")
            st.code(traceback.format_exc())


# ---------------------------------------------------------------------- #
# Session state bootstrap
# ---------------------------------------------------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ⚖️ ConstiGraph")
    st.caption("A graph-RAG, multi-agent constitutional assistant.")

    deployment_key = get_deployment_api_key()
    is_shared_key = bool(deployment_key)
    if deployment_key:
        # Deployment key is present (Streamlit Cloud secrets / env var) —
        # never render it in any widget, just confirm it's active.
        api_key = deployment_key
        model = get_deployment_model()
        cap = get_session_query_cap()
        used = st.session_state.get("query_count", 0)
        st.caption("🔑 Using the Groq API key configured for this deployment.")
        if cap:
            st.caption(f"Session usage: {used}/{cap} questions.")
    else:
        # No deployment key found (e.g. local dev without secrets.toml) —
        # fall back to letting the visitor supply their own, same as before.
        st.caption("No deployment API key configured — enter your own to try it out.")
        api_key = st.text_input(
            "Groq API key", value="", type="password", help="https://console.groq.com/keys"
        )
        model = st.text_input("Groq model", value=DEFAULT_MODEL)

    mode_label = st.radio("Mode", list(MODE_LABELS.keys()))
    mode = MODE_LABELS[mode_label]

    graph_layout = st.radio(
        "Graph layout",
        ["Hierarchical (recommended)", "Physics (organic)"],
        help="Hierarchical lays the graph out top-down by legal depth: Part → Article → Clause → Subclause.",
    )
    layout_value = "hierarchical" if graph_layout.startswith("Hierarchical") else "physics"

    st.divider()
    st.markdown(
        "**Pipeline**\n\n"
        "1. **Preprocess** — normalize, resolve pronouns against your last "
        "question, detect language.\n"
        "2. **Analyze** (LLM) — intent, complexity, entities, confidence.\n"
        "3. **Plan** — choose retrieval branch(es): Direct KG lookup, "
        "Hybrid GraphRAG (graph + TF-IDF vector search), and/or an "
        "external-tool branch for out-of-scope questions.\n"
        "4. **Fuse + rerank** the evidence from whichever branches ran.\n"
        "5. **Generate** the answer (Q&A / Research / Debate), grounded "
        "only in that evidence.\n"
        "6. **Verify** — a faithfulness check flags any claim or citation "
        "not actually backed by the retrieved evidence."
    )

    with st.expander("⚠️ Limitations — please read"):
        st.markdown(
            "- **Partial document.** The graph currently covers Parts I–III "
            "(Articles 1–35: Union & territory, citizenship, fundamental "
            "rights). Later Parts (Directive Principles, the Judiciary, "
            "Panchayats, Schedules, etc.) are **not** in the graph — the "
            "planner will try the external-tool branch for these, but by "
            "default that branch is a no-op (see below).\n"
            "- **No case law by default.** Only the bare constitutional "
            "text is modelled. The external-tool branch exists to fetch "
            "case law/current context, but ships **disabled** — set a "
            "`TAVILY_API_KEY` (or wire in your own adapter) to enable it. "
            "Without it, case-law questions will honestly say so rather "
            "than fabricate a citation.\n"
            "- **The 'vector DB' is local TF-IDF, not real embeddings.** "
            "It helps with paraphrased wording but is not semantic search "
            "in the deep-learning sense.\n"
            "- **Structural links only** in the graph itself — edges "
            "capture containment (Part→Article→Clause), not cross-"
            "references between articles.\n"
            "- **LLMs can still hallucinate.** The faithfulness-verification "
            "step is a second LLM call checking the first one, not a "
            "formal guarantee — always check the cited Article/Clause "
            "yourself.\n"
            "- **Debate mode isn't a verdict.** It illustrates competing "
            "interpretations, not a prediction of how a court would rule.\n"
            "- **Not legal advice**, and not affiliated with any court, "
            "government body, or bar association."
        )

    st.divider()
    st.caption("Educational tool — not legal advice.")

store = get_store()

# ---------------------------------------------------------------------- #
# Main layout
# ---------------------------------------------------------------------- #
st.title("Constitution of India — Graph-RAG Assistant")
st.caption(
    "Ask a question about a fundamental right or constitutional provision. "
    "The assistant runs a multi-stage pipeline — analysis, retrieval "
    "planning, graph + vector search, fusion, reranking, generation, and "
    "faithfulness verification — before answering."
)

col_chat, col_graph = st.columns([3, 2], gap="large")

with col_chat:
    with st.expander("💡 Try a suggested question", expanded=not st.session_state.history):
        btn_cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if btn_cols[i % 2].button(q, key=f"suggested_{i}", use_container_width=True):
                st.session_state.question_input = q
                st.session_state.pending_question = q
                st.rerun()

    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="e.g. Can the government restrict freedom of speech, and under what conditions?",
    )
    ask_clicked = st.button("Ask ConstiGraph", type="primary", use_container_width=True)

    # A suggested-question click also fires the query immediately, same as
    # typing your own question and clicking Ask.
    trigger_question = None
    if st.session_state.pending_question:
        trigger_question = st.session_state.pending_question
        st.session_state.pending_question = None
    elif ask_clicked and question.strip():
        trigger_question = question.strip()

    if trigger_question:
        run_question(trigger_question, mode, mode_label, api_key, model, is_shared_key)

    for turn in st.session_state.history:
        result = turn["result"]
        with st.container(border=True):
            st.markdown(f"**Q ({turn['mode']}):** {turn['question']}")
            st.markdown(result.get("final_answer", "_No answer produced._"))

            citations = result.get("citations") or []
            if citations:
                st.caption("Cited: " + ", ".join(citations))

            verification = result.get("verification") or {}
            if verification:
                if verification.get("faithful", True):
                    st.caption("✅ Faithfulness check passed")
                else:
                    st.caption("⚠️ Faithfulness check flagged issues (see answer above)")

            with st.expander("Pipeline details"):
                plan = result.get("retrieval_plan", {})
                st.markdown(
                    f"- **Intent:** {result.get('intent', '—')}  \n"
                    f"- **Complexity:** {result.get('complexity', '—')}  \n"
                    f"- **Analyzer confidence:** {result.get('confidence', 0):.2f}  \n"
                    f"- **Concepts detected:** {', '.join(result.get('constitutional_concepts', [])) or '—'}  \n"
                    f"- **Article(s) named:** {', '.join(result.get('entities', {}).get('article_numbers', [])) or '—'}  \n"
                    f"- **Retrieval branches used:** "
                    f"{'Direct KG, ' if plan.get('run_direct_kg') else ''}"
                    f"{'Hybrid GraphRAG, ' if plan.get('run_hybrid_graphrag') else ''}"
                    f"{'External tools' if plan.get('run_mcp_tools') else ''}".rstrip(", ") + "  \n"
                    f"- **Planner reasoning:** {plan.get('reason', '—')}"
                )
                evidence = result.get("reranked_evidence", [])
                if evidence:
                    st.caption(f"Top evidence ({len(evidence)} items):")
                    for e in evidence[:6]:
                        st.text(f"  [{e['citation']}] score={e['score']} sources={e['sources']}")

with col_graph:
    st.subheader("Knowledge graph")
    latest_ids = set()
    if st.session_state.history:
        latest_result = st.session_state.history[0]["result"]
        latest_ids = {n["id"] for n in latest_result.get("reranked_evidence", []) if n.get("group") != "external"}
        st.caption("Highlighted nodes were retrieved for your most recent question.")
    else:
        st.caption("Full graph shown. Ask a question to see retrieval highlighted.")

    html = render_graph_html(store, highlight_ids=latest_ids, layout=layout_value)
    components.html(html, height=580, scrolling=False)

    legend_cols = st.columns(4)
    for col, (group, color) in zip(legend_cols, [
        ("Part", "#ffd166"), ("Article", "#06d6a0"), ("Clause", "#118ab2"), ("Subclause", "#ef476f")
    ]):
        col.markdown(
            f"<span style='color:{color}; font-size:20px;'>&#9679;</span> {group}",
            unsafe_allow_html=True,
        )

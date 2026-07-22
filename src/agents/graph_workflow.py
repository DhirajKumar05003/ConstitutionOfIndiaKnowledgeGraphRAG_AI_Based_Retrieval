"""
graph_workflow.py
===================
The LangGraph state machine implementing the full retrieval-and-generation
pipeline:

    preprocess -> analyze -> plan ──▶ direct_kg ────┐
                                   ├─▶ hybrid_graphrag ─┼─▶ fuse ─▶ rerank ─▶ route ─▶ [qa|research|debate*] ─▶ verify ─▶ compose ─▶ END
                                   └─▶ mcp_tools ───────┘

    * debate is itself a 3-step sub-chain: debate_for -> debate_against -> debate_judge

Design notes:
- `plan` fans out to all three retrieval branches unconditionally; each
  branch checks the `retrieval_plan` itself and simply returns empty
  results if it wasn't selected. This keeps the fan-out/fan-in shape of
  the architecture diagram intact under LangGraph's Pregel-style execution
  (a node with 3 outgoing edges runs all 3 targets in the same superstep;
  a node with 3 incoming edges runs once, after all 3 have completed).
- Every agent node is a thin function: pull what it needs out of
  `AgentState`, optionally call the Groq LLM with a role-specific system
  prompt, write its output back into the state. Retrieval, fusion,
  reranking, and prompt-context building are all pure Python (fast, free,
  fully inspectable) — the LLM budget is reserved for query analysis,
  answer generation, and faithfulness verification, where judgment is
  actually needed.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from langgraph.graph import END, StateGraph

from src.agents import prompts
from src.agents.state import AgentState
from src.graph_store import ConstitutionGraphStore, RetrievedNode
from src.llm_client import GroqLLMClient
from src.query_preprocessing import preprocess
from src.retrieval.fusion import EvidenceItem, citations_from, format_context, fuse_evidence
from src.retrieval.planner import plan_retrieval
from src.retrieval.reranker import rerank
from src.tools.mcp_adapter import ToolResult, get_default_adapter
from src.vector_index import VectorIndex


def build_workflow(llm: GroqLLMClient, store: ConstitutionGraphStore):
    """Construct and compile the LangGraph workflow bound to a given LLM + graph store."""

    vector_index = VectorIndex(store)
    tool_adapter = get_default_adapter()

    # ------------------------------------------------------------------ #
    # 1. Query Preprocessor
    # ------------------------------------------------------------------ #
    def preprocess_node(state: AgentState) -> dict:
        return preprocess(state["question"], state.get("prior_question"))

    # ------------------------------------------------------------------ #
    # 2. Query Analyzer (LLM)
    # ------------------------------------------------------------------ #
    def analyze_node(state: AgentState) -> dict:
        q = state["resolved_question"]
        article_numbers = store.extract_article_numbers(q)
        default = {
            "intent": state.get("mode", "qa"),
            "complexity": "simple",
            "constitutional_concepts": [],
            "case_names": [],
            "confidence": 0.6,
        }
        analysis = llm.chat_json(prompts.ANALYZER_SYSTEM_PROMPT, q, default=default, temperature=0.0, max_tokens=350)
        entities = {
            "article_numbers": article_numbers,
            "case_names": analysis.get("case_names") or [],
        }
        return {
            "intent": analysis.get("intent", "qa"),
            "complexity": analysis.get("complexity", "simple"),
            "constitutional_concepts": analysis.get("constitutional_concepts") or [],
            "confidence": float(analysis.get("confidence", 0.6) or 0.6),
            "entities": entities,
        }

    # ------------------------------------------------------------------ #
    # 3. Retrieval Planner
    # ------------------------------------------------------------------ #
    def plan_node(state: AgentState) -> dict:
        plan = plan_retrieval(
            {
                "entities": state.get("entities", {}),
                "confidence": state.get("confidence", 0.6),
                "intent": state.get("intent", "qa"),
            }
        )
        return {"retrieval_plan": dataclasses.asdict(plan)}

    # ------------------------------------------------------------------ #
    # 4. Retrieval branches (run in parallel; each is a no-op if unselected)
    # ------------------------------------------------------------------ #
    def direct_kg_node(state: AgentState) -> dict:
        plan = state.get("retrieval_plan", {})
        if not plan.get("run_direct_kg"):
            return {"direct_kg_nodes": []}
        nodes: list[RetrievedNode] = []
        for num in plan.get("article_numbers", []):
            nodes.extend(store.get_article_and_descendants(num))
        return {"direct_kg_nodes": [n.to_dict() for n in nodes]}

    def hybrid_graphrag_node(state: AgentState) -> dict:
        plan = state.get("retrieval_plan", {})
        if not plan.get("run_hybrid_graphrag"):
            return {"hybrid_nodes": []}
        q = state["resolved_question"]

        keyword_hits = store.search(q, top_k=6)
        seed_scores = {n.id: n.score for n in keyword_hits}
        expanded = store.expand_context([n.id for n in keyword_hits], seed_scores=seed_scores) if keyword_hits else []
        vector_hits = [h.node for h in vector_index.query(q, top_k=6)]

        merged: dict[str, RetrievedNode] = {n.id: n for n in expanded}
        for n in keyword_hits + vector_hits:
            if n.id in merged:
                merged[n.id].score = max(merged[n.id].score, n.score)
            else:
                merged[n.id] = n

        nodes = sorted(merged.values(), key=lambda n: n.score, reverse=True)[:15]
        return {"hybrid_nodes": [n.to_dict() for n in nodes]}

    def mcp_tools_node(state: AgentState) -> dict:
        plan = state.get("retrieval_plan", {})
        if not plan.get("run_mcp_tools"):
            return {"mcp_results": []}
        results = tool_adapter.search(state["resolved_question"], max_results=3)
        return {"mcp_results": [dataclasses.asdict(r) for r in results]}

    # ------------------------------------------------------------------ #
    # 5. Evidence Fusion Layer
    # ------------------------------------------------------------------ #
    def fuse_node(state: AgentState) -> dict:
        direct = [RetrievedNode(**d) for d in state.get("direct_kg_nodes", [])]
        hybrid = [RetrievedNode(**d) for d in state.get("hybrid_nodes", [])]
        mcp_results = [ToolResult(**d) for d in state.get("mcp_results", [])]
        fused = fuse_evidence(direct, hybrid, mcp_results)
        return {"fused_evidence": [e.to_dict() for e in fused]}

    # ------------------------------------------------------------------ #
    # 6. Reranker / Relevance Filter
    # ------------------------------------------------------------------ #
    def rerank_node(state: AgentState) -> dict:
        items = [EvidenceItem(**d) for d in state.get("fused_evidence", [])]
        ranked = rerank(items, state["resolved_question"], top_k=12)
        return {
            "reranked_evidence": [e.to_dict() for e in ranked],
            "context": format_context(ranked),
            "citations": citations_from(ranked),
        }

    # ------------------------------------------------------------------ #
    # Routing (Prompt Builder happens inline inside each generation node)
    # ------------------------------------------------------------------ #
    def route_mode(state: AgentState) -> Literal["qa", "research", "debate"]:
        mode = state.get("mode", "qa")
        if mode in ("qa", "research", "debate"):
            return mode
        intent = state.get("intent", "qa")
        return intent if intent in ("qa", "research", "debate") else "qa"

    def _analysis_note(state: AgentState) -> str:
        concepts = ", ".join(state.get("constitutional_concepts", [])) or "none identified"
        return (
            f"(Query analyzer: intent={state.get('intent')}, "
            f"confidence={state.get('confidence', 0):.2f}, concepts={concepts}. "
            f"Note: this graph only covers Parts I-III / Articles 1-35.)"
        )

    # ------------------------------------------------------------------ #
    # 7. Response Generation LLM (Prompt Builder + Citations happens here)
    # ------------------------------------------------------------------ #
    def qa_node(state: AgentState) -> dict:
        if not state.get("context"):
            text = "I couldn't find a relevant provision in the Constitution graph (Parts I-III) for this question."
            return {"qa_answer": text, "draft_answer": text}
        user_prompt = f"Question: {state['resolved_question']}\n{_analysis_note(state)}\n\nContext:\n{state['context']}"
        resp = llm.chat(prompts.QA_SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=500)
        return {"qa_answer": resp.text, "draft_answer": resp.text}

    def research_node(state: AgentState) -> dict:
        if not state.get("context"):
            text = "No relevant provisions were retrieved from the knowledge graph (Parts I-III only)."
            return {"research_answer": text, "draft_answer": text}
        user_prompt = f"Research question: {state['resolved_question']}\n{_analysis_note(state)}\n\nContext:\n{state['context']}"
        resp = llm.chat(prompts.RESEARCH_SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=800)
        return {"research_answer": resp.text, "draft_answer": resp.text}

    def debate_for_node(state: AgentState) -> dict:
        user_prompt = f"Debate question: {state['resolved_question']}\n\nContext:\n{state['context']}"
        resp = llm.chat(prompts.DEBATE_FOR_SYSTEM_PROMPT, user_prompt, temperature=0.4, max_tokens=350)
        return {"debate_for": resp.text}

    def debate_against_node(state: AgentState) -> dict:
        user_prompt = f"Debate question: {state['resolved_question']}\n\nContext:\n{state['context']}"
        resp = llm.chat(prompts.DEBATE_AGAINST_SYSTEM_PROMPT, user_prompt, temperature=0.4, max_tokens=350)
        return {"debate_against": resp.text}

    def debate_judge_node(state: AgentState) -> dict:
        user_prompt = (
            f"Debate question: {state['resolved_question']}\n\n"
            f"Advocate FOR expansive interpretation said:\n{state.get('debate_for', '')}\n\n"
            f"Advocate FOR restrictive interpretation said:\n{state.get('debate_against', '')}\n\n"
            f"Context:\n{state['context']}"
        )
        resp = llm.chat(prompts.JUDGE_SYSTEM_PROMPT, user_prompt, temperature=0.2, max_tokens=350)
        combined = (
            f"For an expansive reading: {state.get('debate_for', '')}\n"
            f"For a restrictive reading: {state.get('debate_against', '')}\n"
            f"Moderator's synthesis: {resp.text}"
        )
        return {"debate_verdict": resp.text, "draft_answer": combined}

    # ------------------------------------------------------------------ #
    # 8. Faithfulness Verification
    # ------------------------------------------------------------------ #
    def verify_node(state: AgentState) -> dict:
        draft = state.get("draft_answer", "")
        default = {"faithful": True, "issues": [], "unsupported_citations": []}
        if not draft or not state.get("context"):
            return {"verification": default}
        user_prompt = f"Draft answer:\n{draft}\n\nEvidence context:\n{state['context']}"
        result = llm.chat_json(prompts.VERIFIER_SYSTEM_PROMPT, user_prompt, default=default, temperature=0.0, max_tokens=300)
        return {"verification": result}

    # ------------------------------------------------------------------ #
    # Final composition
    # ------------------------------------------------------------------ #
    def compose_node(state: AgentState) -> dict:
        mode = route_mode(state)
        if mode == "qa":
            final = state.get("qa_answer", "")
        elif mode == "research":
            final = state.get("research_answer", "")
        else:  # debate
            final = (
                f"**For an expansive reading:**\n{state.get('debate_for', '')}\n\n"
                f"**For a restrictive reading:**\n{state.get('debate_against', '')}\n\n"
                f"**Moderator's synthesis:**\n{state.get('debate_verdict', '')}"
            )

        verification = state.get("verification") or {}
        if verification and not verification.get("faithful", True):
            issues = verification.get("issues") or []
            detail = "; ".join(issues) if issues else "unsupported claims detected"
            final += f"\n\n---\n⚠️ **Faithfulness check flagged this answer:** {detail}"

        return {"final_answer": final}

    # ------------------------------------------------------------------ #
    # Graph assembly
    # ------------------------------------------------------------------ #
    graph = StateGraph(AgentState)

    graph.add_node("preprocess", preprocess_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("plan", plan_node)
    graph.add_node("direct_kg", direct_kg_node)
    graph.add_node("hybrid_graphrag", hybrid_graphrag_node)
    graph.add_node("mcp_tools", mcp_tools_node)
    graph.add_node("fuse", fuse_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("qa_agent", qa_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("debate_for", debate_for_node)
    graph.add_node("debate_against", debate_against_node)
    graph.add_node("debate_judge", debate_judge_node)
    graph.add_node("verify", verify_node)
    graph.add_node("compose", compose_node)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "analyze")
    graph.add_edge("analyze", "plan")

    # Fan-out: the three retrieval branches
    graph.add_edge("plan", "direct_kg")
    graph.add_edge("plan", "hybrid_graphrag")
    graph.add_edge("plan", "mcp_tools")

    # Fan-in: fusion waits for all three
    graph.add_edge("direct_kg", "fuse")
    graph.add_edge("hybrid_graphrag", "fuse")
    graph.add_edge("mcp_tools", "fuse")

    graph.add_edge("fuse", "rerank")

    graph.add_conditional_edges(
        "rerank",
        route_mode,
        {"qa": "qa_agent", "research": "research_agent", "debate": "debate_for"},
    )

    graph.add_edge("qa_agent", "verify")
    graph.add_edge("research_agent", "verify")
    graph.add_edge("debate_for", "debate_against")
    graph.add_edge("debate_against", "debate_judge")
    graph.add_edge("debate_judge", "verify")

    graph.add_edge("verify", "compose")
    graph.add_edge("compose", END)

    return graph.compile()


def run_query(llm: GroqLLMClient, store: ConstitutionGraphStore, question: str, mode: str = "qa", prior_question: str | None = None) -> AgentState:
    """Convenience helper: build (or reuse) the workflow and run it once."""
    app = build_workflow(llm, store)
    result = app.invoke({"question": question, "mode": mode, "prior_question": prior_question})
    return result

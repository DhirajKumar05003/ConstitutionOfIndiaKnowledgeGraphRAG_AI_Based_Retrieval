"""
planner.py
===========
The Retrieval Planner node: decides, per question, which of the three
retrieval branches to run — Direct KG lookup, Hybrid GraphRAG
(graph keyword search + TF-IDF vector search), and/or MCP Tools
(external search) — based on the Query Analyzer's output.

Rules (deliberately simple and inspectable rather than another LLM call —
this is a routing decision, not a reasoning task):

  - Any explicit "Article N" entity -> always run Direct KG lookup for it.
  - Always run Hybrid GraphRAG as the general-purpose fallback, UNLESS the
    analyzer's confidence is high AND a direct-KG hit fully covers the
    question (single, clearly-named article, high confidence).
  - Run MCP Tools only when the analyzer flags the question as being about
    something the graph structurally cannot contain — case law / judicial
    precedent, current-affairs, or anything scored low-confidence /
    out-of-scope. The graph only covers Parts I-III of the Constitution, so
    "out of scope" also covers later Parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalPlan:
    run_direct_kg: bool
    run_hybrid_graphrag: bool
    run_mcp_tools: bool
    article_numbers: list[str] = field(default_factory=list)
    reason: str = ""


OUT_OF_SCOPE_INTENTS = {"case_law", "out_of_scope", "current_affairs"}


def plan_retrieval(analysis: dict) -> RetrievalPlan:
    article_numbers = analysis.get("entities", {}).get("article_numbers", [])
    confidence = float(analysis.get("confidence", 0.5) or 0.5)
    intent = analysis.get("intent", "qa")
    mentions_case_law = bool(analysis.get("entities", {}).get("case_names")) or intent in OUT_OF_SCOPE_INTENTS

    run_direct_kg = bool(article_numbers)
    run_mcp_tools = mentions_case_law or confidence < 0.35

    # Skip the broader graph/vector search only in the narrow case of a
    # single, clearly-named article with high analyzer confidence — direct
    # lookup alone already gives full, precise context for that case.
    run_hybrid_graphrag = not (run_direct_kg and len(article_numbers) == 1 and confidence >= 0.75)

    reasons = []
    if run_direct_kg:
        reasons.append(f"named article(s) {article_numbers} -> direct KG lookup")
    if run_hybrid_graphrag:
        reasons.append("hybrid graph+vector search for broader/paraphrased coverage")
    if run_mcp_tools:
        reasons.append("question looks out-of-graph-scope (case law / low confidence) -> external tool branch")
    if not reasons:
        reasons.append("defaulting to hybrid graph+vector search")
        run_hybrid_graphrag = True

    return RetrievalPlan(
        run_direct_kg=run_direct_kg,
        run_hybrid_graphrag=run_hybrid_graphrag,
        run_mcp_tools=run_mcp_tools,
        article_numbers=article_numbers,
        reason="; ".join(reasons),
    )

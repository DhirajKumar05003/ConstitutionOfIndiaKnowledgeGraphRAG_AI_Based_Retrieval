"""
state.py
=========
The shared state object threaded through every node of the LangGraph
workflow. Using a single TypedDict keeps every agent's input/output
contract explicit and testable in isolation.

Field groups mirror the pipeline stages in order: preprocessing ->
analysis -> planning -> retrieval (x3 branches) -> fusion -> reranking ->
generation -> verification -> final.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ---- input ----
    question: str
    mode: str  # "qa" | "research" | "debate" — user/UI-selected response style
    prior_question: Optional[str]  # last turn's question, for pronoun resolution

    # ---- 1. query preprocessing ----
    raw_question: str
    normalized_question: str
    resolved_question: str
    language: str

    # ---- 2. query analysis (LLM) ----
    intent: str
    complexity: str
    entities: dict  # {"article_numbers": [...], "case_names": [...]}
    constitutional_concepts: List[str]
    confidence: float

    # ---- 3. retrieval planning ----
    retrieval_plan: dict

    # ---- 4. retrieval branches ----
    direct_kg_nodes: List[dict]
    hybrid_nodes: List[dict]
    mcp_results: List[dict]

    # ---- 5. fusion + 6. rerank ----
    fused_evidence: List[dict]
    reranked_evidence: List[dict]
    context: str
    citations: List[str]

    # ---- 7. generation ----
    qa_answer: str
    research_answer: str
    debate_for: str
    debate_against: str
    debate_verdict: str
    draft_answer: str

    # ---- 8. faithfulness verification ----
    verification: dict

    # ---- final ----
    final_answer: str
    error: Optional[str]

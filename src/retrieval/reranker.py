"""
reranker.py
============
The Reranker / Relevance Filter node: a final scoring pass over the fused
evidence before it goes into the prompt.

Kept as a fast heuristic reranker (multi-source agreement bonus + a fresh
term-overlap check against the *resolved* question) rather than another LLM
call — with only a handful of candidates, an LLM rerank would cost more
than it buys. The extension point for a learned/LLM reranker is noted below
if the evidence pool ever gets large enough to need one.
"""

from __future__ import annotations

import re

from src.retrieval.fusion import EvidenceItem


def rerank(evidence: list[EvidenceItem], question: str, top_k: int = 12) -> list[EvidenceItem]:
    """
    Re-score fused evidence and return the top_k, most-relevant-first.

    To upgrade this to an LLM-based reranker: replace the loop body with a
    single batched call that asks the model to score each (question,
    evidence.text) pair 0-1, keeping the same EvidenceItem shape in/out so
    nothing downstream (prompt builder, UI) needs to change.
    """
    terms = [t for t in re.split(r"\W+", question.lower()) if len(t) > 2]

    for item in evidence:
        text_lower = item.text.lower()
        overlap = sum(1 for t in terms if t in text_lower) / max(len(terms), 1)
        multi_source_bonus = 1.5 if len(item.sources) > 1 else 0.0
        # Small weight: this is a tie-breaker on top of scores that already
        # reflect real relevance (IDF-weighted keyword match + TF-IDF vector
        # similarity upstream), not the primary signal.
        item.score = item.score + (0.8 * overlap) + multi_source_bonus

    ranked = sorted(evidence, key=lambda e: e.score, reverse=True)
    return ranked[:top_k]

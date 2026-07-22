"""
fusion.py
==========
The Evidence Fusion Layer: merges whatever the (up to three) retrieval
branches produced into a single deduplicated evidence list, tagging each
item with which branch(es) found it. A node found by more than one branch
is a genuine cross-signal of relevance, so its fused score is boosted
rather than just deduplicated away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.graph_store import RetrievedNode
from src.tools.mcp_adapter import ToolResult


@dataclass
class EvidenceItem:
    id: str
    text: str
    citation: str
    group: str
    score: float
    sources: list[str] = field(default_factory=list)
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "citation": self.citation,
            "group": self.group,
            "score": round(self.score, 3),
            "sources": self.sources,
            "url": self.url,
        }


def fuse_evidence(
    direct_kg_nodes: list[RetrievedNode],
    hybrid_nodes: list[RetrievedNode],
    mcp_results: list[ToolResult],
) -> list[EvidenceItem]:
    fused: dict[str, EvidenceItem] = {}

    def add_kg(nodes: list[RetrievedNode], source: str, weight: float):
        for n in nodes:
            if n.id in fused:
                item = fused[n.id]
                item.score += n.score * weight
                if source not in item.sources:
                    item.sources.append(source)
            else:
                fused[n.id] = EvidenceItem(
                    id=n.id,
                    text=n.title or n.label,
                    citation=n.citation(),
                    group=n.group,
                    score=n.score * weight,
                    sources=[source],
                )

    add_kg(direct_kg_nodes, "direct_kg", weight=1.0)
    add_kg(hybrid_nodes, "hybrid_graphrag", weight=0.6)

    for i, r in enumerate(mcp_results):
        key = f"external:{i}:{r.url or r.title}"
        fused[key] = EvidenceItem(
            id=key,
            text=f"{r.title} — {r.snippet}",
            citation=r.title or r.source,
            group="external",
            score=3.0 - (i * 0.3),  # external results are ranked but weighted below KG evidence
            sources=[r.source],
            url=r.url,
        )

    return sorted(fused.values(), key=lambda e: e.score, reverse=True)


def format_context(items: list[EvidenceItem], max_chars: int = 6000) -> str:
    """Render reranked evidence into a citation-friendly context block for the LLM."""
    blocks, used = [], 0
    for item in items:
        text = item.text.strip() or "(no body text — structural node)"
        block = f"[{item.citation}]\n{text}\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n---\n".join(blocks)


def citations_from(items: list[EvidenceItem]) -> list[str]:
    seen, ordered = set(), []
    for item in items:
        if item.citation not in seen:
            seen.add(item.citation)
            ordered.append(item.citation)
    return ordered

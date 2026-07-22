"""
graph_store.py
================
Loads the Constitution knowledge graph (nodes = Parts / Articles / Clauses /
Subclauses, edges = contains / has_clause / has_subclause) into a NetworkX
directed graph, and exposes retrieval primitives used by the agent workflow.

This is the "retrieval" half of a Graph-RAG system: instead of embedding
chunks of text into a vector store, we keep the document's native legal
hierarchy as a graph and retrieve by traversing it. This preserves structure
(e.g. "which clauses belong to Article 19") that flat vector search throws
away.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import networkx as nx

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "constitution_graph.json"


@dataclass
class RetrievedNode:
    """A single graph node returned by a search, with a relevance score."""

    id: str
    label: str
    title: str
    group: str
    score: float = 0.0

    def citation(self) -> str:
        """Human-readable citation string, e.g. 'Article 19' or 'Clause 2 of Article 19'."""
        return self.label.split(":", 1)[-1].strip() if ":" in self.label else self.label

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "title": self.title,
            "group": self.group,
            "score": round(self.score, 3),
        }


class ConstitutionGraphStore:
    """Loads and queries the constitutional knowledge graph."""

    def __init__(self, data_path: str | Path = DEFAULT_DATA_PATH):
        self.data_path = Path(data_path)
        self.graph: nx.DiGraph = nx.DiGraph()
        self._load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        with open(self.data_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for n in payload["nodes"]:
            self.graph.add_node(
                n["id"],
                label=n.get("label", n["id"]),
                title=n.get("title", ""),
                group=n.get("group", "article"),
            )
        for e in payload["edges"]:
            self.graph.add_edge(e["from"], e["to"], relation=e.get("group") or e.get("title", ""))

        self._build_term_stats()

    def _build_term_stats(self) -> None:
        """
        Precompute document frequency per term across every node's label+body
        text, so `search()` can weight matches by rarity (IDF-style) instead
        of treating every query word equally. This is what stops generic
        words like "government" or "conditions" (which appear in dozens of
        articles) from outranking the actually distinctive words in a query
        like "freedom of speech" — the same problem TF-IDF solves for the
        vector index, applied here to the plain keyword-search branch too.
        """
        doc_freq: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            text = f"{data['label']} {data.get('title', '')}".lower()
            terms = {t for t in re.split(r"\W+", text) if len(t) > 2}
            for t in terms:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        self._doc_freq = doc_freq
        self._total_docs = max(self.graph.number_of_nodes(), 1)

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        return math.log((1 + self._total_docs) / (1 + df)) + 1.0

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 6, groups: Iterable[str] | None = None) -> list[RetrievedNode]:
        """
        Keyword + fuzzy search over node labels and body text.

        This is intentionally simple (no embeddings, no external index) —
        the graph is small enough that a scored substring/fuzzy match over
        every node is fast and fully transparent, which matters for a
        legal-citation use case where you want to be able to explain *why*
        a passage was retrieved.

        Scoring, in priority order:
          1. Explicit "Article N" mention -> that article (and its clauses)
             dominate the ranking. This is what the Direct-KG-lookup branch
             of the retrieval pipeline relies on.
          2. Query-term overlap with the node's label/body, normalized by
             query length so long questions don't get diluted.
          3. Fuzzy string similarity as a small tie-breaker only — it used
             to be weighted heavily enough to outrank an exact article
             match, which was a real bug (Article 21 losing to Article 33
             on a question that named Article 21 explicitly).
        """
        query_norm = query.lower().strip()
        query_terms = [t for t in re.split(r"\W+", query_norm) if len(t) > 2]
        article_num_match = re.search(r"\barticle\s*(\d+[a-zA-Z]?)\b", query_norm)
        article_prefix = f"article:{article_num_match.group(1)}".lower() if article_num_match else None

        results: list[RetrievedNode] = []
        for node_id, data in self.graph.nodes(data=True):
            if groups and data["group"] not in groups:
                continue

            label = data["label"].lower()
            title = (data.get("title") or "").lower()
            node_id_lower = node_id.lower()
            score = 0.0

            # (1) Strong, dominant signal: explicit article number match.
            if article_prefix:
                if node_id_lower == article_prefix:
                    score += 25.0
                elif node_id_lower.startswith(article_prefix + ":"):
                    score += 15.0  # a clause/subclause of the named article

            # (2) Term overlap, weighted by term rarity (IDF) so distinctive
            # words like "speech" matter far more than boilerplate words
            # like "government" or "conditions" that appear in dozens of
            # articles. Normalized against the query's own total IDF mass
            # so longer questions don't get an unfair boost.
            if query_terms:
                label_terms = set(re.split(r"\W+", label))
                title_terms = set(re.split(r"\W+", title))
                total_idf = sum(self._idf(t) for t in query_terms) or 1.0
                label_idf = sum(self._idf(t) for t in query_terms if t in label_terms)
                title_idf = sum(self._idf(t) for t in query_terms if t in title_terms)
                score += 6.0 * (label_idf / total_idf)
                score += 3.0 * (title_idf / total_idf)

            # (3) Fuzzy fallback — small weight, only breaks ties / catches
            # paraphrases that share no exact terms.
            if query_terms:
                score += 0.8 * SequenceMatcher(None, query_norm, label).ratio()
                if title:
                    score += 0.4 * SequenceMatcher(None, query_norm, title[: len(query_norm) + 40]).ratio()

            if score > 0.6:
                results.append(
                    RetrievedNode(
                        id=node_id,
                        label=data["label"],
                        title=data.get("title", ""),
                        group=data["group"],
                        score=score,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def extract_article_numbers(self, text: str) -> list[str]:
        """Pull explicit 'Article N' / 'Article 21A' mentions out of free text."""
        return [m.group(1).upper() for m in re.finditer(r"\barticle\s*(\d+[a-zA-Z]?)\b", text.lower())]

    def get_article_and_descendants(self, article_num: str) -> list[RetrievedNode]:
        """
        Direct-lookup retrieval path: given an article number like '19' or
        '21A', return that article node plus every clause/subclause beneath
        it. This is the "Direct KG (Article lookup)" branch of the
        retrieval pipeline — used when the user names a specific article,
        which needs no fuzzy search at all.
        """
        target = f"article:{article_num}".lower()
        article_id = next((n for n in self.graph.nodes if n.lower() == target), None)
        if article_id is None:
            return []

        ids = [article_id] + list(nx.descendants(self.graph, article_id))
        nodes = []
        for i, node_id in enumerate(ids):
            data = self.graph.nodes[node_id]
            nodes.append(
                RetrievedNode(
                    id=node_id,
                    label=data["label"],
                    title=data.get("title", ""),
                    group=data["group"],
                    score=20.0 if i == 0 else 12.0,
                )
            )
        return nodes

    # ------------------------------------------------------------------ #
    # Traversal / context expansion
    # ------------------------------------------------------------------ #
    def expand_context(
        self,
        node_ids: Iterable[str],
        include_children: bool = True,
        seed_scores: dict[str, float] | None = None,
    ) -> list[RetrievedNode]:
        """
        Given a set of seed node ids, pull in their structural neighbourhood:
        parent Article (for a Clause/Subclause) and, optionally, immediate
        child clauses (for an Article). This mirrors how a lawyer reads a
        provision — never in isolation from its parent article.

        `seed_scores`, if given, maps node_id -> its real relevance score
        from search/vector ranking, and that score (not a flat constant) is
        used as the base for the seed and its neighbourhood. This matters:
        an earlier version always set every seed to a flat 10.0 regardless
        of how relevant it actually was, which let a barely-relevant seed's
        expanded neighbourhood drown out a genuinely on-point match found
        by a different retrieval branch. Falls back to a flat 10.0 base for
        callers that just want "expand these ids" without ranking context.
        """
        seed_scores = seed_scores or {}
        expanded: dict[str, RetrievedNode] = {}

        for node_id in node_ids:
            if node_id not in self.graph:
                continue
            base = seed_scores.get(node_id, 10.0)
            self._add_node(expanded, node_id, score=base)

            # Walk up to parent(s), scored as a fraction of the seed's own relevance
            for parent_id in self.graph.predecessors(node_id):
                self._add_node(expanded, parent_id, score=base * 0.5)
                for grandparent_id in self.graph.predecessors(parent_id):
                    self._add_node(expanded, grandparent_id, score=base * 0.15)

            # Pull in immediate children (clauses of an article) for fuller context
            if include_children:
                for child_id in self.graph.successors(node_id):
                    self._add_node(expanded, child_id, score=base * 0.3)

        return sorted(expanded.values(), key=lambda r: r.score, reverse=True)

    def _add_node(self, bucket: dict[str, RetrievedNode], node_id: str, score: float) -> None:
        if node_id in bucket:
            bucket[node_id].score = max(bucket[node_id].score, score)
            return
        data = self.graph.nodes[node_id]
        bucket[node_id] = RetrievedNode(
            id=node_id,
            label=data["label"],
            title=data.get("title", ""),
            group=data["group"],
            score=score,
        )

    def get_node(self, node_id: str) -> RetrievedNode | None:
        if node_id not in self.graph:
            return None
        data = self.graph.nodes[node_id]
        return RetrievedNode(id=node_id, label=data["label"], title=data.get("title", ""), group=data["group"])

    # ------------------------------------------------------------------ #
    # Formatting helpers used by agent prompts
    # ------------------------------------------------------------------ #
    @staticmethod
    def format_context(nodes: list[RetrievedNode], max_chars: int = 6000) -> str:
        """Render retrieved nodes into a citation-friendly context block for the LLM."""
        blocks = []
        used = 0
        for n in nodes:
            text = n.title.strip() or "(no body text — structural node)"
            block = f"[{n.citation()}]\n{text}\n"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n---\n".join(blocks)

    @staticmethod
    def citations(nodes: list[RetrievedNode]) -> list[str]:
        seen, ordered = set(), []
        for n in nodes:
            c = n.citation()
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

"""
vector_index.py
=================
A local TF-IDF vector index over every node's label+body text, used as the
"Vector DB" half of the Hybrid GraphRAG retrieval branch.

Honest scope note: this is a lightweight, dependency-free (no external
service, no API key) stand-in for a real hosted vector database with dense
embeddings. It captures lexical/semantic overlap that keyword search alone
misses (different wording, synonyms sklearn's tokenizer can still relate),
but it is NOT a substitute for real sentence embeddings. Swapping this
class out for, say, a `sentence-transformers` encoder + FAISS/Chroma is a
drop-in extension — see the docstring on `VectorIndex.query`.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.graph_store import ConstitutionGraphStore, RetrievedNode


@dataclass
class VectorHit:
    node: RetrievedNode
    similarity: float


class VectorIndex:
    """Builds once from a ConstitutionGraphStore and answers similarity queries."""

    def __init__(self, store: ConstitutionGraphStore):
        self.store = store
        self._ids: list[str] = []
        self._texts: list[str] = []
        for node_id, data in store.graph.nodes(data=True):
            text = f"{data['label']} {data.get('title', '')}".strip()
            self._ids.append(node_id)
            self._texts.append(text)

        # sublinear_tf + stop-word removal keeps common legal boilerplate
        # ("shall", "any person") from swamping the more distinctive terms.
        self._vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def query(self, text: str, top_k: int = 8, min_similarity: float = 0.05) -> list[VectorHit]:
        """
        Return the top_k nodes by cosine similarity to `text`.

        To upgrade this to real dense embeddings later: replace
        `self._vectorizer.transform([text])` with a call to an embedding
        model, and `self._matrix` with a precomputed embedding matrix
        (e.g. stored in FAISS/Chroma) — `cosine_similarity` and the return
        shape stay the same.
        """
        if not text.strip():
            return []
        query_vec = self._vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self._matrix).ravel()
        ranked_idx = sims.argsort()[::-1][:top_k]

        hits = []
        for idx in ranked_idx:
            sim = float(sims[idx])
            if sim < min_similarity:
                continue
            node_id = self._ids[idx]
            data = self.store.graph.nodes[node_id]
            node = RetrievedNode(
                id=node_id,
                label=data["label"],
                title=data.get("title", ""),
                group=data["group"],
                score=sim * 10.0,  # scale onto roughly the same range as graph_store scores
            )
            hits.append(VectorHit(node=node, similarity=sim))
        return hits

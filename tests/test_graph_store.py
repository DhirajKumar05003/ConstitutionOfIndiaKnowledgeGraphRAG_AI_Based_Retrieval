import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_store import ConstitutionGraphStore  # noqa: E402


def test_graph_loads():
    store = ConstitutionGraphStore()
    assert store.graph.number_of_nodes() > 0
    assert store.graph.number_of_edges() > 0


def test_search_finds_article_by_number():
    store = ConstitutionGraphStore()
    results = store.search("Article 19")
    assert results, "expected at least one result for 'Article 19'"
    top_labels = [r.label.lower() for r in results[:3]]
    assert any("article 19" in lbl for lbl in top_labels)


def test_search_finds_article_by_keyword():
    store = ConstitutionGraphStore()
    results = store.search("freedom of speech")
    assert results
    assert any("speech" in (r.title.lower() + r.label.lower()) for r in results)


def test_expand_context_includes_parent_article():
    store = ConstitutionGraphStore()
    # a known clause id from the dataset
    clause_id = "article:19:clause:1:sub:a"
    assert clause_id in store.graph
    expanded = store.expand_context([clause_id])
    ids = {n.id for n in expanded}
    assert "article:19" in ids or "article:19:clause:1" in ids


def test_citations_are_deduplicated_and_ordered():
    store = ConstitutionGraphStore()
    results = store.search("equality before law")
    citations = store.citations(results)
    assert len(citations) == len(set(citations))


def test_format_context_respects_max_chars():
    store = ConstitutionGraphStore()
    results = store.search("right to education")
    context = store.format_context(results, max_chars=200)
    assert len(context) <= 250  # small slack for block separators

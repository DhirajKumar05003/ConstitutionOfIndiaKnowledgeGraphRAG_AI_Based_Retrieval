import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_store import ConstitutionGraphStore, RetrievedNode  # noqa: E402
from src.retrieval.fusion import fuse_evidence, citations_from, format_context  # noqa: E402
from src.retrieval.planner import plan_retrieval  # noqa: E402
from src.retrieval.reranker import rerank  # noqa: E402
from src.tools.mcp_adapter import NoOpToolAdapter, ToolResult  # noqa: E402
from src.vector_index import VectorIndex  # noqa: E402


def test_planner_triggers_direct_kg_for_named_article():
    plan = plan_retrieval({"entities": {"article_numbers": ["19"], "case_names": []}, "confidence": 0.9, "intent": "qa"})
    assert plan.run_direct_kg is True
    assert plan.article_numbers == ["19"]


def test_planner_triggers_mcp_tools_for_case_law_intent():
    plan = plan_retrieval({"entities": {"article_numbers": [], "case_names": ["Kesavananda Bharati"]}, "confidence": 0.7, "intent": "case_law"})
    assert plan.run_mcp_tools is True


def test_planner_falls_back_to_hybrid_when_nothing_else_fires():
    plan = plan_retrieval({"entities": {"article_numbers": [], "case_names": []}, "confidence": 0.6, "intent": "qa"})
    assert plan.run_hybrid_graphrag is True


def test_noop_tool_adapter_returns_nothing():
    adapter = NoOpToolAdapter()
    assert adapter.is_available() is True
    assert adapter.search("anything") == []


def test_fusion_boosts_nodes_found_by_multiple_branches():
    n = RetrievedNode(id="article:19", label="Article 19", title="...", group="article", score=5.0)
    fused = fuse_evidence(direct_kg_nodes=[n], hybrid_nodes=[n], mcp_results=[])
    assert len(fused) == 1
    assert len(fused[0].sources) == 2


def test_fusion_includes_external_results():
    mcp = [ToolResult(title="News piece", snippet="...", url="https://example.com")]
    fused = fuse_evidence(direct_kg_nodes=[], hybrid_nodes=[], mcp_results=mcp)
    assert len(fused) == 1
    assert fused[0].group == "external"


def test_reranker_boosts_term_overlap():
    from src.retrieval.fusion import EvidenceItem

    a = EvidenceItem(id="a", text="freedom of speech and expression", citation="Article 19", group="article", score=1.0)
    b = EvidenceItem(id="b", text="unrelated procedural clause", citation="Article 99", group="article", score=1.0)
    ranked = rerank([a, b], "freedom of speech question")
    assert ranked[0].id == "a"


def test_format_context_and_citations_from_reranked_evidence():
    from src.retrieval.fusion import EvidenceItem

    items = [EvidenceItem(id="a", text="body text", citation="Article 19", group="article", score=5.0)]
    ctx = format_context(items)
    assert "[Article 19]" in ctx
    assert citations_from(items) == ["Article 19"]


def test_vector_index_finds_paraphrased_double_jeopardy_clause():
    store = ConstitutionGraphStore()
    vi = VectorIndex(store)
    hits = vi.query("punished twice for the same offence")
    assert any("20" in h.node.id for h in hits)

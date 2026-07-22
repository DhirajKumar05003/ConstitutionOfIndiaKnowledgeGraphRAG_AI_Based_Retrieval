import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.graph_workflow import build_workflow  # noqa: E402
from src.graph_store import ConstitutionGraphStore  # noqa: E402
from src.llm_client import LLMResponse  # noqa: E402


class FakeLLM:
    """A stand-in for GroqLLMClient that returns deterministic canned text/JSON,
    so the whole pipeline can be tested without hitting the network."""

    def __init__(self):
        self.calls = []

    def chat(self, system_prompt, user_prompt, temperature=0.2, max_tokens=500, json_mode=False):
        self.calls.append((system_prompt[:30], user_prompt[:30]))
        if "Expansive" in system_prompt:
            return LLMResponse(text="Expansive-side argument. [Article 19]", model="fake")
        if "Restrictive" in system_prompt:
            return LLMResponse(text="Restrictive-side argument. [Article 19]", model="fake")
        if "moderator" in system_prompt.lower():
            return LLMResponse(text="Both sides agree on X, disagree on Y.", model="fake")
        return LLMResponse(text="Plain-language answer. [Article 19]", model="fake")

    def chat_json(self, system_prompt, user_prompt, default, temperature=0.0, max_tokens=400):
        self.calls.append(("json:" + system_prompt[:20], user_prompt[:30]))
        if "Query Analyzer" in system_prompt:
            return {
                "intent": "qa",
                "complexity": "simple",
                "constitutional_concepts": ["freedom of speech"],
                "case_names": [],
                "confidence": 0.8,
            }
        if "Faithfulness" in system_prompt:
            return {"faithful": True, "issues": [], "unsupported_citations": []}
        return dict(default)


def test_qa_mode_returns_final_answer_with_citation():
    store = ConstitutionGraphStore()
    llm = FakeLLM()
    app = build_workflow(llm, store)
    result = app.invoke({"question": "What does Article 19 protect?", "mode": "qa"})
    assert result["final_answer"]
    assert result["citations"]
    assert result["entities"]["article_numbers"] == ["19"]
    assert result["retrieval_plan"]["run_direct_kg"] is True


def test_research_mode_produces_research_answer():
    store = ConstitutionGraphStore()
    llm = FakeLLM()
    app = build_workflow(llm, store)
    result = app.invoke({"question": "Explain the scope of Article 21", "mode": "research"})
    assert result["final_answer"]
    assert result["reranked_evidence"]


def test_debate_mode_runs_all_three_debate_agents():
    store = ConstitutionGraphStore()
    llm = FakeLLM()
    app = build_workflow(llm, store)
    result = app.invoke({"question": "Should free speech ever be restricted?", "mode": "debate"})
    assert result.get("debate_for")
    assert result.get("debate_against")
    assert result.get("debate_verdict")
    assert "expansive" in result["final_answer"].lower()


def test_direct_kg_branch_fires_for_named_article():
    store = ConstitutionGraphStore()
    llm = FakeLLM()
    app = build_workflow(llm, store)
    result = app.invoke({"question": "What does Article 21 say?", "mode": "qa"})
    assert any(n["id"] == "article:21" for n in result["direct_kg_nodes"])


def test_fusion_deduplicates_nodes_found_by_multiple_branches():
    store = ConstitutionGraphStore()
    llm = FakeLLM()
    app = build_workflow(llm, store)
    result = app.invoke({"question": "What does Article 19 say about freedom of speech?", "mode": "qa"})
    ids = [e["id"] for e in result["fused_evidence"]]
    assert len(ids) == len(set(ids))


def test_no_context_short_circuits_qa_gracefully():
    store = ConstitutionGraphStore()
    llm = FakeLLM()
    app = build_workflow(llm, store)
    result = app.invoke({"question": "asdkjhasldkjhasdlkjh nonsense query zzz", "mode": "qa"})
    assert "final_answer" in result


def test_verification_failure_appends_warning_to_final_answer():
    store = ConstitutionGraphStore()

    class FlaggingLLM(FakeLLM):
        def chat_json(self, system_prompt, user_prompt, default, temperature=0.0, max_tokens=400):
            if "Faithfulness" in system_prompt:
                return {"faithful": False, "issues": ["cites Article 99 which does not exist"], "unsupported_citations": ["Article 99"]}
            return super().chat_json(system_prompt, user_prompt, default, temperature, max_tokens)

    app = build_workflow(FlaggingLLM(), store)
    result = app.invoke({"question": "What does Article 19 protect?", "mode": "qa"})
    assert "Faithfulness check flagged" in result["final_answer"]


def test_pronoun_resolution_uses_prior_question():
    store = ConstitutionGraphStore()
    llm = FakeLLM()
    app = build_workflow(llm, store)
    result = app.invoke(
        {"question": "What about its exceptions?", "mode": "qa", "prior_question": "What does Article 19 protect?"}
    )
    assert "Article 19" in result["resolved_question"]

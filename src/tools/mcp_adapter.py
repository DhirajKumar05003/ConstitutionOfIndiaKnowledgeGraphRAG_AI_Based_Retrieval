"""
mcp_adapter.py
================
The "MCP Tools (Web/Search/API)" branch of the retrieval planner.

Honest scope note: this repo does not ship a hardwired web-search
integration, because that would mean silently depending on a third-party
search API with its own key/quota/ToS that this project has no way to
guarantee for you. Instead this module defines a small adapter interface —
`ToolAdapter.search(query) -> list[ToolResult]` — with:

  - `NoOpToolAdapter`, the default: returns nothing, but tells the pipeline
    *why* (so the UI can say "external lookup skipped: no tool configured"
    instead of silently pretending it checked the web).
  - `TavilyToolAdapter`, an optional example implementation that activates
    automatically if `TAVILY_API_KEY` is set and the `tavily-python`
    package is installed. Swap in Serper, Bing, SerpAPI, or a real MCP
    server the same way: implement `search()`, register it in
    `get_default_adapter()`.

This branch only fires when the Query Analyzer flags the question as
outside the knowledge graph's scope (see `retrieval_planner.py`) — for
in-scope constitutional questions it never runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ToolResult:
    title: str
    snippet: str
    url: str = ""
    source: str = "external"


class ToolAdapter:
    """Interface every external-tool adapter must implement."""

    name: str = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def search(self, query: str, max_results: int = 3) -> list[ToolResult]:
        raise NotImplementedError


class NoOpToolAdapter(ToolAdapter):
    """Default adapter: always available, always returns nothing."""

    name = "none (no external tool configured)"

    def is_available(self) -> bool:
        return True

    def search(self, query: str, max_results: int = 3) -> list[ToolResult]:
        return []


class TavilyToolAdapter(ToolAdapter):
    """Optional example web-search adapter using Tavily's search API."""

    name = "tavily"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self._client = None
        if self.api_key:
            try:
                from tavily import TavilyClient  # optional dependency

                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                self._client = None

    def is_available(self) -> bool:
        return self._client is not None

    def search(self, query: str, max_results: int = 3) -> list[ToolResult]:
        if not self._client:
            return []
        response = self._client.search(query=query, max_results=max_results)
        results = []
        for item in response.get("results", []):
            results.append(
                ToolResult(
                    title=item.get("title", ""),
                    snippet=item.get("content", "")[:600],
                    url=item.get("url", ""),
                    source="tavily",
                )
            )
        return results


def get_default_adapter() -> ToolAdapter:
    """Auto-selects Tavily if configured, otherwise the safe no-op adapter."""
    tavily = TavilyToolAdapter()
    if tavily.is_available():
        return tavily
    return NoOpToolAdapter()

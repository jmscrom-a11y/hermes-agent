"""Web search tool for Hermes V4.

Uses DuckDuckGo (no API key required) — the same backend v3 already
depends on for web search.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hermes_v4.core.base import Tool, ToolResult

DEFAULT_MAX_RESULTS = 5


class WebSearchTool(Tool):
    """Searches the live web for information not in the local document index."""

    name = "web_search"
    description = (
        "실시간 웹 검색이 필요한 질문(최신 뉴스, 시세, 외부 정보 등)에 사용합니다. "
        "저장된 문서에 없는 정보를 찾을 때 사용하세요."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    }
    output_schema = {
        "type": "object",
        "properties": {"results": {"type": "array"}},
    }

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        query = input.get("query")
        if not query or not str(query).strip():
            errors.append("'query' is required and must be a non-empty string")
        return errors

    def _search_sync(self, query: str, max_results: int) -> list[dict]:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=max_results)
            if not results:
                results = ddgs.text(query, max_results=max_results)
        return results or []

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))

        query = str(input["query"]).strip()
        max_results = int(input.get("max_results") or DEFAULT_MAX_RESULTS)

        start = time.monotonic()
        try:
            results = await asyncio.to_thread(self._search_sync, query, max_results)
        except Exception as exc:
            return ToolResult.fail(f"Web search failed: {exc}")

        if not results:
            return ToolResult(success=True, output="검색 결과가 없습니다.", metadata={"num_results": 0})

        formatted = "\n\n".join(
            f"[{i}] {r.get('title', 'untitled')}\n{r.get('body') or r.get('excerpt', '')}\n{r.get('url') or r.get('href', '')}"
            for i, r in enumerate(results, start=1)
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return ToolResult(
            success=True,
            output=formatted,
            metadata={"num_results": len(results)},
            duration_ms=duration_ms,
        )

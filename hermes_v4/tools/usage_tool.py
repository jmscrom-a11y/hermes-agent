"""Claude Code usage/cost reporting tool for Hermes V4.

Reads back what ClaudeCodeTool has been logging via hermes_v4.usage since
every claude -p call (e.g. PPTX design) reports a real cost that otherwise
just disappears — this is the read side of that log.
"""

from __future__ import annotations

from typing import Any

from hermes_v4.core.base import Tool, ToolResult
from hermes_v4.usage import summarize_usage


class UsageTool(Tool):
    """Reports accumulated Claude Code usage/cost."""

    name = "usage_report"
    description = (
        "Claude Code(AI 디자인/코드 작업) 사용량이나 비용을 물어볼 때 사용하세요 — "
        "예: '이번 달 클로드코드 얼마 썼어?', 'AI 사용량 알려줘'."
    )
    input_schema = {"type": "object", "properties": {}}
    output_schema = {
        "type": "object",
        "properties": {
            "all_time_cost_usd": {"type": "number"},
            "all_time_calls": {"type": "integer"},
            "month_cost_usd": {"type": "number"},
            "month_calls": {"type": "integer"},
        },
    }

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        return []

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        stats = summarize_usage()
        summary = (
            f"이번 달 Claude Code 사용: ${stats['month_cost_usd']:.2f} ({stats['month_calls']}회 호출)\n"
            f"전체 누적: ${stats['all_time_cost_usd']:.2f} ({stats['all_time_calls']}회 호출)"
        )
        return ToolResult.ok(output=summary, metadata=stats)

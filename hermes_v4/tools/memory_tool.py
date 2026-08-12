"""Long-term user memory tool for Hermes V4.

Lets the user explicitly ask to be remembered ("기억해줘"). Saved facts are
re-injected into every subsequent planning call (planner/strategy.py
format_facts) and the chitchat/fallback reply (telegram/bot.py) so answers
can draw on them automatically — there is no separate "recall" tool the
planner has to remember to call.

`user_id` is never supplied by the planner, same rationale as
ReminderTool's "chat_id": the Telegram layer injects it into the action
input after planning, before execution.
"""

from __future__ import annotations

from typing import Any

from hermes_v4.core.base import Tool, ToolResult
from hermes_v4.memory.store import SqliteMemoryStore


class RememberTool(Tool):
    """Saves a fact/preference about the user for future conversations."""

    name = "remember"
    description = (
        "사용자가 '기억해줘', '앞으로 ~라고 불러줘', '내 ~는 ~야 기억해' 처럼 "
        "특정 정보를 앞으로도 계속 기억해달라고 명시적으로 요청했을 때만 "
        "사용하세요. 단순한 질문에 답하거나 지금 이 요청에만 필요한 정보에는 "
        "사용하지 마세요."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "기억할 사실을 한 문장으로 (예: '생일은 3월 5일이다', '고수 못 먹음')",
            },
        },
        "required": ["content"],
    }
    output_schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
    }

    def __init__(self, memory: SqliteMemoryStore) -> None:
        self.memory = memory

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        if not str(input.get("content", "")).strip():
            errors.append("'content' is required and must be a non-empty string")
        if not input.get("user_id"):
            errors.append("'user_id' missing (internal error — the caller must inject it)")
        return errors

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))

        content = str(input["content"]).strip()
        await self.memory.save_fact(str(input["user_id"]), content)
        return ToolResult.ok(output=f"기억했습니다: {content}")

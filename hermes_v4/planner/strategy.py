"""Planning strategies for Hermes V4.

Strategies define how the planner decomposes user requests into plans.
The planner NEVER hardcodes tool routing — it always uses the LLM
to decide which tools to use based on the request and available tools.
"""

from __future__ import annotations

import abc
import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import ToolInfo
from hermes_v4.planner.plan import Action, Plan, Step


def format_tool_list(tools: list[ToolInfo]) -> str:
    """Render tools with their input_schema so the LLM uses the right
    parameter names in each action's "input" — without this, a small
    model will guess plausible-but-wrong keys (e.g. "query" instead of
    "question") and every action silently fails validation.
    """
    lines = []
    for t in tools:
        schema = json.dumps(t.input_schema, ensure_ascii=False) if t.input_schema else "{}"
        lines.append(f"- {t.name}: {t.description}\n  input schema: {schema}")
    return "\n".join(lines)


def _parse_steps(steps_data: list[dict[str, Any]]) -> list[Step]:
    """Parse steps, assigning deterministic "step_N" ids instead of
    Step's default random UUID.

    The LLM has no way to know a step's real (randomly generated) id when
    writing "depends_on" — it reliably invents readable placeholders like
    "step_1" instead. Randomly-generated ids never match those, so
    WorkflowGraph's dependency check (`d in self.nodes`) silently treats
    the bogus reference as satisfied and the intended ordering is lost.
    Assigning "step_N" ids in parse order — the same convention the LLM
    already gravitates toward — makes "depends_on" actually work.
    """
    steps = []
    for i, step_data in enumerate(steps_data, start=1):
        actions = [
            Action(
                tool_name=a.get("tool_name", ""),
                input=a.get("input", {}),
                output_key=a.get("output_key", "output"),
            )
            for a in step_data.get("actions", [])
        ]
        steps.append(
            Step(
                id=step_data.get("id") or f"step_{i}",
                name=step_data.get("name", ""),
                description=step_data.get("description", ""),
                actions=actions,
                depends_on=step_data.get("depends_on", []),
            )
        )
    return steps


_WEEKDAYS_KR = ["월", "화", "수", "목", "금", "토", "일"]


def format_current_datetime() -> str:
    """Render "today" for the planner, so a relative request like "내일 3시에
    ~" or "이번 주 금요일에 ~" can be turned into an absolute ISO date for
    schedule_reminder's "date" field — without this the LLM has no way to
    know what day it currently is.
    """
    settings = get_settings()
    now = datetime.now(ZoneInfo(settings.REMINDER_TIMEZONE))
    weekday = _WEEKDAYS_KR[now.weekday()]
    return f"\n\n현재 날짜/시각: {now.strftime('%Y-%m-%d %H:%M')} ({weekday}요일)"


def format_facts(context: dict[str, Any]) -> str:
    """Render facts the user previously asked to be remembered (see the
    "remember" tool) so answers can draw on them without the planner ever
    calling a "recall" tool — absent by default, only present when the
    caller (e.g. the Telegram handler) populated context["user_facts"].
    """
    facts = context.get("user_facts") or []
    if not facts:
        return ""
    lines = [f"- {f.get('content', '')}" for f in facts]
    return "\n\n사용자에 대해 기억하고 있는 정보 (참고만 하고 그 자체를 답변하지 마세요):\n" + "\n".join(lines)


def format_history(context: dict[str, Any]) -> str:
    """Render recent conversation turns (if any) for the LLM, so follow-up
    requests like "그거 좀 더 설명해줘" can be understood without the user
    repeating themselves. Absent by default — only present when the caller
    (e.g. the Telegram handler) populated context["previous_context"].
    """
    history = context.get("previous_context") or []
    if not history:
        return ""
    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history]
    return "\n\nRecent conversation (for context; only reference it, don't re-answer it):\n" + "\n".join(lines)


class PlanningStrategy(abc.ABC):
    """Base class for planning strategies.

    Each strategy defines how to format the LLM prompt for plan
    generation and how to parse the LLM response into a Plan.
    """

    @abc.abstractmethod
    def build_prompt(
        self,
        request: str,
        tools: list[ToolInfo],
        context: dict[str, Any],
    ) -> str:
        """Build the LLM prompt for plan generation.

        Args:
            request: User's original request.
            tools: Available tools from the registry.
            context: Additional context (memory, previous plans, etc.).

        Returns:
            Formatted prompt string for the LLM.
        """
        ...

    @abc.abstractmethod
    def parse_response(self, response: str) -> Plan:
        """Parse LLM response into a Plan.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed Plan object.
        """
        ...


class DefaultPlanningStrategy(PlanningStrategy):
    """Default strategy: sequential plan with LLM-decided tools.

    The planner sends the request + available tools to the LLM.
    The LLM decides which tools to use and in what order.
    """

    SYSTEM_PROMPT = """You are the planning engine of Hermes V4, an autonomous AI operating system.

Your job is to convert user requests into structured execution plans.

RULES:
1. NEVER hardcode tool routing — always let the plan specify which tools to use.
2. Each step must declare which tool to invoke, using ONLY the tools listed below.
3. Steps can depend on other steps (DAG execution).
4. Use the fewest steps that answer the request. Do not add a "report" or
   "summarize results" step — there is no such tool. The output of the last
   step is shown to the user automatically.
5. Every action's "input" must contain exactly the parameters that tool's
   description requires — never leave a required parameter empty.
6. Only use "generate_report" when the user explicitly asks for a file —
   a report, PPT/slides, or PDF. A bare topic, a question about a topic, or
   a request like "recipe 좀 알려줘"/"보내줘" is NOT a report request — use
   "rag" or answer directly instead. When in doubt, do not call
   "generate_report".
7. Use "schedule_reminder" both for recurring alerts at a fixed time (e.g.
   "매일 아침 8시에 ~") and one-time future reminders (e.g. "내일 3시에 ~",
   "이번 주 금요일에 ~"). For a recurring reminder leave "date" empty; for a
   one-time reminder set "date" to an ISO date (YYYY-MM-DD) computed from
   the current date/time given below. Never include "chat_id" in its
   "input" — that is injected automatically after planning.
8. Only use "remember" when the user explicitly asks to be remembered
   (e.g. "기억해줘", "앞으로 ~라고 불러줘"), not for information only
   relevant to the current request. Never include "user_id" in its
   "input" — that is injected automatically after planning.
9. Use "list_reminders" when the user asks what reminders are scheduled,
   "cancel_reminder" when they ask to cancel/remove one, and "edit_reminder"
   when they ask to change the time/date/content of one they already set
   (put a topic or time in "query" to identify which one; for "edit_reminder"
   only fill "new_time"/"new_date"/"new_prompt" for the fields actually
   changing). Never include "chat_id" in any of these — that is injected
   automatically after planning.

Available tools:
{tool_list}

Respond with a JSON plan:
{{
  "steps": [
    {{
      "id": "step_1",
      "name": "step name",
      "description": "what this step does",
      "actions": [
        {{
          "tool_name": "tool_name",
          "input": {{"param": "value"}},
          "output_key": "result"
        }}
      ],
      "depends_on": []
    }},
    {{
      "id": "step_2",
      "name": "...",
      "description": "...",
      "actions": [{{"tool_name": "...", "input": {{}}, "output_key": "..."}}],
      "depends_on": ["step_1"]
    }}
  ]
}}"""

    def build_prompt(
        self,
        request: str,
        tools: list[ToolInfo],
        context: dict[str, Any],
    ) -> str:
        tool_list = format_tool_list(tools)
        prompt = self.SYSTEM_PROMPT.format(tool_list=tool_list)
        return (
            f"{prompt}{format_current_datetime()}{format_facts(context)}"
            f"{format_history(context)}\n\nUser request: {request}"
        )

    def parse_response(self, response: str) -> Plan:
        """Parse LLM JSON response into a Plan."""
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # No "telegram" or other reporting tool exists in the registry —
            # return an empty plan so Planner._validate_plan rejects it and
            # the caller's own fallback (e.g. a direct chat reply) kicks in,
            # instead of failing later with a confusing "tool not found".
            return Plan(request="", steps=[])

        return Plan(request="", steps=_parse_steps(data.get("steps", [])))


class ParallelPlanningStrategy(PlanningStrategy):
    """Strategy: identify independent steps for parallel execution.

    The planner analyzes the request to find steps that don't
    depend on each other and marks them for parallel execution.
    """

    SYSTEM_PROMPT = """You are the planning engine of Hermes V4.

Identify independent steps that can run in parallel.

RULES:
1. Steps with no dependencies can run in parallel.
2. Each step must declare which tool to use.
3. Mark parallel steps with "parallel": true.

Available tools:
{tool_list}

Respond with a JSON plan:
{{
  "steps": [
    {{
      "id": "step_1",
      "name": "step name",
      "description": "what this step does",
      "actions": [
        {{
          "tool_name": "tool_name",
          "input": {{"param": "value"}},
          "output_key": "result"
        }}
      ],
      "depends_on": [],
      "parallel": false
    }}
  ]
}}"""

    def build_prompt(
        self,
        request: str,
        tools: list[ToolInfo],
        context: dict[str, Any],
    ) -> str:
        tool_list = format_tool_list(tools)
        prompt = self.SYSTEM_PROMPT.format(tool_list=tool_list)
        return (
            f"{prompt}{format_current_datetime()}{format_facts(context)}"
            f"{format_history(context)}\n\nUser request: {request}"
        )

    def parse_response(self, response: str) -> Plan:
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return Plan(request="", steps=[])

        return Plan(request="", steps=_parse_steps(data.get("steps", [])))


class IterativePlanningStrategy(PlanningStrategy):
    """Strategy: plan one step, execute, then plan the next based on results.

    Useful for complex tasks where the full plan depends on intermediate results.
    """

    SYSTEM_PROMPT = """You are the planning engine of Hermes V4.

Plan ONE step at a time. After each step executes, you'll receive the result
and plan the next step.

RULES:
1. Plan only one step at a time.
2. Each step must declare which tool to use.
3. Use results from previous steps in your input.

Available tools:
{tool_list}

Respond with a JSON step:
{{
  "name": "step name",
  "description": "what this step does",
  "actions": [
    {{
      "tool_name": "tool_name",
      "input": {{"param": "value"}},
      "output_key": "result"
    }}
  ],
  "depends_on": ["previous_step_id"]
}}"""

    def build_prompt(
        self,
        request: str,
        tools: list[ToolInfo],
        context: dict[str, Any],
    ) -> str:
        tool_list = format_tool_list(tools)
        previous_results = context.get("previous_results", [])
        results_text = ""
        if previous_results:
            results_text = "\n\nPrevious step results:\n" + "\n".join(
                f"- {r.get('step_name', 'unknown')}: {r.get('output', '')}"
                for r in previous_results[-5:]
            )
        prompt = self.SYSTEM_PROMPT.format(tool_list=tool_list)
        return (
            f"{prompt}{format_current_datetime()}{format_facts(context)}"
            f"{format_history(context)}\n\nUser request: {request}{results_text}"
        )

    def parse_response(self, response: str) -> Plan:
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return Plan(request="", steps=[])

        actions = [
            Action(
                tool_name=a.get("tool_name", ""),
                input=a.get("input", {}),
                output_key=a.get("output_key", "output"),
            )
            for a in data.get("actions", [])
        ]
        step = Step(
            name=data.get("name", ""),
            description=data.get("description", ""),
            actions=actions,
            depends_on=data.get("depends_on", []),
        )
        return Plan(request="", steps=[step])

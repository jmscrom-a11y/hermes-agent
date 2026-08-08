"""Planning strategies for Hermes V4.

Strategies define how the planner decomposes user requests into plans.
The planner NEVER hardcodes tool routing — it always uses the LLM
to decide which tools to use based on the request and available tools.
"""

from __future__ import annotations

import abc
from typing import Any

from hermes_v4.core.base import ToolInfo
from hermes_v4.planner.plan import Action, Plan, Step


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
2. Each step must declare which tool to invoke.
3. Steps can depend on other steps (DAG execution).
4. Include validation steps after write/modification operations.
5. Include a final reporting step.

Available tools:
{tool_list}

Respond with a JSON plan:
{{
  "steps": [
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
      "depends_on": []
    }}
  ]
}}"""

    def build_prompt(
        self,
        request: str,
        tools: list[ToolInfo],
        context: dict[str, Any],
    ) -> str:
        tool_list = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        prompt = self.SYSTEM_PROMPT.format(tool_list=tool_list)
        return f"{prompt}\n\nUser request: {request}"

    def parse_response(self, response: str) -> Plan:
        """Parse LLM JSON response into a Plan."""
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # If not valid JSON, wrap in a single step
            return Plan(
                request="",
                steps=[
                    Step(
                        name="respond",
                        description="Respond to user",
                        actions=[
                            Action(
                                tool_name="telegram",
                                input={"message": response},
                                output_key="response",
                            )
                        ],
                    )
                ],
            )

        steps = []
        for step_data in data.get("steps", []):
            actions = [
                Action(
                    tool_name=a.get("tool_name", ""),
                    input=a.get("input", {}),
                    output_key=a.get("output_key", "output"),
                )
                for a in step_data.get("actions", [])
            ]
            step = Step(
                name=step_data.get("name", ""),
                description=step_data.get("description", ""),
                actions=actions,
                depends_on=step_data.get("depends_on", []),
            )
            steps.append(step)

        return Plan(request="", steps=steps)


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
        tool_list = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        prompt = self.SYSTEM_PROMPT.format(tool_list=tool_list)
        return f"{prompt}\n\nUser request: {request}"

    def parse_response(self, response: str) -> Plan:
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return Plan(request="", steps=[])

        steps = []
        for step_data in data.get("steps", []):
            actions = [
                Action(
                    tool_name=a.get("tool_name", ""),
                    input=a.get("input", {}),
                    output_key=a.get("output_key", "output"),
                )
                for a in step_data.get("actions", [])
            ]
            step = Step(
                name=step_data.get("name", ""),
                description=step_data.get("description", ""),
                actions=actions,
                depends_on=step_data.get("depends_on", []),
            )
            steps.append(step)

        return Plan(request="", steps=steps)


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
        tool_list = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        previous_results = context.get("previous_results", [])
        results_text = ""
        if previous_results:
            results_text = "\n\nPrevious step results:\n" + "\n".join(
                f"- {r.get('step_name', 'unknown')}: {r.get('output', '')}"
                for r in previous_results[-5:]
            )
        prompt = self.SYSTEM_PROMPT.format(tool_list=tool_list)
        return f"{prompt}\n\nUser request: {request}{results_text}"

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

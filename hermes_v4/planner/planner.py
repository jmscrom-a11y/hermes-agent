"""Main Planner class for Hermes V4.

The planner is the brain of Hermes V4. It receives raw user input and
produces structured execution plans using an LLM.

Key principle: The planner NEVER hardcodes tool routing. It always:
1. Receives the user request + available tools
2. Uses the LLM to analyze intent
3. Generates a Plan with Steps
4. Each Step declares which tool to use
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any

from hermes_v4.core.base import ToolInfo, ToolRegistry
from hermes_v4.core.event import EventBus, PlanCreated, PlanFailed
from hermes_v4.core.context import ExecutionContext
from hermes_v4.llm.provider import LLMProvider
from hermes_v4.planner.plan import Plan, PlanStatus, Step
from hermes_v4.planner.strategy import (
    DefaultPlanningStrategy,
    PlanningStrategy,
)

logger = logging.getLogger(__name__)


class PlanGenerationError(Exception):
    """Raised when LLM fails to produce a valid plan."""


class PlanValidationError(Exception):
    """Raised when a generated plan fails validation."""


class Planner:
    """LLM-driven planner that converts user requests into executable plans.

    The planner NEVER hardcodes tool routing. It always uses the LLM
    to decide which tools to use based on the request and available tools.

    Example:
        >>> planner = Planner(llm_provider=ollama, tool_registry=registry)
        >>> plan = await planner.plan("Improve README", context)
        >>> # plan contains Steps that the executor will run
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        strategy: PlanningStrategy | None = None,
        max_steps: int = 20,
        context_window: int = 8000,
        event_bus: EventBus | None = None,
        cache_ttl_seconds: float = 300.0,
        cache_size: int = 128,
    ) -> None:
        self.llm = llm_provider
        self.tools = tool_registry
        self.strategy = strategy or DefaultPlanningStrategy()
        self.max_steps = max_steps
        self.context_window = context_window
        self.event_bus = event_bus or EventBus()
        # Cache of validated plans keyed on the exact (normalized) request
        # text, so a repeated identical request skips the planning LLM call
        # entirely. Only matches literal repeats, not paraphrases — but that's
        # the safe case: the cached plan was already validated for that exact
        # text. TTL bounds staleness (e.g. a stale plan outliving a RAG index
        # update); size bound is a simple LRU via OrderedDict.
        self._cache_ttl = cache_ttl_seconds
        self._cache_size = cache_size
        self._plan_cache: OrderedDict[str, tuple[float, Plan]] = OrderedDict()

    async def plan(self, request: str, context: ExecutionContext | None = None) -> Plan:
        """Generate an execution plan for the given request.

        This is the main entry point. It:
        1. Gathers available tools from the registry
        2. Gathers context (memory, previous plans, user state)
        3. Uses the strategy to format the LLM prompt
        4. Calls the LLM to generate a plan
        5. Validates the plan
        6. Publishes a PlanCreated event

        Args:
            request: User's original request.
            context: Execution context (optional).

        Returns:
            A Plan object with Steps.

        Raises:
            PlanGenerationError: If LLM fails to produce a valid plan.
            PlanValidationError: If the generated plan fails validation.
        """
        cache_key = request.strip().lower()
        cached_plan = self._get_cached_plan(cache_key)
        if cached_plan is not None:
            logger.info("Plan cache hit for request; skipping planning LLM call")
            self.event_bus.publish(PlanCreated(plan_id=cached_plan.id, request=request))
            return cached_plan

        # 1. Gather available tools
        available_tools = self.tools.get_available_tools()

        # 2. Gather context
        context_data = await self._gather_context(request, context or ExecutionContext(request_id=""))

        # 3. Build prompt using strategy
        prompt = self.strategy.build_prompt(request, available_tools, context_data)

        # 4. Call LLM to generate plan
        plan = await self._generate_plan(prompt)

        # 5. Validate the plan
        self._validate_plan(plan)

        # 6. Publish event
        self.event_bus.publish(PlanCreated(
            plan_id=plan.id,
            request=request,
        ))

        plan.status = PlanStatus.PENDING
        plan.request = request
        self._cache_plan(cache_key, plan)
        return plan

    def _get_cached_plan(self, key: str) -> Plan | None:
        entry = self._plan_cache.get(key)
        if entry is None:
            return None
        cached_at, plan = entry
        if time.monotonic() - cached_at > self._cache_ttl:
            del self._plan_cache[key]
            return None
        self._plan_cache.move_to_end(key)
        return self._clone_plan(plan)

    def _cache_plan(self, key: str, plan: Plan) -> None:
        self._plan_cache[key] = (time.monotonic(), copy.deepcopy(plan))
        self._plan_cache.move_to_end(key)
        while len(self._plan_cache) > self._cache_size:
            self._plan_cache.popitem(last=False)

    @staticmethod
    def _clone_plan(plan: Plan) -> Plan:
        """Deep-copy a cached plan into a fresh, unexecuted instance —
        reusing the cached Plan object directly would share mutable
        execution state (status, context, per-step _status) across every
        request that hits the cache.
        """
        clone = copy.deepcopy(plan)
        clone.id = uuid.uuid4().hex[:8]
        clone.status = PlanStatus.PENDING
        clone.completed_at = None
        clone.error = None
        clone.context = {}
        for step in clone.steps:
            if hasattr(step, "_status"):
                delattr(step, "_status")
        return clone

    async def _generate_plan(self, prompt: str) -> Plan:
        """Call the LLM to generate a plan from the prompt."""
        try:
            response = await self.llm.generate(
                messages=[
                    {"role": "system", "content": "You are a planning engine. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format="json",
            )
            plan = self.strategy.parse_response(response.content)
            return plan
        except Exception as exc:
            raise PlanGenerationError(f"Failed to generate plan: {exc}") from exc

    def _validate_plan(self, plan: Plan) -> None:
        """Validate a generated plan.

        Checks:
        - Plan has at least one step
        - No step exceeds max_steps
        - No circular dependencies
        - All tool names in steps exist in registry
        """
        if not plan.steps:
            raise PlanValidationError("Plan has no steps")

        if len(plan.steps) > self.max_steps:
            raise PlanValidationError(
                f"Plan has {len(plan.steps)} steps, exceeds max of {self.max_steps}"
            )

        # Check for circular dependencies
        self._check_circular_deps(plan)

        # Check all tool names exist
        for step in plan.steps:
            for action in step.actions:
                if action.tool_name and action.tool_name not in self.tools:
                    logger.warning("Tool '%s' not found in registry for step '%s'", action.tool_name, step.name)

    def _check_circular_deps(self, plan: Plan) -> None:
        """Check for circular dependencies in the plan's DAG."""
        step_ids = {step.id for step in plan.steps}
        step_map = {step.id: step for step in plan.steps}

        visited = set()
        in_stack = set()

        def dfs(step_id: str) -> bool:
            if step_id in in_stack:
                return True  # Cycle detected
            if step_id in visited:
                return False
            visited.add(step_id)
            in_stack.add(step_id)
            step = step_map.get(step_id)
            if step:
                for dep in step.depends_on:
                    if dfs(dep):
                        return True
            in_stack.discard(step_id)
            return False

        for step_id in step_ids:
            if dfs(step_id):
                raise PlanValidationError("Circular dependency detected in plan")

    async def _gather_context(
        self,
        request: str,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Gather context for plan generation.

        Includes:
        - Available tools list
        - Previous task history (if available)
        - Current workflow state (if any)
        """
        return {
            "available_tools": self.tools.list_tools(),
            "previous_context": context.metadata.get("previous_context", []),
            "user_facts": context.metadata.get("user_facts", []),
            "request": request,
        }

"""Step executor for Hermes V4.

Runs a single Step's actions against tools resolved from the registry,
with retry, timeout, and result validation. The WorkflowEngine uses
this for each DAG node; it has no notion of dependencies itself.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hermes_v4.core.base import Tool, ToolRegistry, ToolResult
from hermes_v4.core.context import ExecutionContext
from hermes_v4.core.event import EventBus, ToolExecuted
from hermes_v4.executor.retry import RetryHandler
from hermes_v4.executor.validator import ResultValidator
from hermes_v4.planner.plan import Step


class Executor:
    """Executes a Step's actions in order, handling retry/timeout/validation."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        event_bus: EventBus | None = None,
        default_timeout: float | None = None,
        validator: ResultValidator | None = None,
    ) -> None:
        self.tools = tool_registry
        self.event_bus = event_bus or EventBus()
        self.default_timeout = default_timeout
        self.validator = validator or ResultValidator()

    async def run_step(self, step: Step, context: ExecutionContext) -> ToolResult:
        """Run every action in a step sequentially, returning the last ToolResult."""
        if not step.actions:
            return ToolResult.ok(output=None)

        last_result = ToolResult.ok(output=None)
        for action in step.actions:
            tool = self.tools.get_tool(action.tool_name)
            if tool is None:
                return ToolResult.fail(f"Tool '{action.tool_name}' not found in registry")

            last_result = await self.run_action(tool, action.input, step)
            if context.plan is not None:
                context.plan.context[action.output_key] = (
                    last_result.output if last_result.success else None
                )
                # Tools that produce non-text artifacts (e.g. ReportTool's
                # generated files) surface them via ToolResult.metadata,
                # which otherwise never reaches the plan — stash it under a
                # parallel key so callers (e.g. the Telegram layer) can find
                # e.g. metadata["file_paths"] for the step that answered.
                context.plan.context[f"{action.output_key}__metadata"] = (
                    last_result.metadata if last_result.success else {}
                )

            if not last_result.success:
                return last_result

        return last_result

    async def run_action(self, tool: Tool, tool_input: dict[str, Any], step: Step) -> ToolResult:
        """Run a single tool call with retry/timeout, publishing a ToolExecuted event."""
        timeout = step.timeout or self.default_timeout

        async def _call() -> ToolResult:
            if timeout:
                return await asyncio.wait_for(tool.execute(tool_input), timeout=timeout)
            return await tool.execute(tool_input)

        start = time.monotonic()
        try:
            if step.retry:
                handler = RetryHandler(
                    max_attempts=step.retry.max_attempts,
                    initial_delay=step.retry.initial_delay,
                    max_delay=step.retry.max_delay,
                    backoff_factor=step.retry.backoff_factor,
                    retryable_exceptions=step.retry.retryable_exceptions,
                )
                result = await handler.execute(_call, step_name=step.name)
            else:
                result = await _call()
        except asyncio.TimeoutError:
            # str(TimeoutError()) is "" — asyncio.wait_for raises it with no
            # args, so without this a timed-out step fails with a blank
            # error message that's impossible to diagnose from logs alone.
            result = ToolResult.fail(f"Step '{step.name}' timed out after {timeout}s")
        except Exception as exc:
            result = ToolResult.fail(str(exc) or f"{type(exc).__name__} (no message)")

        is_valid, validation_error = self.validator.validate(result)
        if result.success and not is_valid:
            result = ToolResult.fail(validation_error, metadata=result.metadata)

        duration_ms = int((time.monotonic() - start) * 1000)
        result.duration_ms = result.duration_ms or duration_ms
        self.event_bus.publish(
            ToolExecuted(
                tool_name=tool.name,
                success=result.success,
                duration_ms=result.duration_ms,
                metadata=result.metadata,
            )
        )
        return result

"""Execution context for Hermes V4.

Carries request, plan, step, tool, and workflow context through
the execution pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExecutionContext:
    """Context passed through the execution pipeline.

    Immutable in practice — use ``with_*`` methods to create new
    contexts with updated fields.

    Attributes:
        request_id: Unique request identifier.
        user_id: User identifier (optional).
        request: Raw user request string.
        plan_id: Current plan ID.
        plan: Current plan object.
        step_id: Current step ID.
        step: Current step object.
        tool_name: Current tool name.
        tool_result: Current tool result.
        workflow_id: Current workflow ID.
        started_at: When the request started.
        metadata: Arbitrary metadata dictionary.
    """

    request_id: str
    user_id: str | None = None
    request: str = ""
    plan_id: str | None = None
    plan: Any = None  # Plan object — avoid circular import
    step_id: str | None = None
    step: Any = None  # Step object
    tool_name: str | None = None
    tool_result: Any = None  # ToolResult
    workflow_id: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_plan(self, plan: Any) -> ExecutionContext:
        """Return a new context with the plan set."""
        return ExecutionContext(
            request_id=self.request_id,
            user_id=self.user_id,
            request=self.request,
            plan_id=plan.id if hasattr(plan, "id") else str(plan),
            plan=plan,
            step_id=self.step_id,
            step=self.step,
            tool_name=self.tool_name,
            tool_result=self.tool_result,
            workflow_id=self.workflow_id,
            started_at=self.started_at,
            metadata={**self.metadata},
        )

    def with_step(self, step: Any) -> ExecutionContext:
        """Return a new context with the step set."""
        return ExecutionContext(
            request_id=self.request_id,
            user_id=self.user_id,
            request=self.request,
            plan_id=self.plan_id,
            plan=self.plan,
            step_id=step.id if hasattr(step, "id") else str(step),
            step=step,
            tool_name=self.tool_name,
            tool_result=self.tool_result,
            workflow_id=self.workflow_id,
            started_at=self.started_at,
            metadata={**self.metadata},
        )

    def with_tool(self, tool_name: str, result: Any) -> ExecutionContext:
        """Return a new context with the tool result set."""
        return ExecutionContext(
            request_id=self.request_id,
            user_id=self.user_id,
            request=self.request,
            plan_id=self.plan_id,
            plan=self.plan,
            step_id=self.step_id,
            step=self.step,
            tool_name=tool_name,
            tool_result=result,
            workflow_id=self.workflow_id,
            started_at=self.started_at,
            metadata={**self.metadata},
        )

    @property
    def duration_seconds(self) -> float:
        """Elapsed time since the request started."""
        return (datetime.now() - self.started_at).total_seconds()

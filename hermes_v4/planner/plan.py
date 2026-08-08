"""Plan and Step data structures for Hermes V4.

A Plan is a list of Steps with dependencies.
A Step is a unit of work that may use one or more tools.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PlanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class Action:
    """A single atomic operation within a step.

    Attributes:
        tool_name: Name of the tool to invoke.
        input: Input parameters for the tool.
        output_key: Key to store the result in step context.
    """

    tool_name: str
    input: dict[str, Any] = field(default_factory=dict)
    output_key: str = "output"


@dataclass
class RetryConfig:
    """Retry configuration for a step.

    Attributes:
        max_attempts: Maximum number of attempts (default 3).
        initial_delay: Initial delay in seconds (default 1.0).
        max_delay: Maximum delay in seconds (default 60.0).
        backoff_factor: Exponential backoff factor (default 2.0).
        retryable_exceptions: Tuple of exception types to retry on.
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
    )


@dataclass
class Step:
    """A step in a plan — one or more actions.

    Steps can depend on other steps (DAG execution).
    The planner decides which tools each step uses.

    Attributes:
        id: Unique step identifier.
        name: Human-readable step name.
        description: What this step does.
        actions: List of atomic actions for this step.
        depends_on: Step IDs this step depends on.
        condition: Optional expression to evaluate before executing.
        retry: Retry configuration (None = no retry).
        timeout: Step timeout in seconds (None = use default).
        created_at: When the step was created.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    description: str = ""
    actions: list[Action] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    condition: str | None = None
    retry: RetryConfig | None = None
    timeout: int | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_complete(self) -> bool:
        return hasattr(self, "_status") and getattr(self, "_status") == StepStatus.COMPLETED  # type: ignore[attr-defined]

    @property
    def is_failed(self) -> bool:
        return hasattr(self, "_status") and getattr(self, "_status") == StepStatus.FAILED  # type: ignore[attr-defined]


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Plan:
    """A complete execution plan — a DAG of Steps.

    The planner generates this from a user request.
    The executor runs it through the workflow engine.

    Attributes:
        id: Unique plan identifier.
        request: Original user request.
        steps: Ordered list of steps (topological order).
        context: Shared context between steps.
        status: Current plan status.
        created_at: When the plan was created.
        completed_at: When the plan finished (if completed).
        error: Error message if failed.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    request: str = ""
    steps: list[Step] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.status == PlanStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.status == PlanStatus.FAILED

    @property
    def is_running(self) -> bool:
        return self.status == PlanStatus.RUNNING

    def add_step(self, step: Step) -> None:
        """Add a step to the plan."""
        self.steps.append(step)

    def get_step(self, step_id: str) -> Step | None:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_ready_steps(self) -> list[Step]:
        """Get steps whose dependencies are all completed."""
        ready = []
        for step in self.steps:
            if step.id not in [s.id for s in self.steps if hasattr(s, "_status") and getattr(s, "_status") == StepStatus.COMPLETED]:
                deps = step.depends_on
                if all(
                    self._is_completed(d)
                    for d in deps
                ):
                    ready.append(step)
        return ready

    def _is_completed(self, step_id: str) -> bool:
        """Check if a step is completed."""
        for step in self.steps:
            if step.id == step_id:
                return hasattr(step, "_status") and getattr(step, "_status") == StepStatus.COMPLETED  # type: ignore[attr-defined]
        return False

    def __repr__(self) -> str:
        return (
            f"Plan(id={self.id!r}, status={self.status.value}, "
            f"steps={len(self.steps)})"
        )

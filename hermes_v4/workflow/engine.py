"""Workflow engine for Hermes V4.

Executes a Plan's steps as a DAG: steps whose dependencies are all
completed run concurrently (bounded). Each node delegates to the
Executor to actually run its step's actions; this class only handles
DAG readiness, concurrency, and plan/workflow-level events.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

from hermes_v4.core.base import ToolRegistry
from hermes_v4.core.context import ExecutionContext
from hermes_v4.core.event import (
    EventBus,
    PlanFailed,
    StepCompleted,
    StepFailed,
    StepStarted,
    TaskSaved,
    WorkflowCompleted,
    WorkflowStarted,
)
from hermes_v4.executor.executor import Executor
from hermes_v4.planner.plan import Plan, PlanStatus, StepStatus
from hermes_v4.workflow.builder import WorkflowBuilder
from hermes_v4.workflow.graph import NodeStatus, WorkflowGraph, WorkflowNode

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Executes workflow graphs step by step (DAG, bounded concurrency)."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        event_bus: EventBus | None = None,
        max_parallel: int = 4,
        default_step_timeout: float | None = None,
        executor: Executor | None = None,
        memory=None,
    ) -> None:
        self.tools = tool_registry
        self.event_bus = event_bus or EventBus()
        self.max_parallel = max_parallel
        self.builder = WorkflowBuilder(tool_registry)
        self.executor = executor or Executor(
            tool_registry, event_bus=self.event_bus, default_timeout=default_step_timeout
        )
        self.memory = memory

    async def run(self, plan: Plan, context: ExecutionContext | None = None) -> Plan:
        """Run every step of the plan to completion (or first blocking failure)."""
        context = (context or ExecutionContext(request_id=plan.id)).with_plan(plan)
        graph = self.builder.build(plan, context)

        plan.status = PlanStatus.RUNNING
        self.event_bus.publish(WorkflowStarted(workflow_id=plan.id))
        start = time.monotonic()

        semaphore = asyncio.Semaphore(self.max_parallel)
        while not graph.is_complete():
            ready = graph.get_ready_nodes()
            if not ready:
                self._skip_blocked_nodes(graph)
                break
            await asyncio.gather(*(self._run_node(node, semaphore) for node in ready))

        duration_ms = int((time.monotonic() - start) * 1000)
        failed_nodes = graph.get_failed_nodes()
        if failed_nodes:
            plan.status = PlanStatus.FAILED
            plan.error = "; ".join(
                f"{n.step.name or n.step_id}: {n.result.error if n.result else 'blocked by failed dependency'}"
                for n in failed_nodes
            )
            self.event_bus.publish(PlanFailed(plan_id=plan.id, error=plan.error or ""))
        else:
            plan.status = PlanStatus.COMPLETED

        plan.completed_at = datetime.now()
        self.event_bus.publish(WorkflowCompleted(workflow_id=plan.id, duration_ms=duration_ms))

        if self.memory is not None:
            try:
                await self.memory.save_task(plan)
                self.event_bus.publish(TaskSaved(task_id=plan.id))
            except Exception:
                logger.exception("Failed to persist plan '%s' to memory", plan.id)

        return plan

    def _skip_blocked_nodes(self, graph: WorkflowGraph) -> None:
        """Mark remaining PENDING nodes as failed; their deps never completed."""
        for node in graph.nodes.values():
            if node.status == NodeStatus.PENDING:
                node.status = NodeStatus.FAILED
                node.step._status = StepStatus.FAILED  # type: ignore[attr-defined]

    async def _run_node(self, node: WorkflowNode, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            node.status = NodeStatus.RUNNING
            node.started_at = datetime.now()
            step = node.step
            self.event_bus.publish(
                StepStarted(plan_id=node.context.plan_id or "", step_id=step.id, step_name=step.name)
            )
            step_start = time.monotonic()

            try:
                result = await self.executor.run_step(step, node.context)
                error = None if result.success else result.error
            except Exception as exc:
                logger.exception("Step '%s' raised unexpectedly", step.name)
                result = None
                error = str(exc)

            duration_ms = int((time.monotonic() - step_start) * 1000)
            node.result = result
            node.completed_at = datetime.now()

            if error is None and result is not None and result.success:
                node.status = NodeStatus.COMPLETED
                step._status = StepStatus.COMPLETED  # type: ignore[attr-defined]
                self.event_bus.publish(
                    StepCompleted(
                        plan_id=node.context.plan_id or "",
                        step_id=step.id,
                        step_name=step.name,
                        duration_ms=duration_ms,
                    )
                )
            else:
                node.status = NodeStatus.FAILED
                step._status = StepStatus.FAILED  # type: ignore[attr-defined]
                self.event_bus.publish(
                    StepFailed(
                        plan_id=node.context.plan_id or "",
                        step_id=step.id,
                        step_name=step.name,
                        error=error or "unknown error",
                    )
                )

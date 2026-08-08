"""Workflow builder for Hermes V4.

Constructs a WorkflowGraph from a Plan, resolving each step's primary
tool from the registry (informational — the engine resolves each
action's own tool at execution time since a step may use several).
"""

from __future__ import annotations

from hermes_v4.core.base import ToolRegistry
from hermes_v4.core.context import ExecutionContext
from hermes_v4.planner.plan import Plan
from hermes_v4.workflow.graph import WorkflowGraph, WorkflowNode


class WorkflowBuilder:
    """Builds a WorkflowGraph from a Plan."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tools = tool_registry

    def build(self, plan: Plan, context: ExecutionContext) -> WorkflowGraph:
        graph = WorkflowGraph()
        for step in plan.steps:
            primary_tool_name = step.actions[0].tool_name if step.actions else None
            tool = self.tools.get_tool(primary_tool_name) if primary_tool_name else None
            node = WorkflowNode(
                step_id=step.id,
                step=step,
                tool=tool,
                context=context.with_step(step),
            )
            graph.add_node(node)
        return graph

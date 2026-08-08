"""DAG-based workflow graph for Hermes V4.

WorkflowGraph represents the plan as a directed acyclic graph.
WorkflowNode represents each step in the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from hermes_v4.core.base import Tool
from hermes_v4.core.context import ExecutionContext
from hermes_v4.planner.plan import Step


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowNode:
    """A node in the workflow DAG.

    Wraps a Step with execution state and tool reference.

    Attributes:
        step_id: ID of the associated Step.
        step: The Step this node represents.
        tool: The tool to use for this step.
        context: Execution context for this node.
        status: Current node status.
        result: Tool result (if completed).
        started_at: When execution started.
        completed_at: When execution completed.
    """

    step_id: str
    step: Step
    tool: Tool
    context: ExecutionContext
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def is_ready(self) -> bool:
        """Check if all dependencies are completed."""
        if self.status != NodeStatus.PENDING:
            return False
        return True  # Dependencies checked by WorkflowGraph


class WorkflowGraph:
    """DAG-based workflow graph.

    Manages nodes and their dependencies.
    Provides methods to discover ready nodes and check completion.

    Example:
        >>> graph = WorkflowGraph()
        >>> graph.add_node(node_a)
        >>> graph.add_node(node_b)  # depends on node_a
        >>> ready = graph.get_ready_nodes()
        >>> # ready contains node_a (no deps)
    """

    def __init__(self) -> None:
        self.nodes: dict[str, WorkflowNode] = {}
        self.edges: dict[str, list[str]] = {}  # node_id -> dependent node_ids

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.step_id] = node
        self.edges[node.step_id] = node.step.depends_on or []

    def get_ready_nodes(self) -> list[WorkflowNode]:
        """Return nodes whose dependencies are all completed.

        Only returns PENDING nodes that have no unmet dependencies.
        """
        ready = []
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            deps = self.edges.get(node.step_id, [])
            if not deps:
                ready.append(node)
                continue
            # Check all dependencies are completed
            if all(
                self.nodes[d].status == NodeStatus.COMPLETED
                for d in deps
                if d in self.nodes
            ):
                ready.append(node)
        return ready

    def is_complete(self) -> bool:
        """Check if all nodes are completed or skipped."""
        if not self.nodes:
            return True
        return all(
            n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
            for n in self.nodes.values()
        )

    def get_failed_nodes(self) -> list[WorkflowNode]:
        """Return nodes that failed."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.FAILED]

    def get_completed_count(self) -> int:
        """Return count of completed nodes."""
        return sum(
            1 for n in self.nodes.values()
            if n.status == NodeStatus.COMPLETED
        )

    def __len__(self) -> int:
        return len(self.nodes)

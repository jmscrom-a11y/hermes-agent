"""Base classes for Hermes V4.

Defines the foundational interfaces:
- Tool: all tools must implement this
- ToolResult: structured result from tool execution
- ToolRegistry: registry of available tools
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any


class ToolResult:
    """Structured result from a tool execution.

    Attributes:
        success: Whether the tool execution succeeded.
        output: The tool's output (type depends on tool).
        error: Error message if failed, None otherwise.
        metadata: Additional metadata (timing, counts, etc.).
        duration_ms: Execution time in milliseconds.
    """

    __slots__ = ("success", "output", "error", "metadata", "duration_ms")

    def __init__(
        self,
        success: bool,
        output: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        duration_ms: int = 0,
    ) -> None:
        self.success = success
        self.output = output
        self.error = error
        self.metadata = metadata or {}
        self.duration_ms = duration_ms

    @classmethod
    def ok(cls, output: Any, metadata: dict[str, Any] | None = None) -> ToolResult:
        """Create a successful result."""
        return cls(success=True, output=output, error=None, metadata=metadata)

    @classmethod
    def fail(cls, error: str, metadata: dict[str, Any] | None = None) -> ToolResult:
        """Create a failed result."""
        return cls(success=False, output=None, error=error, metadata=metadata)

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL: {self.error}"
        return f"ToolResult({status}, {self.duration_ms}ms)"


class Tool(abc.ABC):
    """Base class for all Hermes V4 tools.

    Every tool MUST:
    - Have a unique ``name`` (lowercase, no spaces)
    - Have a ``description`` the planner uses for tool selection
    - Implement ``execute(input)`` returning a ToolResult
    - Implement ``validate_input(input)`` returning list of error strings

    The planner queries the ToolRegistry for tool specs and decides
    which tools to use — never hardcode routing.
    """

    name: str = ""
    description: str = ""
    enabled: bool = True
    version: str = "1.0.0"
    input_schema: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]
    output_schema: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]

    @abc.abstractmethod
    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given input.

        Args:
            input: Tool-specific input parameters matching input_schema.

        Returns:
            ToolResult with success/failure status and output.
        """
        ...

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        """Validate input parameters. Returns list of error messages (empty = valid)."""
        return []

    def get_spec(self) -> dict[str, Any]:
        """Return tool specification for planner discovery.

        The planner uses this to know what tools are available and
        how to invoke them.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "enabled": self.enabled,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled})"


@dataclass
class ToolInfo:
    """Metadata about a registered tool (lightweight, no tool instance)."""

    name: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "enabled": self.enabled,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


class ToolRegistry:
    """Registry of available tools.

    The planner queries this to discover tools. Tools register themselves
    and are discovered by name.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(GitTool())
        >>> registry.register(TavilyTool())
        >>> specs = registry.get_available_tools()
        >>> # specs is a list of ToolInfo for the planner
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Only enabled tools are discoverable."""
        if tool.enabled:
            self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_available_tools(self) -> list[ToolInfo]:
        """Get all enabled tools for planner discovery.

        Returns:
            List of ToolInfo that the planner can use to decide
            which tools to invoke.
        """
        return [
            ToolInfo(
                name=t.name,
                description=t.description,
                version=t.version,
                enabled=t.enabled,
                input_schema=t.input_schema,
                output_schema=t.output_schema,
            )
            for t in self._tools.values()
            if t.enabled
        ]

    def list_tools(self) -> list[str]:
        """List all enabled tool names."""
        return [name for name, tool in self._tools.items() if tool.enabled]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

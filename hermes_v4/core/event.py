"""Event system for Hermes V4.

Publish/subscribe event bus for all system events.
Every state change publishes events for logging, notifications, and monitoring.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Event type aliases
EventHandler = Callable[["Event"], None] | Callable[["Event"], Any]


class Event:
    """Base class for all Hermes V4 events."""

    event_type: str = "event"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event for storage/transmission."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.event_type!r})"


# ── Plan Events ──────────────────────────────────────────────────

@dataclass
class PlanCreated(Event):
    event_type = "plan_created"
    plan_id: str = ""
    request: str = ""


@dataclass
class PlanCompleted(Event):
    event_type = "plan_completed"
    plan_id: str = ""
    duration_ms: int = 0


@dataclass
class PlanFailed(Event):
    event_type = "plan_failed"
    plan_id: str = ""
    error: str = ""


# ── Step Events ──────────────────────────────────────────────────

@dataclass
class StepStarted(Event):
    event_type = "step_started"
    plan_id: str = ""
    step_id: str = ""
    step_name: str = ""


@dataclass
class StepCompleted(Event):
    event_type = "step_completed"
    plan_id: str = ""
    step_id: str = ""
    step_name: str = ""
    duration_ms: int = 0


@dataclass
class StepFailed(Event):
    event_type = "step_failed"
    plan_id: str = ""
    step_id: str = ""
    step_name: str = ""
    error: str = ""
    retryable: bool = False


# ── Tool Events ──────────────────────────────────────────────────

@dataclass
class ToolExecuted(Event):
    event_type = "tool_executed"
    tool_name: str = ""
    success: bool = False
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Workflow Events ──────────────────────────────────────────────

@dataclass
class WorkflowStarted(Event):
    event_type = "workflow_started"
    workflow_id: str = ""


@dataclass
class WorkflowCompleted(Event):
    event_type = "workflow_completed"
    workflow_id: str = ""
    duration_ms: int = 0


@dataclass
class WorkflowResumed(Event):
    event_type = "workflow_resumed"
    workflow_id: str = ""


# ── Memory Events ────────────────────────────────────────────────

@dataclass
class TaskSaved(Event):
    event_type = "task_saved"
    task_id: str = ""


@dataclass
class WorkflowStateSaved(Event):
    event_type = "workflow_state_saved"
    workflow_id: str = ""


# ── EventBus ─────────────────────────────────────────────────────

class EventBus:
    """Pub/sub event bus for all system events.

    Events are dispatched synchronously by default.
    For async handlers, use ``publish_async``.

    Example:
        >>> bus = EventBus()
        >>> bus.subscribe("step_completed", print_event)
        >>> bus.publish(StepCompleted(...))
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, callback: EventHandler) -> None:
        """Subscribe to an event type."""
        self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: EventHandler) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers (sync)."""
        for callback in self._subscribers.get(event.event_type, []):
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                logger.exception("Event handler error for %s", event.event_type)

    def publish_all(self, event: Event) -> None:
        """Publish to ALL subscribers regardless of event type."""
        for callbacks in self._subscribers.values():
            for callback in callbacks:
                try:
                    result = callback(event)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception:
                    logger.exception("Event handler error (all) for %s", event.event_type)

    def clear(self) -> None:
        """Clear all subscriptions."""
        self._subscribers.clear()

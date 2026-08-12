"""Claude Code usage/cost tracking for Hermes V4.

`claude -p --output-format json` reports total_cost_usd per call, but
nothing accumulates it anywhere — ClaudeCodeTool.execute() calls
record_usage() on every successful call, and UsageTool (tools/usage_tool.py)
reads it back on request. Plain JSONL so it stays inspectable/appendable
without a DB dependency; call volume here is low (one line per Claude Code
invocation, at most a few dozen a day).
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timezone
from typing import Any

from hermes_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


def _log_path() -> pathlib.Path:
    return pathlib.Path(get_settings().CLAUDE_CODE_USAGE_LOG_PATH)


def record_usage(cost_usd: float | None, num_turns: int | None, session_id: str | None, purpose: str = "") -> None:
    """Best-effort append — a logging failure must never fail the calling tool."""
    if cost_usd is None:
        return
    path = _log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cost_usd": cost_usd,
            "num_turns": num_turns,
            "session_id": session_id,
            "purpose": purpose,
        }
        with path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("Failed to record claude_code usage", exc_info=True)


def summarize_usage() -> dict[str, Any]:
    path = _log_path()
    if not path.exists():
        return {"all_time_cost_usd": 0.0, "all_time_calls": 0, "month_cost_usd": 0.0, "month_calls": 0}

    now = datetime.now(timezone.utc)
    all_time_cost = 0.0
    all_time_calls = 0
    month_cost = 0.0
    month_calls = 0

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cost = float(entry["cost_usd"])
                ts = datetime.fromisoformat(entry["timestamp"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            all_time_cost += cost
            all_time_calls += 1
            if ts.year == now.year and ts.month == now.month:
                month_cost += cost
                month_calls += 1

    return {
        "all_time_cost_usd": all_time_cost,
        "all_time_calls": all_time_calls,
        "month_cost_usd": month_cost,
        "month_calls": month_calls,
    }

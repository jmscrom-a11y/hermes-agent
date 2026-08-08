"""SQLite-backed task history store for Hermes V4.

Persists completed/failed plans so past runs can be inspected and old
ones pruned. sqlite3 is blocking, so every call goes through
asyncio.to_thread to avoid stalling the event loop.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from typing import Any

from hermes_v4.config.settings import get_settings
from hermes_v4.planner.plan import Plan

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    request TEXT,
    status TEXT,
    context TEXT,
    error TEXT,
    created_at TEXT,
    completed_at TEXT
)
"""


class SqliteMemoryStore:
    """Stores Plan runs as task records in SQLite."""

    def __init__(self, db_path: str | pathlib.Path | None = None, retention_days: int | None = None) -> None:
        settings = get_settings()
        self.db_path = pathlib.Path(db_path or settings.MEMORY_SQLITE_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = (
            retention_days if retention_days is not None else settings.MEMORY_TASK_RETENTION_DAYS
        )
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def _save_task_sync(self, plan: Plan) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks
                    (id, request, status, context, error, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.id,
                    plan.request,
                    plan.status.value,
                    json.dumps(plan.context, default=str),
                    plan.error,
                    plan.created_at.isoformat(),
                    plan.completed_at.isoformat() if plan.completed_at else None,
                ),
            )

    async def save_task(self, plan: Plan) -> None:
        await asyncio.to_thread(self._save_task_sync, plan)

    def _get_task_sync(self, task_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            return self._row_to_dict(row) if row else None

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_task_sync, task_id)

    def _list_recent_sync(self, limit: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    async def list_recent_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_recent_sync, limit)

    def _prune_sync(self) -> int:
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute("DELETE FROM tasks WHERE created_at < ?", (cutoff,))
            return cursor.rowcount

    async def prune_old_tasks(self) -> int:
        """Delete tasks older than retention_days. Returns rows deleted."""
        if not self.retention_days or self.retention_days <= 0:
            return 0
        return await asyncio.to_thread(self._prune_sync)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["context"] = json.loads(data["context"]) if data["context"] else {}
        return data

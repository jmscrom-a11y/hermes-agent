"""Claude Code tool for Hermes V4.

Invokes the `claude` CLI in headless mode (`-p --output-format json`) so
the planner can delegate coding tasks (bug fixes, refactors, analysis)
to Claude Code against a real workspace on disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import time
from typing import Any

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import Tool, ToolResult

VALID_PERMISSION_MODES = {"acceptEdits", "auto", "bypassPermissions", "manual", "dontAsk", "plan"}


class ClaudeCodeTool(Tool):
    """Delegates a coding task to the Claude Code CLI against a workspace."""

    name = "claude_code"
    description = (
        "코드 작업(버그 수정, 리팩터링, 기능 구현, 코드/저장소 분석)을 Claude Code에게 위임합니다. "
        "실제 파일을 읽고 수정할 수 있습니다."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "cwd": {"type": "string"},
            "permission_mode": {"type": "string"},
            "allowed_tools": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["task"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "session_id": {"type": "string"},
            "total_cost_usd": {"type": "number"},
            "num_turns": {"type": "integer"},
        },
    }

    def __init__(
        self,
        workspace: str | pathlib.Path | None = None,
        permission_mode: str | None = None,
        timeout: float | None = None,
        claude_bin: str | None = None,
    ) -> None:
        settings = get_settings()
        self._workspace = pathlib.Path(workspace or settings.HERMES_WORKSPACE).resolve()
        self._permission_mode = permission_mode or settings.CLAUDE_CODE_PERMISSION_MODE
        self._timeout = timeout if timeout is not None else settings.CLAUDE_CODE_TIMEOUT
        self._claude_bin = claude_bin or settings.CLAUDE_CODE_BIN
        self._unset_api_key = settings.CLAUDE_CODE_UNSET_API_KEY

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        task = input.get("task")
        if not task or not str(task).strip():
            errors.append("'task' is required and must be a non-empty string")

        mode = input.get("permission_mode")
        if mode is not None and mode not in VALID_PERMISSION_MODES:
            errors.append(f"'permission_mode' must be one of {sorted(VALID_PERMISSION_MODES)}")

        cwd = input.get("cwd")
        if cwd is not None and not pathlib.Path(cwd).is_dir():
            errors.append(f"'cwd' does not exist or is not a directory: {cwd}")

        return errors

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self._unset_api_key:
            env.pop("ANTHROPIC_API_KEY", None)
        return env

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))

        task = str(input["task"]).strip()
        cwd = pathlib.Path(input["cwd"]).resolve() if input.get("cwd") else self._workspace
        permission_mode = input.get("permission_mode") or self._permission_mode
        allowed_tools = input.get("allowed_tools")

        cmd = [
            self._claude_bin,
            "-p", task,
            "--output-format", "json",
            "--permission-mode", permission_mode,
        ]
        if allowed_tools:
            cmd += ["--allowed-tools", *allowed_tools]

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                env=self._build_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult.fail(f"claude_code timed out after {self._timeout}s")
        except FileNotFoundError:
            return ToolResult.fail(f"'{self._claude_bin}' executable not found on PATH")

        duration_ms = int((time.monotonic() - start) * 1000)

        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return ToolResult.fail(
                f"Failed to parse claude_code output: {exc}",
                metadata={"stderr": stderr.decode("utf-8", errors="replace")},
            )

        if payload.get("is_error"):
            return ToolResult.fail(
                payload.get("result") or "claude_code reported an error",
                metadata={
                    "session_id": payload.get("session_id"),
                    "subtype": payload.get("subtype"),
                    "permission_denials": payload.get("permission_denials"),
                },
            )

        return ToolResult(
            success=True,
            output=payload.get("result"),
            metadata={
                "session_id": payload.get("session_id"),
                "total_cost_usd": payload.get("total_cost_usd"),
                "num_turns": payload.get("num_turns"),
                "permission_denials": payload.get("permission_denials"),
            },
            duration_ms=duration_ms,
        )

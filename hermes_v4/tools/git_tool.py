"""Git tool for Hermes V4.

Runs a whitelisted set of git subcommands as a subprocess against a
workspace. push/reset/checkout/rebase/clean are deliberately excluded
(see GIT_TOOL_ALLOWED_COMMANDS) since those can discard local work or
touch the shared remote — not something an LLM-triggered tool call
should do unsupervised.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
from typing import Any

from hermes_v4.config.settings import get_settings
from hermes_v4.core.base import Tool, ToolResult


class GitTool(Tool):
    """Runs whitelisted git subcommands against a repository."""

    name = "git"
    description = (
        "저장소의 git 상태 확인, 변경사항 조회, 커밋 등을 수행합니다 "
        "(status, diff, log, branch, show, add, commit). push나 기록을 되돌리는 "
        "명령은 지원하지 않습니다."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "args": {"type": "array", "items": {"type": "string"}},
            "cwd": {"type": "string"},
        },
        "required": ["command"],
    }
    output_schema = {
        "type": "object",
        "properties": {"stdout": {"type": "string"}},
    }

    def __init__(self, workspace: str | pathlib.Path | None = None, allowed_commands: list[str] | None = None) -> None:
        settings = get_settings()
        self._workspace = pathlib.Path(workspace or settings.HERMES_WORKSPACE).resolve()
        self._allowed_commands = set(allowed_commands or settings.GIT_TOOL_ALLOWED_COMMANDS)

    async def validate_input(self, input: dict[str, Any]) -> list[str]:
        errors = []
        command = input.get("command")
        if not command:
            errors.append("'command' is required")
        elif command not in self._allowed_commands:
            errors.append(
                f"'{command}' is not allowed; must be one of {sorted(self._allowed_commands)}"
            )

        args = input.get("args")
        if args is not None and not isinstance(args, list):
            errors.append("'args' must be a list of strings")

        cwd = input.get("cwd")
        if cwd is not None and not pathlib.Path(cwd).is_dir():
            errors.append(f"'cwd' does not exist or is not a directory: {cwd}")

        return errors

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        errors = await self.validate_input(input)
        if errors:
            return ToolResult.fail("; ".join(errors))

        command = input["command"]
        args = [str(a) for a in input.get("args") or []]
        cwd = pathlib.Path(input["cwd"]).resolve() if input.get("cwd") else self._workspace

        cmd = ["git", command, *args]
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            return ToolResult.fail("'git' executable not found on PATH")

        duration_ms = int((time.monotonic() - start) * 1000)
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            return ToolResult.fail(
                stderr_text or f"git {command} exited with code {proc.returncode}",
                metadata={"returncode": proc.returncode, "stdout": stdout_text},
            )

        return ToolResult(
            success=True,
            output=stdout_text,
            metadata={"returncode": proc.returncode, "stderr": stderr_text},
            duration_ms=duration_ms,
        )

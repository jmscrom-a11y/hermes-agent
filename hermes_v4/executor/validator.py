"""Result validation utilities."""

from __future__ import annotations

import json
from typing import Any

from hermes_v4.core.base import ToolResult


class ResultValidator:
    """도구 실행 결과 검증.

    Attributes:
        require_output: output가 None이면 실패로 간주.
        require_success: success=False이면 실패로 간주.
        custom_checks: 커스텀 검증 함수 목록.
    """

    def __init__(
        self,
        require_output: bool = False,
        require_success: bool = True,
        custom_checks: list | None = None,
    ) -> None:
        self.require_output = require_output
        self.require_success = require_success
        self.custom_checks = custom_checks or []

    def validate(self, result: ToolResult) -> tuple[bool, str]:
        """결과를 검증. (성공 여부, 에러 메시지) 반환."""
        if self.require_success and not result.success:
            return False, f"Tool failed: {result.error}"

        if self.require_output and result.output is None:
            return False, "Tool returned no output"

        for check in self.custom_checks:
            try:
                valid, msg = check(result)
                if not valid:
                    return False, msg
            except Exception as exc:
                return False, f"Custom check failed: {exc}"

        return True, ""

    def validate_json(self, result: ToolResult) -> tuple[bool, dict | None]:
        """결과가 JSON이면 파싱해서 반환."""
        if not result.success:
            return False, None

        content = result.output
        if isinstance(content, str):
            try:
                return True, json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return False, None
        elif isinstance(content, dict):
            return True, content
        else:
            return False, None

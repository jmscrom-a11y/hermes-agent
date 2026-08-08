"""Retry handler with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class RetryHandler:
    """지수 백오프 기반 재시도 핸들러.

    Attributes:
        max_attempts: 최대 시도 횟수.
        initial_delay: 초기 대기 시간 (초).
        max_delay: 최대 대기 시간 (초).
        backoff_factor: 지수 증가율.
        retryable_exceptions: 재시도할 예외 타입 목록.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retryable_exceptions = retryable_exceptions or (TimeoutError, ConnectionError)

    async def execute(
        self,
        coro_factory: Callable[[], Awaitable],
        step_name: str = "unknown",
    ):
        """coroutine을 재시도하며 실행.

        Args:
            coro_factory: 호출할 coroutine을 만드는工厂 함수.
            step_name: 로깅용 단계 이름.

        Returns:
           最后一次成功的结果.

        Raises:
            Exception: 最终attempt失败时的异常.
        """
        last_exc = None
        delay = self.initial_delay

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await coro_factory()
            except tuple(self.retryable_exceptions) as exc:
                last_exc = exc
                logger.warning(
                    "Retry %d/%d for '%s': %s (will retry in %.1fs)",
                    attempt, self.max_attempts, step_name, exc, delay,
                )
                if attempt < self.max_attempts:
                    await asyncio.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
            except Exception as exc:
                # 재시도하지 않을 예외는 즉시 re-raise
                raise

        raise last_exc

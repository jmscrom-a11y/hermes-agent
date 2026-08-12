"""Ollama implementation of LLMProvider for Hermes V4.

Talks to Ollama's native /api/chat endpoint asynchronously.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from hermes_v4.config.settings import get_settings
from hermes_v4.llm.provider import ChatCompletion, LLMProvider

logger = logging.getLogger(__name__)


def _normalize_ollama_host(url: str) -> str:
    """Strip any path (e.g. /v1) from the base URL.

    Ollama's native API (/api/...) expects a bare host; a path like /v1
    left over from OpenAI-compatible configs would be prepended to every
    call and produce 404s.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    return f"{scheme}://{host}:{port}"


class OllamaProvider(LLMProvider):
    """LLMProvider backed by a local Ollama instance."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 600.0,
        think: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.host = _normalize_ollama_host(base_url or settings.LLM_BASE_URL)
        self.default_model = model or settings.LLM_MODEL
        self.default_temperature = settings.LLM_TEMPERATURE
        self.default_max_tokens = settings.LLM_MAX_TOKENS
        self.timeout = timeout
        # Thinking-capable models (e.g. this project's ornith:9b) spend the
        # bulk of their latency on a hidden reasoning pass by default — 22.9s
        # vs 0.7s for the same prompt, measured with think on vs off. Off by
        # default; only worth enabling for genuinely hard reasoning tasks.
        self.think = think if think is not None else settings.LLM_THINKING_ENABLED
        self.chat_keep_alive = settings.LLM_CHAT_KEEP_ALIVE
        self.vision_model = settings.LLM_VISION_MODEL

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        num_ctx: int | None = None,
    ) -> ChatCompletion:
        chat_model = model or self.default_model
        options: dict[str, float | int] = {
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        effective_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        if effective_max_tokens:
            options["num_predict"] = effective_max_tokens
        # Ollama runs with a small default context window (observed:
        # responses to a ~6000-char input got cut off with done_reason
        # "length" at ~1300 output tokens despite num_predict=40000) unless
        # num_ctx is explicitly set — the model's own supported context
        # (e.g. 262K for qwen3.6:27b) doesn't change what Ollama actually
        # allocates per-request.
        if num_ctx:
            options["num_ctx"] = num_ctx

        payload: dict[str, Any] = {
            "model": chat_model,
            "messages": messages,
            "stream": False,
            "options": options,
            "think": self.think,
        }
        if response_format == "json":
            payload["format"] = "json"
        # Only the default chat/planning model gets the longer keep_alive —
        # an explicit model override (e.g. ReportTool's bigger REPORT_LLM_MODEL)
        # falls back to Ollama's own default (5m) so it isn't held in memory
        # for a one-off request.
        if model is None or model == self.default_model:
            # Ollama's keep_alive accepts a duration string ("30m") or a
            # plain number of seconds / -1 for "never unload" — but only as
            # a JSON number, not a numeric string. Sending "-1" as a string
            # hits Go's time.ParseDuration (which requires a unit suffix)
            # and 400s the whole request.
            keep_alive = self.chat_keep_alive
            try:
                keep_alive = int(keep_alive)
            except ValueError:
                pass
            payload["keep_alive"] = keep_alive

        logger.info("Sending chat request to Ollama at %s (model=%s)", self.host, chat_model)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        return ChatCompletion(content=data["message"]["content"], model=chat_model)

    async def generate_vision(
        self, prompt: str, images: list[str], model: str | None = None
    ) -> ChatCompletion:
        """Ask a vision-capable model (default: settings.LLM_VISION_MODEL,
        e.g. llava) about one or more images.

        Not part of the LLMProvider interface — only Ollama has a local
        vision model available, so callers (the Telegram photo handler)
        feature-detect this method rather than every provider needing to
        implement it.

        Args:
            images: Base64-encoded image bytes (no "data:" URI prefix) —
                Ollama's native /api/chat format.
        """
        chat_model = model or self.vision_model
        payload: dict[str, Any] = {
            "model": chat_model,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "stream": False,
            "think": False,
        }

        logger.info("Sending vision chat request to Ollama at %s (model=%s)", self.host, chat_model)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        return ChatCompletion(content=data["message"]["content"], model=chat_model)

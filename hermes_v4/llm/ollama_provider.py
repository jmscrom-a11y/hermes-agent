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
    ) -> None:
        settings = get_settings()
        self.host = _normalize_ollama_host(base_url or settings.LLM_BASE_URL)
        self.default_model = model or settings.LLM_MODEL
        self.default_temperature = settings.LLM_TEMPERATURE
        self.default_max_tokens = settings.LLM_MAX_TOKENS
        self.timeout = timeout

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> ChatCompletion:
        chat_model = model or self.default_model
        options: dict[str, float | int] = {
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        effective_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        if effective_max_tokens:
            options["num_predict"] = effective_max_tokens

        payload: dict[str, Any] = {
            "model": chat_model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if response_format == "json":
            payload["format"] = "json"

        logger.info("Sending chat request to Ollama at %s (model=%s)", self.host, chat_model)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        return ChatCompletion(content=data["message"]["content"], model=chat_model)

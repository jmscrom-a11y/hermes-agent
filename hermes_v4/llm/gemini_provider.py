"""Gemini implementation of LLMProvider for Hermes V4.

Talks to the Generative Language REST API directly (no extra SDK
dependency — the project already depends on httpx).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from hermes_v4.config.settings import get_settings
from hermes_v4.llm.provider import ChatCompletion, LLMProvider

logger = logging.getLogger(__name__)

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    """LLMProvider backed by Google's Gemini API."""

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 120.0) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.default_model = model or settings.GEMINI_MODEL
        self.default_temperature = settings.LLM_TEMPERATURE
        self.default_max_tokens = settings.LLM_MAX_TOKENS
        self.timeout = timeout

    @staticmethod
    def _to_gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Gemini has no "system" role; fold system messages into the first user turn."""
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        contents = []
        for m in messages:
            if m.get("role") == "system":
                continue
            role = "model" if m.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        if system_parts and contents and contents[0]["role"] == "user":
            contents[0]["parts"][0]["text"] = (
                "\n\n".join(system_parts) + "\n\n" + contents[0]["parts"][0]["text"]
            )
        return contents

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> ChatCompletion:
        chat_model = model or self.default_model
        generation_config: dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        effective_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        if effective_max_tokens:
            generation_config["maxOutputTokens"] = effective_max_tokens
        if response_format == "json":
            generation_config["responseMimeType"] = "application/json"

        payload = {
            "contents": self._to_gemini_contents(messages),
            "generationConfig": generation_config,
        }

        logger.info("Sending chat request to Gemini (model=%s)", chat_model)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{_API_BASE}/models/{chat_model}:generateContent",
                params={"key": self.api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return ChatCompletion(content=text, model=chat_model)

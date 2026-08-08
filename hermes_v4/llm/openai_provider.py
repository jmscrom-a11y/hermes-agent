"""OpenAI implementation of LLMProvider for Hermes V4."""

from __future__ import annotations

import logging

from hermes_v4.config.settings import get_settings
from hermes_v4.llm.provider import ChatCompletion, LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """LLMProvider backed by the OpenAI Chat Completions API."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        from openai import AsyncOpenAI

        settings = get_settings()
        key = api_key or settings.OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = AsyncOpenAI(api_key=key)
        self.default_model = model or settings.OPENAI_MODEL
        self.default_temperature = settings.LLM_TEMPERATURE
        self.default_max_tokens = settings.LLM_MAX_TOKENS

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> ChatCompletion:
        chat_model = model or self.default_model
        extra: dict = {}
        if response_format == "json":
            # Requires the word "json" to appear in the messages — the
            # planner's system prompt already says "Respond with JSON only".
            extra["response_format"] = {"type": "json_object"}

        logger.info("Sending chat request to OpenAI (model=%s)", chat_model)
        response = await self.client.chat.completions.create(
            model=chat_model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            **extra,
        )
        return ChatCompletion(content=response.choices[0].message.content or "", model=chat_model)

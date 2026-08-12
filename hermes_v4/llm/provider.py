"""LLM provider interface for Hermes V4.

Provides a unified async interface for LLM providers
(Ollama, OpenAI, etc.).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """A single chat message.

    Attributes:
        role: Message role (system, user, assistant).
        content: Message content.
    """

    role: str
    content: str


@dataclass
class ChatCompletion:
    """LLM response.

    Attributes:
        content: Response text.
        model: Model that generated the response.
    """

    content: str
    model: str = ""


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement ``generate``.

    Example:
        >>> class OllamaProvider(LLMProvider):
        ...     async def generate(self, messages):
        ...         ...
    """

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        num_ctx: int | None = None,
    ) -> ChatCompletion:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Optional model override.
            temperature: Optional temperature override.
            max_tokens: Optional max tokens override.
            response_format: If "json", instruct the provider to force
                syntactically valid JSON output (small local models in
                particular tend to ignore prose instructions to do this).
            num_ctx: Ollama-only context window override (ignored by
                hosted providers, which size context server-side). Ollama
                silently runs with a small default context window (~2-4K
                tokens) regardless of what the model itself supports —
                callers sending a large prompt and/or expecting a large
                response (e.g. summarizing/translating a multi-thousand-
                character document chunk) must raise this explicitly or
                risk the response being cut off mid-generation with no
                error, only a truncated result.

        Returns:
            ChatCompletion with the model's response.
        """
        ...

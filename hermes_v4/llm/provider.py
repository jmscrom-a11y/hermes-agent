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

        Returns:
            ChatCompletion with the model's response.
        """
        ...

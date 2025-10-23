from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class LLMMessage:
    """A single message exchanged with an LLM."""

    role: str
    content: str
    name: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class LLMResult:
    """Represents a full LLM response."""

    text: str
    raw: Mapping[str, Any]
    usage: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class LLMStreamChunk:
    """A streaming chunk of an LLM response."""

    text_delta: str
    raw: Mapping[str, Any]


class LLMClient(ABC):
    """Unified interface to interact with an LLM provider."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResult:
        """Execute a single-turn completion request."""

    @abstractmethod
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """Execute a multi-turn chat style request."""

    def stream_chat(
        self, messages: Sequence[LLMMessage], **kwargs: Any
    ) -> Iterable[LLMStreamChunk]:
        """Optional streaming chat interface."""
        raise NotImplementedError("Streaming not implemented for this client.")

    def warmup(self) -> None:
        """Hook for clients that need a warmup call before first real request."""


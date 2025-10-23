from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .tools import ToolInvocation, ToolResponse


class BaseTool(ABC):
    """Abstract interface that every concrete tool must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool within a registry."""

    @property
    def description(self) -> str:
        """Human readable description used by agents when choosing a tool."""
        return self.__doc__ or self.name

    @abstractmethod
    def prototype(self) -> Dict[str, Any]:
        """Describe the expected input arguments (e.g. a JSON schema)."""

    @abstractmethod
    def response_schema(self) -> Dict[str, Any]:
        """Describe the structure of the tool response."""

    @abstractmethod
    def invoke(self, invocation: ToolInvocation) -> ToolResponse:
        """Execute the tool's logic."""

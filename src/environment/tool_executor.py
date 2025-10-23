from __future__ import annotations

from abc import ABC, abstractmethod

from ..tools.tool_registry import ToolRegistry
from ..tools.tools import ToolInvocation, ToolResponse


class ToolExecutor(ABC):
    """Executes registered tools inside the environment."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @abstractmethod
    def execute(self, invocation: ToolInvocation) -> ToolResponse:
        """Execute a tool invocation."""


class DefaultToolExecutor(ToolExecutor):
    """Basic executor that delegates to the registry synchronously."""

    def execute(self, invocation: ToolInvocation) -> ToolResponse:
        return self.tool_registry.invoke(invocation)

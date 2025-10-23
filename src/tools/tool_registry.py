from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

from .base import BaseTool
from .tools import ToolInvocation, ToolMetadata, ToolResponse


class ToolRegistryError(Exception):
    """Base error class for tool registry issues."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool is not present in the registry."""


class DuplicateToolError(ToolRegistryError):
    """Raised when trying to register a tool that already exists."""


@dataclass(frozen=True)
class RegisteredTool:
    """Snapshot of a registered tool."""

    name: str
    tool: BaseTool


class ToolRegistry:
    """Container responsible for storing and invoking tools."""

    def __init__(self, tools: Optional[Iterable[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        if tools is not None:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool, *, overwrite: bool = False) -> None:
        if not overwrite and tool.name in self._tools:
            raise DuplicateToolError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' does not exist.")
        del self._tools[name]

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' not found.") from exc

    def list_tools(self) -> List[ToolMetadata]:
        return [
            ToolMetadata(
                name=tool.name,
                description=tool.description,
                prototype=tool.prototype(),
                response_schema=tool.response_schema(),
            )
            for tool in self._tools.values()
        ]

    def __iter__(self) -> Iterator[RegisteredTool]:
        for name, tool in self._tools.items():
            yield RegisteredTool(name=name, tool=tool)

    def invoke(self, invocation: ToolInvocation) -> ToolResponse:
        tool = self.get(invocation.tool_name)
        return tool.invoke(invocation)

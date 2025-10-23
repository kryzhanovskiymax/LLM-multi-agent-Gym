from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ToolInvocation:
    """Instruction to execute a tool."""

    tool_name: str
    arguments: Dict[str, Any]
    caller: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ToolMetadata:
    """Describes a tool's interface for discovery."""

    name: str
    description: str
    prototype: Dict[str, Any]
    response_schema: Dict[str, Any]


@dataclass(frozen=True)
class ToolResponse:
    """Represents the result returned by a tool execution."""

    tool_name: str
    output: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

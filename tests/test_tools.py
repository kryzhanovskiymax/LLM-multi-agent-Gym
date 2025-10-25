from __future__ import annotations

import pytest

from src.tools.base import BaseTool
from src.tools.tool_registry import DuplicateToolError, ToolRegistry, ToolNotFoundError
from src.tools.tools import ToolInvocation, ToolResponse


class ExampleTool(BaseTool):
    """Docstring description used for tool discovery."""

    @property
    def name(self) -> str:
        return "example"

    def prototype(self):
        return {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}

    def response_schema(self):
        return {"type": "object", "properties": {"doubled": {"type": "integer"}}}

    def invoke(self, invocation: ToolInvocation) -> ToolResponse:
        value = invocation.arguments["value"]
        return ToolResponse(tool_name=self.name, output={"doubled": value * 2})


def test_base_tool_description_defaults_to_docstring():
    tool = ExampleTool()
    assert tool.description == "Docstring description used for tool discovery."


def test_tool_registry_register_list_and_invoke():
    registry = ToolRegistry()
    tool = ExampleTool()
    registry.register(tool)

    metadata = registry.list_tools()
    assert metadata[0].name == "example"
    assert metadata[0].description == tool.description

    invocation = ToolInvocation(tool_name="example", arguments={"value": 3})
    response = registry.invoke(invocation)
    assert response.output == {"doubled": 6}


def test_tool_registry_duplicate_registration_raises():
    registry = ToolRegistry([ExampleTool()])
    with pytest.raises(DuplicateToolError):
        registry.register(ExampleTool())


def test_tool_registry_get_missing_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("missing")

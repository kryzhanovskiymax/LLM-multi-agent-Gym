from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.agent.base import Agent, AgentObservation, AgentStepOutput
from src.agent.network import AgenticNetwork, ToolExecutionMode
from src.environment.base import Environment, EnvironmentStepInput, EnvironmentStepResult
from src.environment.tool_executor import DefaultToolExecutor
from src.llm_client.base import LLMResult
from src.tools.base import BaseTool
from src.tools.tool_registry import ToolRegistry
from src.tools.tools import ToolInvocation, ToolResponse


class DoublingTool(BaseTool):
    """Simple tool that doubles numeric input."""

    @property
    def name(self) -> str:
        return "double"

    def prototype(self):
        return {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]}

    def response_schema(self):
        return {"type": "object", "properties": {"result": {"type": "number"}}}

    def invoke(self, invocation: ToolInvocation) -> ToolResponse:
        value = invocation.arguments["value"]
        return ToolResponse(tool_name=self.name, output={"result": value * 2})


class RecordingAgent(Agent):
    """Test agent that records interactions."""

    def __init__(self, name: str, llm_client, tool_registry: ToolRegistry):
        super().__init__(name=name, llm_client=llm_client, tool_registry=tool_registry)
        self.observations: List[AgentObservation] = []
        self.tool_results: List[ToolResponse] = []
        self._last_observation: AgentObservation | None = None

    def handle_observation(self, observation: AgentObservation) -> None:
        self.observations.append(observation)
        self._last_observation = observation

    def step(self) -> AgentStepOutput:
        if not self._last_observation:
            return AgentStepOutput()
        prompt = self._last_observation.payload.get("prompt", "")
        llm_result: LLMResult = self.llm_client.complete(prompt)
        invocation = ToolInvocation(tool_name="double", arguments={"value": len(llm_result.text)}, caller=self.name)
        return AgentStepOutput(
            tool_invocations=[invocation],
            environment_actions={"message": llm_result.text},
        )

    def handle_tool_result(self, response: ToolResponse) -> None:
        self.tool_results.append(response)


class RecordingEnvironment(Environment):
    """Environment harness that records the last step input."""

    def __init__(self, *, tool_executor=None, prompt: str = "hello"):
        super().__init__(tool_executor=tool_executor)
        self.prompt = prompt
        self.agents: List[str] = []
        self.last_step_input: EnvironmentStepInput | None = None

    def validate_agents(self, agent_ids):
        self.agents = list(agent_ids)

    def reset(self) -> EnvironmentStepResult:
        observations = {
            agent_id: AgentObservation(source="environment", payload={"prompt": self.prompt})
            for agent_id in self.agents
        }
        return EnvironmentStepResult(observations=observations, terminated=False)

    def step(self, step_input: EnvironmentStepInput) -> EnvironmentStepResult:
        self.last_step_input = step_input
        observations = {
            agent_id: AgentObservation(source="environment", payload={"prompt": self.prompt})
            for agent_id in self.agents
        }
        return EnvironmentStepResult(observations=observations, terminated=False)


class OfflineEnvironment(RecordingEnvironment):
    """Environment that executes deferred tool calls and returns responses."""

    def step(self, step_input: EnvironmentStepInput) -> EnvironmentStepResult:
        self.last_step_input = step_input
        tool_responses: Dict[str, List[ToolResponse]] | None = None
        if step_input.tool_invocations and self.tool_executor:
            tool_responses = {
                agent_id: [self.tool_executor.execute(invocation) for invocation in invocations]
                for agent_id, invocations in step_input.tool_invocations.items()
            }
        observations = {
            agent_id: AgentObservation(source="environment", payload={"prompt": self.prompt})
            for agent_id in self.agents
        }
        return EnvironmentStepResult(
            observations=observations,
            terminated=False,
            tool_responses=tool_responses,
        )


@pytest.fixture
def tool_registry():
    return ToolRegistry([DoublingTool()])


def build_agent_and_network(environment: Environment, tool_registry: ToolRegistry, llm_text: str, mode: ToolExecutionMode):
    llm_mock = MagicMock()
    llm_mock.complete.return_value = LLMResult(text=llm_text, raw={})

    agent = RecordingAgent(name="agent-1", llm_client=llm_mock, tool_registry=tool_registry)
    network = AgenticNetwork(environment, tool_execution_mode=mode)
    network.register_agent(agent)
    return agent, network, llm_mock


def test_streaming_mode_executes_tools_immediately(tool_registry):
    tool_executor = DefaultToolExecutor(tool_registry)
    environment = RecordingEnvironment(tool_executor=tool_executor, prompt="stream")
    agent, network, llm_mock = build_agent_and_network(
        environment, tool_registry, llm_text="stream-response", mode=ToolExecutionMode.STREAMING
    )

    network.reset()
    result = network.step()

    llm_mock.complete.assert_called_once_with("stream")
    assert environment.last_step_input is not None
    assert environment.last_step_input.tool_invocations == {}
    assert agent.tool_results  # streaming should deliver tool output immediately
    assert agent.tool_results[0].output == {"result": len("stream-response") * 2}
    assert not result.tool_responses  # environment didn't have to return responses
    assert agent.context is not None


def test_offline_mode_batches_tool_calls_and_dispatches_results(tool_registry):
    tool_executor = DefaultToolExecutor(tool_registry)
    environment = OfflineEnvironment(tool_executor=tool_executor, prompt="offline")
    agent, network, llm_mock = build_agent_and_network(
        environment, tool_registry, llm_text="offline-response", mode=ToolExecutionMode.OFFLINE
    )

    network.reset()
    result = network.step()

    llm_mock.complete.assert_called_once_with("offline")
    assert environment.last_step_input is not None
    assert "agent-1" in environment.last_step_input.tool_invocations
    # Environment should return tool responses that the agent later receives
    assert result.tool_responses is not None
    assert "agent-1" in result.tool_responses
    assert agent.tool_results  # dispatched after environment step
    assert agent.tool_results[0].output == {"result": len("offline-response") * 2}


def test_streaming_mode_without_executor_skips_tool_execution(tool_registry):
    environment = RecordingEnvironment(tool_executor=None, prompt="noop")
    agent, network, llm_mock = build_agent_and_network(
        environment, tool_registry, llm_text="noop-response", mode=ToolExecutionMode.STREAMING
    )

    network.reset()
    network.step()

    llm_mock.complete.assert_called_once_with("noop")
    assert environment.last_step_input is not None
    assert not agent.tool_results  # no executor, so tool responses never delivered

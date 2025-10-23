from __future__ import annotations

from typing import Dict, Optional

from agent.base import Agent, AgentObservation, AgentStepOutput
from agent.network import AgenticNetwork
from environment.base import Environment, EnvironmentStepResult
from environment.tool_executor import DefaultToolExecutor, ToolExecutor
from llm_client.base import LLMClient, LLMMessage, LLMResult
from tools.base import BaseTool
from tools.tool_registry import ToolRegistry
from tools.tools import ToolInvocation, ToolResponse


# ----- LLM CLIENT -----------------------------------------------------------------


class DummyLLMClient(LLMClient):
    """Minimal LLM client that mirrors the prompt."""

    def complete(self, prompt: str, **kwargs) -> LLMResult:
        return LLMResult(text=f"[echo] {prompt}", raw={"prompt": prompt})

    def chat(self, messages, **kwargs) -> LLMResult:
        last = messages[-1].content if messages else ""
        return LLMResult(text=f"[chat-echo] {last}", raw={"messages": [m.content for m in messages]})


# ----- TOOLS ----------------------------------------------------------------------


class EchoTool(BaseTool):
    """Simple tool that echoes text back to the caller."""

    @property
    def name(self) -> str:
        return "echo"

    def prototype(self):
        return {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo back."}},
            "required": ["text"],
        }

    def response_schema(self):
        return {
            "type": "object",
            "properties": {"echo": {"type": "string"}, "length": {"type": "integer"}},
        }

    def invoke(self, invocation: ToolInvocation) -> ToolResponse:
        text = invocation.arguments.get("text", "")
        response = {"echo": text, "length": len(text)}
        return ToolResponse(tool_name=self.name, output=response, metadata={"caller": invocation.caller})


# ----- ENVIRONMENT ----------------------------------------------------------------


class SimpleEnvironment(Environment):
    """Toy environment looping messages back to agents."""

    def __init__(self, *, tool_executor: ToolExecutor, max_steps: int = 3) -> None:
        super().__init__(tool_executor=tool_executor)
        self._max_steps = max_steps
        self._step_counter = 0
        self._agent_ids: Dict[str, None] = {}

    def validate_agents(self, agent_ids):
        self._agent_ids = {agent_id: None for agent_id in agent_ids}

    def reset(self) -> EnvironmentStepResult:
        self._step_counter = 0
        observations = {
            agent_id: AgentObservation(
                source="environment", payload={"message": "welcome"}, metadata={"step": self._step_counter}
            )
            for agent_id in self._agent_ids
        }
        return EnvironmentStepResult(observations=observations, terminated=False)

    def step(self, actions: Dict[str, Dict[str, str]]) -> EnvironmentStepResult:
        self._step_counter += 1
        observations = {}
        for agent_id in self._agent_ids:
            message = actions.get(agent_id, {}).get("message", "no-action")
            observations[agent_id] = AgentObservation(
                source="environment", payload={"message": message}, metadata={"step": self._step_counter}
            )
        terminated = self._step_counter >= self._max_steps
        return EnvironmentStepResult(observations=observations, terminated=terminated)


# ----- AGENT ----------------------------------------------------------------------


class EchoAgent(Agent):
    """Agent that routes observations through the echo tool."""

    def __init__(self, name: str, llm_client: LLMClient, tool_registry: ToolRegistry) -> None:
        super().__init__(name=name, llm_client=llm_client, tool_registry=tool_registry)
        self._last_observation: Optional[AgentObservation] = None

    def handle_observation(self, observation: AgentObservation) -> None:
        self._last_observation = observation

    def step(self) -> AgentStepOutput:
        if not self._last_observation:
            return AgentStepOutput()

        latest_message = self._last_observation.payload.get("message", "")
        llm_result = self.llm_client.complete(latest_message)

        invocation = ToolInvocation(tool_name="echo", arguments={"text": llm_result.text}, caller=self.name)
        broadcast_message = LLMMessage(role="assistant", content=llm_result.text)

        return AgentStepOutput(
            responses=[broadcast_message],
            tool_invocations=[invocation],
            environment_actions={"message": llm_result.text},
        )

    def handle_tool_result(self, response: ToolResponse) -> None:
        if self.logger:
            self.logger.info("Tool %s responded: %s", response.tool_name, response.output)


# ----- DEMO LOOP ------------------------------------------------------------------


def build_demo_network() -> AgenticNetwork:
    env_registry = ToolRegistry([EchoTool()])
    agent_registry = ToolRegistry([EchoTool()])
    tool_executor = DefaultToolExecutor(env_registry)
    environment = SimpleEnvironment(tool_executor=tool_executor)
    network = AgenticNetwork(environment)

    agent = EchoAgent(name="agent-1", llm_client=DummyLLMClient(), tool_registry=agent_registry)
    network.register_agent(agent)
    return network


def main() -> None:
    network = build_demo_network()
    result = network.reset()
    print("Initial observations:", {k: v.payload for k, v in result.observations.items()})

    while True:
        result = network.step()
        print("Environment step ->", {k: v.payload for k, v in result.observations.items()})
        if result.terminated:
            break


if __name__ == "__main__":
    main()

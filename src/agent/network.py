from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, Optional

from ..environment.base import Environment, EnvironmentStepResult
from ..environment.tool_executor import ToolExecutor
from .base import (
    Agent,
    AgentContext,
    AgentMessage,
    AgentObservation,
    AgentStepOutput,
)


class AgenticNetwork:
    """Coordinates agent-to-agent communication and environment stepping."""

    def __init__(self, environment: Environment) -> None:
        self._environment = environment
        self._agents: Dict[str, Agent] = {}
        self._message_queue: Deque[AgentMessage] = deque()

    @property
    def environment(self) -> Environment:
        return self._environment

    @property
    def tool_executor(self) -> Optional[ToolExecutor]:
        return self._environment.tool_executor

    def register_agent(self, agent: Agent) -> None:
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered.")
        context = AgentContext(environment=self.environment, network=self)
        agent.attach_context(context)
        self._agents[agent.name] = agent
        if self.environment:
            self.environment.validate_agents(list(self._agents.keys()))

    def unregister_agent(self, agent_name: str) -> None:
        self._agents.pop(agent_name, None)

    def reset(self) -> EnvironmentStepResult:
        for agent in self._agents.values():
            agent.reset()
        result = self.environment.reset()
        self._message_queue.clear()
        self._dispatch_observations(result.observations)
        return result

    def step(self) -> EnvironmentStepResult:
        actions: Dict[str, AgentStepOutput] = {}
        for name, agent in self._agents.items():
            output = agent.step()
            actions[name] = output
            self._handle_tool_invocations(agent, output)
            self._enqueue_messages(output.broadcast_messages)
        self._deliver_messages()
        env_actions = {
            agent_id: step_output.environment_actions
            for agent_id, step_output in actions.items()
            if step_output.environment_actions is not None
        }
        result = self.environment.step(env_actions)
        self._dispatch_observations(result.observations)
        return result

    def _handle_tool_invocations(self, agent: Agent, output: AgentStepOutput) -> None:
        if not output.tool_invocations or not self.tool_executor:
            return
        for invocation in output.tool_invocations:
            response = self.tool_executor.execute(invocation)
            agent.handle_tool_result(response)

    def _enqueue_messages(self, messages: Iterable[AgentMessage]) -> None:
        for message in messages:
            self._message_queue.append(message)

    def _deliver_messages(self) -> None:
        while self._message_queue:
            message = self._message_queue.popleft()
            if message.recipient:
                recipient = self._agents.get(message.recipient)
                if recipient:
                    recipient.handle_message(message)
                continue
            for agent in self._agents.values():
                if agent.name != message.sender:
                    agent.handle_message(message)

    def _dispatch_observations(self, observations: Dict[str, AgentObservation]) -> None:
        for name, observation in observations.items():
            agent = self._agents.get(name)
            if agent:
                agent.handle_observation(observation)

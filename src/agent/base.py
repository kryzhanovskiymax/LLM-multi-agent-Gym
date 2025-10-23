from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING

from ..llm_client.base import LLMClient, LLMMessage
from ..tools.tool_registry import ToolRegistry
from ..tools.tools import ToolInvocation, ToolResponse

if TYPE_CHECKING:  # pragma: no cover - hint only
    from ..environment.base import Environment
    from .network import AgenticNetwork


@dataclass
class AgentObservation:
    """Represents an observation delivered to an agent."""

    source: str
    payload: Dict[str, object]
    metadata: Optional[Dict[str, object]] = None


@dataclass
class AgentMessage:
    """Message exchanged between agents via the agentic network."""

    sender: str
    recipient: Optional[str]
    content: str
    metadata: Optional[Dict[str, object]] = None


@dataclass
class AgentStepOutput:
    """Actions an agent wants to perform during a step."""

    responses: Sequence[LLMMessage] = field(default_factory=list)
    tool_invocations: Sequence[ToolInvocation] = field(default_factory=list)
    broadcast_messages: Sequence[AgentMessage] = field(default_factory=list)
    terminated: bool = False
    environment_actions: Optional[Dict[str, object]] = None


@dataclass
class AgentContext:
    """Injected interfaces to operate in an environment."""

    environment: Optional["Environment"] = None
    network: Optional["AgenticNetwork"] = None
    metadata: Optional[Dict[str, object]] = None


class Agent(ABC):
    """Base class for all Agents."""

    def __init__(
        self,
        name: str,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        *,
        logger=None,
    ) -> None:
        self._name = name
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._logger = logger
        self._context: Optional[AgentContext] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def llm_client(self):
        return self._llm_client

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def logger(self):
        return self._logger

    @property
    def context(self) -> Optional[AgentContext]:
        return self._context

    def attach_context(self, context: AgentContext) -> None:
        self._context = context
        self.on_context_attached(context)

    def on_context_attached(self, context: AgentContext) -> None:
        """Hook executed after the agent receives a new runtime context."""

    def reset(self) -> None:
        """Reset the agent's internal state between episodes."""

    @abstractmethod
    def handle_observation(self, observation: AgentObservation) -> None:
        """Consume an observation emitted by the environment or another agent."""

    @abstractmethod
    def step(self) -> AgentStepOutput:
        """Produce actions, tool calls, and messages for this time step."""

    def handle_tool_result(self, response: ToolResponse) -> None:
        """Hook to receive tool execution results."""

    def handle_message(self, message: AgentMessage) -> None:
        """Hook to receive messages from other agents."""

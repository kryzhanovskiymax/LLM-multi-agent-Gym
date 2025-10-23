from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from ..agent.base import AgentObservation

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - hint only
    from .tool_executor import ToolExecutor


@dataclass
class EnvironmentStepResult:
    """Information returned by the environment after executing a step."""

    observations: Dict[str, AgentObservation]
    terminated: bool
    rewards: Optional[Dict[str, float]] = None
    info: Optional[Dict[str, object]] = None


class Environment(ABC):
    """Base interface for environments used inside the gym."""

    def __init__(self, *, tool_executor: Optional["ToolExecutor"] = None) -> None:
        self._tool_executor = tool_executor

    @property
    def tool_executor(self) -> Optional["ToolExecutor"]:
        return self._tool_executor

    def bind_tool_executor(self, tool_executor: "ToolExecutor") -> None:
        self._tool_executor = tool_executor

    @abstractmethod
    def reset(self) -> EnvironmentStepResult:
        """Reset the environment to an initial state and return observations."""

    @abstractmethod
    def step(self, actions: Dict[str, object]) -> EnvironmentStepResult:
        """
        Apply actions emitted by the agents and return the next environment state.
        Implementations decide the structure of the action objects.
        """

    def validate_agents(self, agent_ids: Sequence[str]) -> None:
        """Optional hook to validate agent registration against the environment."""
        pass

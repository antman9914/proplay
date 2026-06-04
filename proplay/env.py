"""
BaseEnv: common interface for all environments in the demo.
Every environment (ALFWorld, TextCraft, HotpotQA) inherits from this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool


class BaseEnv(ABC):
    """Minimal interface required by all agents."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, idx: int) -> str:
        """Reset to task `idx`.  Returns the initial observation."""

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Execute `action` and return (observation, reward, done)."""

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def observe(self) -> str:
        """Current observation (same as last StepResult.observation)."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the episode has ended."""

    @property
    @abstractmethod
    def reward(self) -> float:
        """Cumulative episode reward (or last-step reward for sparse envs)."""

    @property
    def available_actions(self) -> list[str]:
        """
        List of valid action strings for the current state.
        Return an empty list if the action space is open (e.g. HotpotQA).
        """
        return []

    @property
    @abstractmethod
    def task_description(self) -> str:
        """Human-readable description of the current task."""

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Short identifier, e.g. 'alfworld', 'textcraft', 'hotpotqa'."""

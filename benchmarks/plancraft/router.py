"""
PlanCraft environment wrapper.

Wraps the plancraft EnvWrapper with a simplified interface matching our
pipeline conventions. No action grounding — plancraft actions are
structured strings that the environment parses directly.

Slots:
  [0]          output slot of crafting table (source only)
  [A1]-[C3]    3×3 crafting grid
  [I1]-[I36]   personal inventory
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# plancraft is installed as a pip package — no sys.path manipulation needed here.
# (Adding the parent directory would shadow the installed plancraft package.)

try:
    from plancraft.simple import EnvWrapper
    from plancraft.config import PlancraftExample
    _PLANCRAFT_AVAILABLE = True
except ImportError:
    _PLANCRAFT_AVAILABLE = False


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    success: bool


class PlancraftEnv:
    """
    Wraps a single PlancraftExample episode.

    Usage:
        env = PlancraftEnv(data_path="splits/merged_227.json")
        env.reset(episode_idx)
        obs = env.observe            # text observation string
        result = env.step(action)   # StepResult
        env.done, env.reward, env.success
    """

    def __init__(self, data_path: str, max_steps: int = 30) -> None:
        self.max_steps = max_steps
        self._data_path = data_path
        self._examples: list[dict] = self._load(data_path)
        self._env: Optional[EnvWrapper] = None
        self._obs: str = ""
        self._reward: float = 0.0
        self._done: bool = False
        self._success: bool = False
        self._current_example: Optional[dict] = None

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def observe(self) -> str:
        return self._obs

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._done

    @property
    def success(self) -> bool:
        return self._success

    @property
    def task_description(self) -> str:
        if self._current_example is None:
            return ""
        return f"Craft an item of type: {self._current_example['target']}"

    @property
    def target(self) -> str:
        if self._current_example is None:
            return ""
        return self._current_example.get("target", "")

    def reset(self, episode_idx: int) -> str:
        if episode_idx >= len(self._examples):
            raise IndexError(f"Episode index {episode_idx} out of range ({len(self._examples)} total)")
        ex_data = self._examples[episode_idx]
        self._current_example = ex_data
        self._reward = 0.0
        self._done = False
        self._success = False

        if _PLANCRAFT_AVAILABLE:
            example = PlancraftExample(**ex_data)
            self._env = EnvWrapper(
                example=example,
                max_steps=self.max_steps,
                resolution="high",
                use_text_inventory=True,
            )
            # Initial observation
            self._obs = _build_initial_obs(example)
        else:
            self._env = None
            self._obs = _build_initial_obs_from_dict(ex_data)

        return self._obs

    def step(self, action: str) -> StepResult:
        if self._done:
            return StepResult(
                observation="Episode already terminated.",
                reward=self._reward,
                done=True,
                success=self._success,
            )

        if _PLANCRAFT_AVAILABLE and self._env is not None:
            obs_dict, reward, terminated = self._env.step(action)
            obs_text = obs_dict.get("text", "")
            # Remove 'impossible' from the invalid-action error hint so the agent
            # never learns it's a valid choice (all tasks in our eval are possible).
            obs_text = obs_text.replace(", impossible", "").replace("impossible, ", "").replace("impossible", "")
            self._obs = obs_text
            self._reward = reward
            self._done = terminated
            self._success = self._env.success
        else:
            # Fallback for when plancraft is not installed
            obs_text, reward, terminated = self._step_mock(action)
            self._obs = obs_text
            self._reward = reward
            self._done = terminated

        return StepResult(
            observation=self._obs,
            reward=self._reward,
            done=self._done,
            success=self._success,
        )

    def num_episodes(self) -> int:
        return len(self._examples)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: str) -> list[dict]:
        with open(path) as f:
            return json.load(f)

    def _step_mock(self, action: str) -> tuple[str, float, bool]:
        """Minimal mock for when plancraft is not installed — for testing only."""
        return self._obs, 0.0, False


# ── Observation helpers ───────────────────────────────────────────────────────

def _build_initial_obs(example) -> str:
    """Build initial text observation from PlancraftExample object."""
    from plancraft.environment.env import target_and_inventory_to_text_obs, convert_from_slot_index
    # slotted_inventory has int keys
    inv_dict = {}
    for slot_key, item in example.slotted_inventory.items():
        idx = int(slot_key)
        inv_dict[idx] = item
    return target_and_inventory_to_text_obs(
        target=example.target, inventory=inv_dict
    )


def _build_initial_obs_from_dict(ex_data: dict) -> str:
    """Build initial text observation from raw dict (fallback)."""
    target = ex_data.get("target", "unknown")
    inventory = ex_data.get("slotted_inventory", {})
    lines = [f"Craft an item of type: {target}", "inventory:"]
    for slot_key in sorted(inventory.keys(), key=lambda k: int(k)):
        item = inventory[slot_key]
        qty = item.get("quantity", 1)
        if qty > 0:
            slot_label = _convert_slot(int(slot_key))
            lines.append(f" - {item['type']} {slot_label} quantity {qty}")
    return "\n".join(lines)


def _convert_slot(idx: int) -> str:
    grid = {0: "[0]", 1: "[A1]", 2: "[A2]", 3: "[A3]",
            4: "[B1]", 5: "[B2]", 6: "[B3]",
            7: "[C1]", 8: "[C2]", 9: "[C3]"}
    if idx in grid:
        return grid[idx]
    if 10 <= idx <= 45:
        return f"[I{idx-9}]"
    return f"[I{idx}]"

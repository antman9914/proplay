"""
Base agent for PlanCraft.

PlanCraft actions are structured strings (no grounding needed).
No available_actions list is provided by the environment.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

HISTORY_WINDOW = 20


def _parse_think_action(response: str) -> tuple[str, str]:
    """Extract (think, action) from <think>...</think><action>...</action>."""
    think = ""
    think_m = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_m:
        think = think_m.group(1).strip()

    action_m = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
    if action_m:
        return think, action_m.group(1).strip()

    # Fallback: first non-empty non-tag line
    for line in response.strip().splitlines():
        line = line.strip()
        if line and not line.startswith("<"):
            return think, line
    return think, response.strip()


class BaseAgent:

    AGENT_TYPE = "base"

    def __init__(self, llm) -> None:
        self.llm = llm
        self._trajectory: list[dict] = []
        self._step: int = 0

    def reset(self) -> None:
        self._trajectory = []
        self._step = 0

    def get_action(self, obs: str, task: str) -> str:
        raise NotImplementedError

    def finalize_episode(self, success: bool, task: str, reward: float) -> None:
        pass

    # ── Trajectory helpers ────────────────────────────────────────────────────

    def _record_step(
        self,
        obs: str,
        action: str,
        raw_action: str = "",
        think: str = "",
    ) -> None:
        self._trajectory.append({
            "obs": obs,
            "action": action,
            "raw_action": raw_action or action,
            "think": think,
        })
        self._step += 1

    def _recent_history(self) -> list[dict]:
        return self._trajectory[-HISTORY_WINDOW:]

    def _format_history(self) -> str:
        history = self._recent_history()
        if not history:
            return "(start of episode)"
        lines = []
        for i, t in enumerate(history):
            step_num = self._step - len(history) + i + 1
            lines.append(f"Step {step_num}: {t['raw_action'] or t['action']}")
            if t.get("result_obs"):
                short = t["result_obs"][:200]
                lines.append(f"  → {short}")
        return "\n".join(lines)

    def _compact_trajectory(self) -> str:
        """Return a compact text summary of the episode trajectory."""
        lines = []
        for i, t in enumerate(self._trajectory, 1):
            lines.append(f"Step {i}: {t.get('raw_action') or t['action']}")
            result = t.get("result_obs", "")
            if result:
                lines.append(f"  Obs: {result[:200]}")
        return "\n".join(lines) if lines else "(empty)"

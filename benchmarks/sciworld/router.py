"""
ScienceWorld environment wrapper.
Communicates with the AgentGym ScienceWorld REST-API server.

Start the server first:
  python -m uvicorn agentenv_sciworld.server:app --host 0.0.0.0 --port 36006

Then pass env_server_base="http://localhost:<port>" to SciWorldEnv.

Note: available_actions always returns [] — the LLM's raw output is passed
directly to the environment without grounding against a pre-fetched action list.
The /action_hint endpoint only exposes partial action-object combinations that
are not state-specific valid actions, so grounding against them reduces quality.
"""
from __future__ import annotations

import requests

from proplay.env import BaseEnv, StepResult


class SciWorldEnv(BaseEnv):
    """
    Thin HTTP wrapper around the AgentGym ScienceWorld FastAPI server.

    The server exposes:
      POST /create  → {id}
      POST /reset   → {id, task_name, var_num, task_description,
                        observation, reward, score, done}
      POST /step    → {observation, reward, score, done}
    """

    ENV_NAME = "sciworld"

    def __init__(
        self,
        env_server_base: str,
        timeout: int = 60,
    ):
        self.base = env_server_base.rstrip("/")
        self.timeout = timeout

        resp = requests.post(f"{self.base}/create", timeout=self.timeout)
        resp.raise_for_status()
        self._env_id: int = resp.json()["id"]

        # Episode state
        self._obs: str = ""
        self._task_desc: str = ""
        self._task_name: str = ""
        self._done: bool = False
        self._reward: float = 0.0
        self._score: float = 0.0

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self, idx: int) -> str:
        payload = {"id": self._env_id, "data_idx": idx}
        resp = requests.post(f"{self.base}/reset", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        self._done = bool(data.get("done", False))
        self._reward = float(data.get("score", 0.0)) / 100.0  # server returns 0–100
        self._score = self._reward
        self._task_name = data.get("task_name", "")
        self._task_desc = data.get("task_description", "")
        self._obs = data.get("observation", "").strip()

        return self._obs

    def step(self, action: str) -> StepResult:
        if self._done:
            return StepResult(self._obs, self._reward, True)

        action = self._clean_action(action)

        payload = {"id": self._env_id, "action": action}
        resp = requests.post(f"{self.base}/step", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        self._reward = float(data.get("score", 0.0)) / 100.0  # server returns 0–100
        self._score = self._reward
        self._done = bool(data.get("done", False))
        self._obs = data.get("observation", "").strip()

        return StepResult(self._obs, self._reward, self._done)

    @property
    def observe(self) -> str:
        return self._obs

    @property
    def done(self) -> bool:
        return self._done

    @property
    def reward(self) -> float:
        return self._score

    @property
    def available_actions(self) -> list[str]:
        return []

    @property
    def task_description(self) -> str:
        return self._task_desc or self._task_name

    @property
    def env_name(self) -> str:
        return self.ENV_NAME

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_action(self, action: str) -> str:
        """Strip LLM preamble and normalize whitespace."""
        for prefix in ("Action:", "## Immediate Action", "**Action**:"):
            if prefix.lower() in action.lower():
                idx = action.lower().index(prefix.lower()) + len(prefix)
                action = action[idx:].strip()
                break
        return action.split("\n")[0].strip().strip("`").strip()

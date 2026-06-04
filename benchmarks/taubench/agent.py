"""
AWM Agent for τ-Bench — tool-graph preplay variant.

Changes vs the original sciworld-style version:
  - Workflow induction removed (run_induction, workflow_path, _episode_summaries,
    _sync_graph all gone).  Tools are already at workflow granularity in τ-Bench,
    so inducting a higher-level library adds noise.
  - WorkflowGraph is now initialised from actual tool names (load_or_create_from_tools)
    instead of from a workflow text file.  Nodes = tool names; edges = transition
    counts with reliability scores, exactly as before.
  - run_preplay / record_execution / format_plan_for_injection are preserved verbatim.
    The only guard added: _parse_plan now validates plan entries against the known
    tool list so hallucinated names are dropped before injection.
  - finalize_episode is simplified: no induction trace, no re-resolution dance —
    plan_ids are already stable (tool names never change).

Everything else (graph persistence, similarity-weighted edge reliability, failed-
experience injection, plan injection into system prompt) is identical to the
SciWorld version.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# memory_planning_demo deps moved into proplay package

from prompts import (
    SYSTEM_BASE,
    format_episode_summary,
    extract_failed_steps_summary,
)
from workflow_graph import WorkflowGraph
from preplay import run_preplay, format_plan_for_injection, record_execution, record_execution_from_trajectory

logger = logging.getLogger(__name__)

PLAN_MATCH_THRESHOLD = 0.50

_EXPERIENCE_SIM_THRESHOLD = 0.75


# ── Sentence-embedding singleton ──────────────────────────────────────────────

class _EmbeddingGrounder:
    _instance: Optional["_EmbeddingGrounder"] = None

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._np = np
        logger.info("Loading embedding model...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._cache: dict[str, "np.ndarray"] = {}
        logger.info("Embedding model loaded.")

    @classmethod
    def get(cls) -> "_EmbeddingGrounder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _encode(self, texts: list[str]) -> "np.ndarray":
        new = [t for t in texts if t not in self._cache]
        if new:
            embs = self._model.encode(new, normalize_embeddings=True, show_progress_bar=False)
            for t, e in zip(new, embs):
                self._cache[t] = e
        return self._np.stack([self._cache[t] for t in texts])


def _retrieve_similar_experiences(
    query_emb: Optional[list],
    experiences: list[tuple[list, str]],
    threshold: float = _EXPERIENCE_SIM_THRESHOLD,
) -> list[str]:
    if not query_emb:
        return [summary for _, summary in experiences]
    scored: list[tuple[float, str]] = []
    for emb, summary in experiences:
        if not emb:
            continue
        sim = sum(q * e for q, e in zip(query_emb, emb))
        if sim >= threshold:
            scored.append((sim, summary))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [summary for _, summary in scored]


class AWMAgent:

    AGENT_TYPE = "awm_taubench"

    def __init__(
        self,
        llm,
        preplay_graph_path: str,
        site: str,                    # "retail" or "airline"
        use_preplay: bool = True,
        use_reliability: bool = True,
        record_mode: str = "plan",    # "plan" | "trajectory"
    ) -> None:
        self.llm                = llm
        self.preplay_graph_path = preplay_graph_path
        self.site               = site
        self.use_preplay        = use_preplay
        self.use_reliability    = use_reliability
        self.record_mode        = record_mode  # which recorder to use after each episode

        # ── Cross-episode state ───────────────────────────────────────────────
        self._preplay_graph:      Optional[WorkflowGraph] = None
        self._preplay_site:       str = site
        self._failed_experiences: list[tuple[list, str]] = []

        _EmbeddingGrounder.get()

        # ── Episode-local state ───────────────────────────────────────────────
        self._task:              str = ""
        self._trajectory:        list[dict] = []
        self._step:              int = 0
        self._graph_initialized: bool = False
        self._preplay_plan:      list[str] = []
        self._preplay_plan_ids:  list[str] = []
        self._preplay_plan_text: str = ""
        self._task_embedding:    Optional[list] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self) -> None:
        self._task              = ""
        self._trajectory        = []
        self._step              = 0
        self._graph_initialized = False
        self._preplay_plan      = []
        self._preplay_plan_ids  = []
        self._preplay_plan_text = ""
        self._task_embedding    = None

    def solve(self, env, task_index: int, max_steps: int = 30) -> tuple[float, list]:
        self.reset()

        try:
            reset_resp  = env.reset(task_index=task_index)
            initial_obs = self._extract_obs(reset_resp)
        except Exception as exc:
            logger.error("env.reset(task_index=%d) failed: %s", task_index, exc)
            return 0.0, []

        self._task = initial_obs
        logger.info("AWMAgent: task_index=%d  task=%r", task_index, initial_obs[:200])

        if self.preplay_graph_path and not self._graph_initialized:
            self._init_graph_and_preplay(initial_obs, env)

        sys_msg  = SYSTEM_BASE
        wf_section = self._get_plan_section()
        if wf_section:
            sys_msg += f"\n\n{wf_section}"

        tools    = getattr(env, "tools_info", []) or []
        messages: list[dict] = [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": initial_obs},
        ]

        done       = False
        step_count = 0

        while not done and step_count < max_steps:
            try:
                content, tool_calls = self.llm.chat_with_tools(messages, tools=tools)
            except Exception as exc:
                logger.error("LLM call failed at step %d: %s", step_count, exc)
                break

            if not tool_calls:
                if content:
                    done, step_count = self._execute_respond(
                        env, messages, content, step_count
                    )
                else:
                    break
                continue

            assistant_msg: dict = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in tool_calls:
                if done:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "(episode already ended)",
                    })
                    continue

                tool_name   = tc["name"]
                tool_kwargs = tc["arguments"]
                tool_id     = tc["id"]

                if tool_name == "think":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": "",
                    })
                    continue

                try:
                    from tau_bench.types import Action
                    step_resp = env.step(Action(name=tool_name, kwargs=tool_kwargs))
                    obs       = self._extract_obs(step_resp)
                    is_done   = getattr(step_resp, "done", False)

                    self._record_step(tool_name, tool_kwargs, obs)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": obs,
                    })

                    if is_done:
                        done = True
                    step_count += 1
                    logger.info("  step=%d  tool=%s  done=%s", step_count, tool_name, done)

                except Exception as exc:
                    logger.warning("Tool %s failed: %s", tool_name, exc)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": f"Error: {exc}",
                    })
                    step_count += 1

        try:
            reward_result = env.calculate_reward()
            reward = float(getattr(reward_result, "reward", reward_result))
        except Exception as exc:
            logger.warning("calculate_reward failed: %s", exc)
            reward = 0.0

        if self._trajectory:
            self._trajectory[-1]["reward"] = reward

        logger.info("AWMAgent: episode done  steps=%d  reward=%.1f", step_count, reward)
        return reward, self._trajectory

    def finalize_episode(self, success: bool, task: str, reward: float = 0.0) -> None:
        """
        Post-episode processing: record failed experience → record_execution → save.

        Workflow induction and _sync_graph are removed — tools are stable and
        already at the right abstraction level.  The preplay plan (list of tool
        names) is used directly; plan_ids were resolved at preplay time and are
        stable because tool names never change.
        """
        if self._trajectory:
            summary = format_episode_summary(task, self._trajectory)

            if extract_failed_steps_summary(summary):
                self._failed_experiences.append((self._task_embedding or [], summary))
                logger.info("AWMAgent: recorded failed experience")

        if self.preplay_graph_path and self._preplay_graph is not None:
            plan      = self._preplay_plan
            plan_ids  = self._preplay_plan_ids

            if self.record_mode == "trajectory":
                if self._trajectory:
                    record_execution_from_trajectory(
                        trajectory=self._trajectory,
                        graph=self._preplay_graph,
                        site=self._preplay_site,
                        reward=reward,
                        task_embedding=self._task_embedding,
                    )
            else:  # "plan" (default)
                if plan and len(plan) >= 2:
                    record_execution(
                        plan=plan,
                        graph=self._preplay_graph,
                        site=self._preplay_site,
                        reward=reward,
                        plan_ids=plan_ids,
                        task_embedding=self._task_embedding,
                    )

            self._preplay_graph.save(self.preplay_graph_path)
            logger.info(
                "AWMAgent: graph saved  nodes=%d  edges=%d",
                len(self._preplay_graph.nodes),
                len(self._preplay_graph.edges),
            )

    # ── Graph / preplay ───────────────────────────────────────────────────────

    def _init_graph_and_preplay(self, task: str, env) -> None:
        tool_names = [
            t["function"]["name"]
            for t in (getattr(env, "tools_info", []) or [])
            if t.get("type") == "function"
        ]

        self._preplay_graph = WorkflowGraph.load_or_create_from_tools(
            path=self.preplay_graph_path,
            tool_names=tool_names,
            site=self._preplay_site,
        )
        self._graph_initialized = True
        self._task_embedding = _EmbeddingGrounder.get()._encode([task])[0].tolist()

        if self.use_preplay:
            relevant_experiences = _retrieve_similar_experiences(
                self._task_embedding, self._failed_experiences,
            )
            logger.info(
                "AWMAgent: preplay experience retrieval  total=%d  retrieved=%d",
                len(self._failed_experiences), len(relevant_experiences),
            )
            self._preplay_plan, self._preplay_plan_text = run_preplay(
                llm_chat_fn=self._llm_chat,
                goal=task,
                site=self._preplay_site,
                graph=self._preplay_graph,
                use_reliability=self.use_reliability,
                workflow_content=None,   # tool-graph mode: no workflow content
                task_embedding=self._task_embedding,
                failed_experiences=relevant_experiences,
                valid_tool_names=tool_names,
            )
            # Resolve plan entries to stable node IDs
            name_to_id = {
                n.name.lower().strip(): n.node_id
                for n in self._preplay_graph.get_nodes_for_site(self._preplay_site)
            }
            self._preplay_plan_ids = [
                name_to_id.get(p.strip().lower(), "") for p in self._preplay_plan
            ]
            logger.info("AWMAgent: preplay plan     = %s", self._preplay_plan)
            logger.info("AWMAgent: preplay plan ids = %s", self._preplay_plan_ids)

    def _get_plan_section(self) -> str:
        if not (self.use_preplay and self._preplay_plan):
            return ""
        return format_plan_for_injection(
            plan=self._preplay_plan,
            plan_text=self._preplay_plan_text,
            workflow_content=None,
        )

    # ── LLM helpers ──────────────────────────────────────────────────────────

    def _llm_chat(self, messages: list[dict]) -> str:
        return self.llm.chat(messages)

    # ── Trajectory helpers ────────────────────────────────────────────────────

    def _record_step(self, tool_name: str, tool_kwargs: dict, result: str) -> None:
        kwargs_str = ", ".join(
            f"{k}={repr(v)}" for k, v in (tool_kwargs or {}).items()
        )
        self._trajectory.append({
            "action":      f"{tool_name}({kwargs_str})",
            "tool_name":   tool_name,
            "tool_kwargs": tool_kwargs or {},
            "result":      result,
            "reward":      0.0,
        })
        self._step += 1

    def _execute_respond(
        self,
        env,
        messages: list[dict],
        content: str,
        step_count: int,
    ) -> tuple[bool, int]:
        done = False
        try:
            from tau_bench.types import Action
            step_resp = env.step(Action(name="respond", kwargs={"content": content}))
            obs       = self._extract_obs(step_resp)
            is_done   = getattr(step_resp, "done", False)

            self._record_step("respond", {"content": content}, obs)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user",      "content": obs})

            if is_done:
                done = True
            step_count += 1
        except Exception as exc:
            logger.warning("respond fallback failed: %s", exc)
        return done, step_count

    @staticmethod
    def _extract_obs(resp) -> str:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, tuple):
            return str(resp[0])
        return str(getattr(resp, "observation", resp))

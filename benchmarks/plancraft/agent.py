"""
ProPlay agent for PlanCraft.

Workflow = individual recipe procedure (named by output item, e.g. "Craft Oak Planks").
The recipe dependency graph G = (P, E) captures which recipe sequences
have been observed to work (edges) and how reliably (reliability scores).

Per-episode flow:
  reset()
      Clear episode-local state. Recipe graph persists across episodes.

  get_action(obs, task)
      Step 0: load/create recipe graph + run preplay to generate ordered recipe plan.
      All steps: standard think/action loop with plan injected in system prompt.

  finalize_episode(success, task, reward)
      1. Induction (if reward > 0): update recipe library from episode trajectory.
      2. Sync graph to updated library.
      3. Record recipe transitions in graph.
      4. Save graph.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from base_agent import BaseAgent, _parse_think_action
from prompts.proplay_prompts import (
    PROPLAY_SYSTEM_TEMPLATE,
    PROPLAY_STEP_USER,
)
from prompts.shared import SYSTEM_BASE
from induce import run_induction, build_episode_summary
from workflow_graph import WorkflowGraph, parse_workflows
from preplay import run_preplay, format_plan_for_injection, record_execution

logger = logging.getLogger(__name__)

GRAPH_MATCH_THRESHOLD = 0.75
PLAN_MATCH_THRESHOLD  = 0.50
HISTORY_WINDOW = 20

_EXPERIENCE_SIM_THRESHOLD = 0.75


def _retrieve_similar_experiences(
    query_emb: Optional[list],
    experiences: list[tuple[list, str]],
    threshold: float = _EXPERIENCE_SIM_THRESHOLD,
) -> list[str]:
    if not query_emb:
        return [summary for _, summary in experiences]
    scored = []
    for emb, summary in experiences:
        if not emb:
            continue
        sim = sum(q * e for q, e in zip(query_emb, emb))
        if sim >= threshold:
            scored.append((sim, summary))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored]


class ProPlayAgent(BaseAgent):

    AGENT_TYPE = "proplay"

    def __init__(
        self,
        llm,
        workflow_path: str,
        preplay_graph_path: Optional[str] = None,
        use_preplay: bool = True,
        use_reliability: bool = True,
    ) -> None:
        super().__init__(llm)
        self.llm = llm
        self.workflow_path      = workflow_path
        self.preplay_graph_path = preplay_graph_path
        self.use_preplay        = use_preplay
        self.use_reliability    = use_reliability

        self._preplay_graph: Optional[WorkflowGraph] = None
        self._preplay_site: str = "plancraft"
        self._episode_summaries: list[str] = []
        self._failed_experiences: list[tuple[list, str]] = []

        # Eagerly load embedding model
        _EmbeddingGrounder.get()

        # Episode-local state
        self._graph_initialized: bool = False
        self._preplay_plan: list[str] = []
        self._preplay_plan_ids: list[str] = []
        self._preplay_plan_text: str = ""
        self._task_embedding: Optional[list] = None
        self._memory: str = ""
        self._induction_trace: list[str] = []
        self._preplay_plan_reliability: float = 0.0

    def reset(self) -> None:
        super().reset()
        self._graph_initialized = False
        self._preplay_plan = []
        self._preplay_plan_ids = []
        self._preplay_plan_text = ""
        self._task_embedding = None
        self._memory = ""
        self._induction_trace = []
        self._preplay_plan_reliability = 0.0

    def get_action(self, obs: str, task: str) -> str:
        if not self._graph_initialized and self.preplay_graph_path:
            self._init_graph_and_preplay(task)

        sys_msg = SYSTEM_BASE
        plan_section = self._get_plan_section()
        if plan_section:
            sys_msg = PROPLAY_SYSTEM_TEMPLATE.format(
                base_system=SYSTEM_BASE,
                plan_section=plan_section,
            )

        history = self._trajectory[-HISTORY_WINDOW:]
        user_msg = PROPLAY_STEP_USER.format(
            task=task,
            observation=obs,
            memory=self._memory if self._memory else "(none yet)",
            history_len=len(history),
            history=self._format_history(),
        )
        response = self.llm.chat([
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ])
        think, raw_action = _parse_think_action(response)

        mem_m = re.search(r'<memory>(.*?)</memory>', response, re.DOTALL)
        if mem_m:
            self._memory = mem_m.group(1).strip()

        logger.info("  step=%d  think=%r", self._step, think[:120])
        logger.info("  step=%d  action=%r", self._step, raw_action)

        self._record_step(obs, raw_action, raw_action, think)
        return raw_action

    def finalize_episode(self, success: bool, task: str, reward: float) -> None:
        workflow_changed = False

        if self._trajectory and reward > 0:
            summary = build_episode_summary(task, self._trajectory, reward)
            self._episode_summaries.append(summary)
            updated, self._induction_trace = run_induction(
                llm_chat_fn=self._llm_chat_long,
                episode_summaries=self._episode_summaries,
                workflow_path=self.workflow_path,
            )
            if updated:
                workflow_changed = True
                logger.info(
                    "ProPlayAgent: workflow file updated (batch induction, %d episodes)",
                    len(self._episode_summaries),
                )
        elif self._trajectory and reward <= 0:
            # Collect failed summary for preplay context
            summary = build_episode_summary(task, self._trajectory, reward)
            self._failed_experiences.append((self._task_embedding or [], summary))

        # Sync graph nodes to updated workflow library
        if workflow_changed and self._preplay_graph is not None:
            current_workflows = dict(parse_workflows(self.workflow_path))
            self._sync_graph(current_workflows)
            logger.info(
                "ProPlayAgent: graph synced → %d nodes, %d edges",
                len(self._preplay_graph.nodes),
                len(self._preplay_graph.edges),
            )

        # Record recipe transitions in graph
        if self.preplay_graph_path and self._preplay_graph is not None:
            if len(self._induction_trace) >= 2:
                effective_plan    = self._induction_trace
                effective_plan_ids = []
                logger.info("ProPlayAgent: using induction trace for graph update: %s", effective_plan)
            else:
                effective_plan    = self._preplay_plan
                effective_plan_ids = self._preplay_plan_ids
                logger.info("ProPlayAgent: using preplay plan for graph update")

            # Re-resolve empty plan_ids
            if effective_plan and any(pid == '' for pid in effective_plan_ids):
                post_nodes = self._preplay_graph.get_nodes_for_site(self._preplay_site)
                if post_nodes:
                    grounder = _EmbeddingGrounder.get()
                    empty_idxs = [i for i, pid in enumerate(effective_plan_ids) if not pid]
                    all_embs = grounder._encode(
                        [effective_plan[i] for i in empty_idxs] + [n.name for n in post_nodes]
                    )
                    q_embs = all_embs[:len(empty_idxs)]
                    n_embs = all_embs[len(empty_idxs):]
                    sims = q_embs @ n_embs.T
                    effective_plan_ids = list(effective_plan_ids)
                    for k, idx in enumerate(empty_idxs):
                        j = int(grounder._np.argmax(sims[k]))
                        best_sim = float(sims[k, j])
                        if best_sim < PLAN_MATCH_THRESHOLD:
                            continue
                        effective_plan_ids[idx] = post_nodes[j].node_id

            if effective_plan and any(pid == '' for pid in effective_plan_ids):
                filtered = [(p, pid) for p, pid in zip(effective_plan, effective_plan_ids) if pid]
                if filtered:
                    effective_plan, effective_plan_ids = map(list, zip(*filtered))
                else:
                    effective_plan, effective_plan_ids = [], []

            if len(effective_plan) >= 2:
                record_execution(
                    plan=effective_plan,
                    graph=self._preplay_graph,
                    site=self._preplay_site,
                    reward=reward,
                    plan_ids=effective_plan_ids,
                    task_embedding=self._task_embedding,
                )

        if self.preplay_graph_path and self._preplay_graph is not None:
            self._preplay_graph.save(self.preplay_graph_path)

    # ── Graph / pre-play ──────────────────────────────────────────────────────

    def _init_graph_and_preplay(self, task: str) -> None:
        self._preplay_graph = WorkflowGraph.load_or_create(
            path=self.preplay_graph_path,
            workflow_path=self.workflow_path,
            site=self._preplay_site,
        )
        self._graph_initialized = True
        self._task_embedding = _EmbeddingGrounder.get()._encode([task])[0].tolist()

        if self.use_preplay:
            wf_content = (
                {name.lower(): steps for name, steps in parse_workflows(self.workflow_path)}
                if Path(self.workflow_path).exists() else None
            )
            relevant_experiences = _retrieve_similar_experiences(
                self._task_embedding, self._failed_experiences,
            )
            self._preplay_plan, self._preplay_plan_text = run_preplay(
                llm_chat_fn=self._llm_chat,
                goal=task,
                site=self._preplay_site,
                graph=self._preplay_graph,
                use_reliability=self.use_reliability,
                workflow_content=wf_content,
                task_embedding=self._task_embedding,
                failed_experiences=relevant_experiences,
            )
            name_to_id = {
                re.sub(r'\*+', '', n.name).lower().strip(): n.node_id
                for n in self._preplay_graph.get_nodes_for_site(self._preplay_site)
            }
            self._preplay_plan_ids = [
                name_to_id.get(p.strip().lower(), "") for p in self._preplay_plan
            ]
            self._preplay_plan_reliability = self._compute_plan_reliability()
            Path(self.preplay_graph_path + ".plan.json").write_text(
                json.dumps({"plan": self._preplay_plan})
            )
            logger.info("ProPlayAgent: pre-play plan = %s", self._preplay_plan)
            logger.info("ProPlayAgent: pre-play plan reliability = %.3f", self._preplay_plan_reliability)

    def _compute_plan_reliability(self) -> float:
        plan = self._preplay_plan
        if len(plan) < 2 or self._preplay_graph is None:
            return 0.0
        edge_index = {(e.source_id, e.target_id): e for e in self._preplay_graph.edges.values()}
        name_to_id = {
            re.sub(r'\*+', '', n.name).lower().strip(): n.node_id
            for n in self._preplay_graph.get_nodes_for_site(self._preplay_site)
        }
        reliabilities = []
        for i in range(len(plan) - 1):
            src_id = name_to_id.get(plan[i].strip().lower(), "")
            tgt_id = name_to_id.get(plan[i + 1].strip().lower(), "")
            if src_id and tgt_id:
                edge = edge_index.get((src_id, tgt_id))
                rel = (self._preplay_graph.get_task_reliability(edge, self._task_embedding)
                       if edge is not None else 0.0)
            else:
                rel = 0.0
            reliabilities.append(rel)
        return sum(reliabilities) / len(reliabilities)

    def _sync_graph(self, current_workflows: dict) -> None:
        existing_nodes = self._preplay_graph.get_nodes_for_site(self._preplay_site)
        if not existing_nodes:
            for name, content in current_workflows.items():
                self._preplay_graph.add_node(
                    site=self._preplay_site, name=name, content=content,
                )
            return

        new_names  = list(current_workflows.keys())
        if not new_names:
            return

        old_names = [n.name for n in existing_nodes]
        grounder  = _EmbeddingGrounder.get()
        all_embs  = grounder._encode(new_names + old_names)
        new_embs  = all_embs[:len(new_names)]
        old_embs  = all_embs[len(new_names):]
        sim = new_embs @ old_embs.T

        matched_old: set[int] = set()
        matched_new: set[int] = set()
        flat_order = grounder._np.argsort(sim.ravel())[::-1]
        for flat_idx in flat_order:
            i, j = divmod(int(flat_idx), len(existing_nodes))
            if i in matched_new or j in matched_old:
                continue
            if sim[i, j] < GRAPH_MATCH_THRESHOLD:
                break
            node = existing_nodes[j]
            new_name = new_names[i]
            node.name = new_name
            node.content = current_workflows[new_name]
            matched_old.add(j)
            matched_new.add(i)

        for i, name in enumerate(new_names):
            if i not in matched_new:
                self._preplay_graph.add_node(
                    site=self._preplay_site, name=name, content=current_workflows[name],
                )

    def _get_plan_section(self) -> str:
        if not self.use_preplay or not self._preplay_plan:
            # No preplay: show workflow library directly if available
            wf_path = Path(self.workflow_path)
            if not wf_path.exists():
                return ""
            content = wf_path.read_text()
            m = re.search(r'##\s*Summary Workflows\s*\n', content, re.I)
            wf_text = content[m.end():].strip() if m else content.strip()
            if not wf_text:
                return ""
            return f"## Recipe Library\n\n{wf_text}"

        wf_content = (
            {name.lower(): steps for name, steps in parse_workflows(self.workflow_path)}
            if Path(self.workflow_path).exists() else None
        )
        return format_plan_for_injection(
            plan=self._preplay_plan,
            plan_text=self._preplay_plan_text,
            workflow_content=wf_content,
        )

    def _llm_chat(self, messages: list[dict]) -> str:
        return self.llm.chat(messages)

    def _llm_chat_long(self, messages: list[dict]) -> str:
        return self.llm.chat(messages, max_tokens=8192)


# ── Embedding model ───────────────────────────────────────────────────────────

class _EmbeddingGrounder:
    _instance: Optional["_EmbeddingGrounder"] = None

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._np = np
        logger.info("Loading embedding model (graph sync)...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._cache: dict = {}
        logger.info("Embedding model loaded.")

    @classmethod
    def get(cls) -> "_EmbeddingGrounder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _encode(self, texts: list[str]):
        new = [t for t in texts if t not in self._cache]
        if new:
            embs = self._model.encode(new, normalize_embeddings=True, show_progress_bar=False)
            for t, e in zip(new, embs):
                self._cache[t] = e
        return self._np.stack([self._cache[t] for t in texts])

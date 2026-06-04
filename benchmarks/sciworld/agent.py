"""
AWM Agent for ScienceWorld with optional pre-play.

Per-episode flow
----------------
reset()
    Clear episode-local state. Pre-play graph persists across episodes.

get_action(obs, task) -> str
    LLM reasons in <think> and outputs intended action in <action>.
    The raw action is sent to the environment directly (no grounding).

finalize_episode(success, task)
    1. Induction (if reward > 0): re-run batch induction over all successful
       episodes; writes ## Summary Workflows.
    2. _sync_graph (if any file change): embed-similarity match between updated
       workflow names and existing graph nodes; rename in place, add new nodes.
    3. record_execution: record preplay plan transitions against post-sync graph.
       If no preplay plan existed (first episode / empty graph), synthesize a
       linear plan from the workflow file order so the reward propagates to edges.
    4. Save graph.
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
    SYSTEM_BASE, STEP_USER,
    format_workflow_section, format_history, parse_memory,
    format_episode_summary, extract_failed_steps_summary,
)
from induce import run_induction
from workflow_graph import WorkflowGraph, parse_workflows
from preplay import run_preplay, format_plan_for_injection, record_execution

logger = logging.getLogger(__name__)

GRAPH_MATCH_THRESHOLD = 0.75   # cosine similarity cutoff for workflow name matching
PLAN_MATCH_THRESHOLD  = 0.50   # cosine similarity cutoff for re-resolving preplay plan entries
HISTORY_WINDOW = 20            # max steps shown in prompt; 50-step budget fits in 128K but avoids OOM on CPU node


def _extract_to_focus(task: str) -> list[str]:
    """
    SwiftSage (arXiv:2305.17390): extract target objects for 'focus on' from the
    task description.  SciWorld task descriptions for substance tasks say e.g.
    "First, focus on the ice." → extracts ["ice"].
    """
    pattern = r"focus on\s+(\b\w+\b(?:\s+\b\w+\b)*)"
    items: list[str] = []
    seen: set[str] = set()
    for m in re.findall(pattern, task, re.IGNORECASE):
        item = m.replace("the ", "").strip().lower()
        if item and item not in seen:
            seen.add(item)
            items.append(item)
    return items


_EXPERIENCE_SIM_THRESHOLD = 0.75  # cosine similarity cutoff for preplay experience retrieval


def _retrieve_similar_experiences(
    query_emb: Optional[list],
    experiences: list[tuple[list, str]],
    threshold: float = _EXPERIENCE_SIM_THRESHOLD,
) -> list[str]:
    """
    Return summaries from `experiences` whose embedding cosine-similarity to
    `query_emb` exceeds `threshold`, sorted highest-sim first.

    If query_emb is None/empty, returns all summaries unchanged.
    Embeddings are assumed unit-normalised, so dot-product == cosine similarity.
    """
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

    AGENT_TYPE = "awm_sciworld"

    def __init__(
        self,
        llm,
        workflow_path: str,
        preplay_graph_path: Optional[str] = None,
        use_preplay: bool = True,
        use_reliability: bool = True,
    ) -> None:
        self.llm = llm
        self.workflow_path = workflow_path
        self.preplay_graph_path = preplay_graph_path
        self.use_preplay = use_preplay
        self.use_reliability = use_reliability

        # ── Cross-episode state ───────────────────────────────────────────────
        # Pre-play graph and episode summaries persist across episodes.
        self._preplay_graph: Optional[WorkflowGraph] = None
        self._preplay_site: str = "sciworld"
        self._episode_summaries: list[str] = []                          # successful episodes (for induction)
        self._failed_experiences: list[tuple[list, str]] = []          # (task_embedding, summary) for preplay retrieval

        # Eagerly load the embedding model so neither the first grounding call
        # nor the first finalize_episode incurs a cold-start delay.
        _EmbeddingGrounder.get()

        # ── Episode-local state ───────────────────────────────────────────────
        self._trajectory: list[dict] = []
        self._step: int = 0
        self._graph_initialized: bool = False
        self._preplay_plan: list[str] = []
        self._preplay_plan_ids: list[str] = []   # node_id for each plan entry at preplay time
        self._preplay_plan_text: str = ""        # full structured plan text for agent injection
        self._task_embedding: Optional[list] = None  # unit-normalized embedding of current task
        self._memory: str = ""
        self._room_state: dict[str, str] = {}   # room_name -> last full obs text
        self._current_room: str = "unknown"
        self._to_focus: list[str] = []          # SwiftSage: valid focus targets for this task
        self._induction_trace: list[str] = []   # execution trace from latest induction call
        self._preplay_plan_reliability: float = 0.0  # avg task-specific reliability of plan transitions

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear episode-local state. Pre-play graph persists across episodes."""
        self._trajectory = []
        self._step = 0
        self._graph_initialized = False
        self._preplay_plan = []
        self._preplay_plan_ids = []
        self._preplay_plan_text = ""
        self._task_embedding = None
        self._memory = ""
        self._room_state = {}
        self._current_room = "unknown"
        self._to_focus = []
        self._induction_trace = []
        self._preplay_plan_reliability = 0.0

    def get_action(
        self,
        obs: str,
        task: str,
    ) -> str:
        if not self._graph_initialized and self.preplay_graph_path:
            self._init_graph_and_preplay(task)

        if self._step == 0:
            self._to_focus = _extract_to_focus(task)
            if self._to_focus:
                logger.info("AWMAgent: to_focus = %s", self._to_focus)

        self._update_room_state(obs)

        sys_msg = SYSTEM_BASE
        wf_section = self._get_workflow_section()
        if wf_section:
            sys_msg += f"\n\n{wf_section}"
        if self._to_focus:
            focus_items = ", ".join(self._to_focus)
            sys_msg += (
                f"\n\nRemember that it is ONLY possible to focus on these items: "
                f"{focus_items}! Do NOT focus on other things! "
                f"Focus directly on the substance or object, NOT on any container holding it."
            )

        history = self._trajectory[-HISTORY_WINDOW:]
        user_msg = STEP_USER.format(
            task=task,
            observation=obs,
            room_state=self._format_room_state(),
            memory=self._memory if self._memory else "(none yet)",
            history_len=len(history),
            history=format_history(history),
            workflow_section="",
        )

        response = self.llm.chat([
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ])
        think, action_raw = _parse_think_action(response)
        logger.info("  step=%d  think=%r", self._step, think)
        logger.info("  step=%d  action=%r", self._step, action_raw)
        self._memory = parse_memory(response) or self._memory

        action = action_raw
        self._trajectory.append({"action": action, "raw_action": action_raw, "obs": obs, "think": think})
        self._step += 1
        return action

    def finalize_episode(self, success: bool, task: str, reward: float = 0.0) -> None:
        workflow_changed = False

        if self._trajectory:
            # Always generate the full summary with two-tier markers.
            summary = format_episode_summary(task, self._trajectory)

            # Collect full summary for preplay context when episode had failed steps.
            if extract_failed_steps_summary(summary):
                self._failed_experiences.append((self._task_embedding or [], summary))
                logger.info("AWMAgent: recorded episode summary (with failed steps) for preplay context")

            # Run induction only when the episode produced meaningful reward.
            if reward > 0:
                self._episode_summaries.append(summary)
                updated, self._induction_trace = run_induction(
                    llm_chat_fn=self._llm_chat_long,
                    episode_summaries=self._episode_summaries,
                    workflow_path=self.workflow_path,
                )
                if updated:
                    workflow_changed = True
                    logger.info(
                        "AWMAgent: workflow file updated (batch induction, %d episodes)",
                        len(self._episode_summaries),
                    )
            else:
                logger.info("AWMAgent: episode skipped for induction (no reward gain)")

        # Sync graph nodes to the latest workflow file after induction.
        if workflow_changed and self._preplay_graph is not None:
            current_workflows = dict(parse_workflows(self.workflow_path))
            self._sync_graph(current_workflows)
            logger.info(
                "AWMAgent: graph synced → %d nodes, %d edges",
                len(self._preplay_graph.nodes),
                len(self._preplay_graph.edges),
            )

        # Record edge transitions against the post-sync graph.
        # Use the induction trace when available (≥2 entries): it reflects the
        # workflow sequence the LLM identified as actually executed this episode,
        # preventing spurious reliability reinforcement from the preplay plan.
        # Fall back to the preplay plan when the trace is absent or too short.
        # Note: when using the trace, plan_ids is left empty so record_execution
        # falls through to name-based lookup, which is safe because _extract_trace
        # already validated every name against the updated library.
        if self.preplay_graph_path and self._preplay_graph is not None:
            if len(self._induction_trace) >= 2:
                effective_plan = self._induction_trace
                effective_plan_ids = []   # name-based lookup in record_execution
                logger.info(
                    "AWMAgent: using induction trace for graph update (%d nodes): %s",
                    len(effective_plan), effective_plan,
                )
            else:
                effective_plan = self._preplay_plan
                effective_plan_ids = self._preplay_plan_ids
                logger.info(
                    "AWMAgent: induction trace unavailable (%d entries), using preplay plan",
                    len(self._induction_trace),
                )
            # Re-resolve empty plan_ids against the post-sync graph using
            # embedding similarity (same approach as _sync_graph). Preplay
            # resolves node_ids before induction, so plan entries for workflows
            # that didn't exist yet get ''. After _sync_graph adds them, we
            # match by embedding rather than string because preplay names are
            # often task-specific (e.g. "Focus on the living thing") while
            # induction names are generic (e.g. "Focus on Object").
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
                            logger.info(
                                "AWMAgent: re-resolve SKIPPED plan[%d] %r (hallucinated, best sim=%.3f < %.2f)",
                                idx, effective_plan[idx], best_sim, PLAN_MATCH_THRESHOLD,
                            )
                            continue  # leave plan_id as '' to be filtered below
                        effective_plan_ids[idx] = post_nodes[j].node_id
                        logger.info(
                            "AWMAgent: re-resolved plan[%d] %r → %r (sim=%.3f)",
                            idx, effective_plan[idx], post_nodes[j].name, best_sim,
                        )
            # Filter out plan entries that still have no matching graph node
            # (hallucinated preplay entries that couldn't be grounded).
            if effective_plan and any(pid == '' for pid in effective_plan_ids):
                filtered = [(p, pid) for p, pid in zip(effective_plan, effective_plan_ids) if pid]
                if filtered:
                    effective_plan, effective_plan_ids = map(list, zip(*filtered))
                else:
                    effective_plan, effective_plan_ids = [], []
                logger.info(
                    "AWMAgent: filtered plan to %d grounded entries (removed unresolvable hallucinations)",
                    len(effective_plan),
                )
            if effective_plan:
                plan_to_record = effective_plan
                plan_ids_to_record = effective_plan_ids
                # When the episode ended catastrophically (reward=-1), skip the
                # last transition so the terminal workflow node doesn't accumulate
                # false positive reliability from wrong-focus terminations.
                if self._trajectory and (self._trajectory[-1].get("reward") or 0.0) == -1:
                    plan_to_record = effective_plan[:-1]
                    plan_ids_to_record = effective_plan_ids[:-1] if effective_plan_ids else []
                if len(plan_to_record) >= 2:
                    record_execution(
                        plan=plan_to_record,
                        graph=self._preplay_graph,
                        site=self._preplay_site,
                        reward=reward,
                        plan_ids=plan_ids_to_record,
                        task_embedding=self._task_embedding,
                    )

        if self.preplay_graph_path and self._preplay_graph is not None:
            self._preplay_graph.save(self.preplay_graph_path)

    # ── Graph / pre-play ──────────────────────────────────────────────────────

    def _sync_graph(self, current_workflows: dict) -> None:
        """
        Sync graph nodes to the freshly-induced workflow file using embedding
        similarity so that LLM renames don't destroy accumulated edge data.

        For each new workflow name, find the most similar existing node:
          - similarity >= GRAPH_MATCH_THRESHOLD → update name + content in place
            (node_id and all edges are preserved)
          - no good match → add as a new node
        Existing nodes with no incoming match are kept to preserve their edge data.
        """
        existing_nodes = self._preplay_graph.get_nodes_for_site(self._preplay_site)
        if not existing_nodes:
            for name, content in current_workflows.items():
                self._preplay_graph.add_node(
                    site=self._preplay_site, name=name, content=content
                )
            return

        new_names = list(current_workflows.keys())
        if not new_names:
            return

        old_names = [n.name for n in existing_nodes]

        grounder = _EmbeddingGrounder.get()
        all_embs = grounder._encode(new_names + old_names)
        new_embs = all_embs[:len(new_names)]
        old_embs = all_embs[len(new_names):]
        sim = new_embs @ old_embs.T   # shape: (n_new, n_old)

        matched_old: set[int] = set()   # indices into existing_nodes already matched
        matched_new: set[int] = set()   # indices into new_names already matched

        # Greedy best-first matching
        flat_order = grounder._np.argsort(sim.ravel())[::-1]
        for flat_idx in flat_order:
            i, j = divmod(int(flat_idx), len(existing_nodes))
            if i in matched_new or j in matched_old:
                continue
            if sim[i, j] < GRAPH_MATCH_THRESHOLD:
                break
            node = existing_nodes[j]
            new_name = new_names[i]
            logger.info(
                "AWMAgent: graph match  %r → %r  (sim=%.3f)",
                node.name, new_name, float(sim[i, j]),
            )
            node.name = new_name
            node.content = current_workflows[new_name]
            matched_old.add(j)
            matched_new.add(i)

        # Add genuinely new workflows (unmatched new names become new nodes).
        # Unmatched existing nodes are intentionally kept — deleting them would
        # destroy their accumulated edge data, preventing the graph from growing.
        for i, name in enumerate(new_names):
            if i not in matched_new:
                logger.info("AWMAgent: graph new node  %r", name)
                self._preplay_graph.add_node(
                    site=self._preplay_site, name=name, content=current_workflows[name]
                )

        for j, node in enumerate(existing_nodes):
            if j not in matched_old:
                logger.info("AWMAgent: graph keep unmatched  %r", node.name)

    def _init_graph_and_preplay(self, task: str) -> None:
        self._preplay_graph = WorkflowGraph.load_or_create(
            path=self.preplay_graph_path,
            workflow_path=self.workflow_path,
            site=self._preplay_site,
        )
        self._graph_initialized = True   # set after load succeeds
        # Embed the task string once; reused for preplay scoring and edge recording.
        self._task_embedding = _EmbeddingGrounder.get()._encode([task])[0].tolist()
        if self.use_preplay:
            wf_content_for_preplay = (
                {name.lower(): steps for name, steps in parse_workflows(self.workflow_path)}
                if Path(self.workflow_path).exists() else None
            )
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
                workflow_content=wf_content_for_preplay,
                task_embedding=self._task_embedding,
                failed_experiences=relevant_experiences,
            )
            # Resolve plan names → node_ids immediately, before any induction
            # or _sync_graph can rename nodes this episode.
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
            logger.info("AWMAgent: pre-play plan = %s", self._preplay_plan)
            logger.info("AWMAgent: pre-play plan ids = %s", self._preplay_plan_ids)
            logger.info("AWMAgent: pre-play plan reliability = %.3f", self._preplay_plan_reliability)

    def _compute_plan_reliability(self) -> float:
        """Compute avg task-specific reliability of consecutive transitions in the preplay plan.

        For each pair (plan[i], plan[i+1]):
          - edge exists in graph  → graph.get_task_reliability(edge, task_embedding)
          - edge missing          → 0.0  (penalises hallucinated or unseen transitions)
        Episodes with plan length < 2 have no transitions and return 0.0.
        """
        plan = self._preplay_plan
        if len(plan) < 2 or self._preplay_graph is None:
            return 0.0

        edge_index = {
            (e.source_id, e.target_id): e
            for e in self._preplay_graph.edges.values()
        }
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
                if edge is not None:
                    rel = self._preplay_graph.get_task_reliability(edge, self._task_embedding)
                else:
                    rel = 0.0   # transition not yet observed in graph
            else:
                rel = 0.0       # hallucinated workflow name not in graph
            reliabilities.append(rel)

        return sum(reliabilities) / len(reliabilities)

    def _get_induction_trajectory(self, final_reward: float) -> Optional[list[dict]]:
        """Return the productive portion of the trajectory for batch induction.

        Truncates at the last step where cumulative reward increased, plus up to
        5 follow-through steps. Steps where reward == -1 (invalid actions) are
        removed from the returned slice. Returns None when no reward gain occurred.
        """
        if final_reward <= 0:
            return None

        last_reward_idx = -1
        prev_reward = 0.0
        for i, step in enumerate(self._trajectory):
            r = step.get("reward") or 0.0
            if r > prev_reward:
                last_reward_idx = i
            prev_reward = r

        if last_reward_idx < 0:
            return None

        cutoff = min(last_reward_idx + 5, len(self._trajectory) - 1)
        return [
            step for step in self._trajectory[:cutoff + 1]
            if (step.get("reward") or 0.0) != -1
        ]

    def _get_workflow_section(self) -> str:
        wf_path = Path(self.workflow_path)
        if not wf_path.exists():
            return ""

        # Inject only the Summary Workflows section (skip Concrete Examples)
        content = wf_path.read_text()
        summary_match = re.search(r'##\s*Summary Workflows\s*\n', content, re.I)
        wf_text = content[summary_match.end():].strip() if summary_match else content.strip()
        if not wf_text:
            return ""

        if self.use_preplay and self._preplay_plan:
            wf_content = {name.lower(): steps for name, steps in parse_workflows(self.workflow_path)}
            plan_section = format_plan_for_injection(
                plan=self._preplay_plan,
                plan_text=self._preplay_plan_text,
                workflow_content=wf_content,
            )
            if plan_section:
                return plan_section

        return format_workflow_section(wf_text)

    def _llm_chat(self, messages: list[dict]) -> str:
        return self.llm.chat(messages)

    def _llm_chat_long(self, messages: list[dict]) -> str:
        """Higher token budget for induction — library + trace can exceed the default limit."""
        return self.llm.chat(messages, max_tokens=8192)

    # ── Room state tracking ───────────────────────────────────────────────────

    def _update_room_state(self, obs: str) -> None:
        """Parse obs to update room name and per-room object cache."""
        # "You move to the kitchen." / "You are now in the kitchen."
        move_m = re.search(
            r'[Yy]ou (?:move|go|are now) (?:to |in )?(?:the )?([A-Za-z][\w\s]+?)[\.\!]',
            obs,
        )
        if move_m:
            self._current_room = move_m.group(1).strip()

        # "In the kitchen you see: ..." — full room description
        room_m = re.search(r'[Ii]n (?:the )?([A-Za-z][\w\s]+?) you see[:\s]', obs)
        if room_m:
            room = room_m.group(1).strip()
            self._current_room = room
            self._room_state[room] = obs.strip()

    def _format_room_state(self) -> str:
        if not self._room_state:
            return "(no room observations yet)"
        lines = []
        for room, obs_text in self._room_state.items():
            marker = " [you are here]" if room == self._current_room else ""
            short = obs_text[:400] + "..." if len(obs_text) > 400 else obs_text
            lines.append(f"{room}{marker}: {short}")
        return "\n".join(lines)



# ── Module-level helpers ──────────────────────────────────────────────────────

def _parse_think_action(response: str) -> tuple[str, str]:
    """
    Extract (think, action) from AWM-style <think>...</think><action>...</action>.
    Falls back to treating the whole response as an action if tags are absent.
    """
    think = ""
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()

    action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
    if action_match:
        return think, action_match.group(1).strip()

    # Fallback: first non-empty line
    for line in response.strip().splitlines():
        line = line.strip()
        if line and not line.startswith("<"):
            return think, line
    return think, response.strip()


class _EmbeddingGrounder:
    """Singleton sentence-embedding model for workflow graph matching.
    Loaded once on first use; subsequent calls reuse the cached model.
    """
    _instance: Optional["_EmbeddingGrounder"] = None

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._np = np
        logger.info("Loading embedding model (graph sync)...")
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

    def best_match(self, intent: str, candidates: list[str]) -> str:
        embs = self._encode([intent] + candidates)
        scores = embs[1:] @ embs[0]
        best = int(self._np.argmax(scores))
        logger.info(
            "  grounding  embedding  best=%r  score=%.3f",
            candidates[best], float(scores[best]),
        )
        return candidates[best]



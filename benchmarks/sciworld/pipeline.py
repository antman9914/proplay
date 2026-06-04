"""
AWM + Pre-play evaluation pipeline for ScienceWorld.

Usage:
    python pipeline.py \\
        --n 100 --steps 30 \\
        --server http://localhost:36006 \\
        --model Qwen/Qwen2.5-32B-Instruct-AWQ \\
        --api_base http://localhost:8000/v1 \\
        [--preplay]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # /proplay root — needed for proplay.* imports
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# memory_planning_demo deps moved into proplay package
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from proplay.llm import LLMClient, LLMConfig
from benchmarks.sciworld.router import SciWorldEnv
from agent import AWMAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    task_idx: int
    reward: float
    steps: int
    success: bool
    n_workflows: int   # number of workflows in file after this episode
    graph_nodes: int = 0
    graph_edges: int = 0
    raw_reward: float = 0.0          # env.reward before no_stop correction
    step_rewards: list = None        # per-step reward from env after each action
    plan_reliability: float = 0.0    # avg task-specific reliability of preplay plan transitions
    plan_length: int = 0             # number of workflows in preplay plan
    preplay_plan: list = None        # workflow names in preplay plan


def run_episode(
    agent: AWMAgent,
    env: SciWorldEnv,
    task_idx: int,
    max_steps: int,
) -> EpisodeResult:
    env.reset(task_idx)
    agent.reset()

    task = env.task_description
    steps_taken = 0
    last_nonneg_reward = 0.0   # SwiftSage no_stop=True: freeze score before going negative
    step_rewards: list[float] = []

    for step in range(max_steps):
        if env.done:
            break

        obs = env.observe
        logger.info("  step=%d  obs=%r", step, obs[:200])

        try:
            action = agent.get_action(obs, task)
        except Exception as exc:
            logger.error("get_action failed at step %d: %s", step, exc)
            break

        try:
            result = env.step(action)
        except Exception as exc:
            logger.error("env.step failed at step %d: %s", step, exc)
            break

        # SwiftSage: SciWorld sometimes returns "Ambiguous request" when an action
        # matches multiple objects; responding with "0" selects the first match.
        if result.observation.startswith("Ambiguous request"):
            try:
                result = env.step("0")
            except Exception as exc:
                logger.warning("env.step('0') for ambiguous request failed: %s", exc)

        # Track last non-negative cumulative score before any catastrophic action.
        step_rewards.append(env.reward)
        if env.reward >= 0:
            last_nonneg_reward = env.reward

        # Store the post-action observation and per-step reward for trajectory analysis
        if agent._trajectory:
            agent._trajectory[-1]["result_obs"] = result.observation
            agent._trajectory[-1]["reward"] = result.reward

        logger.info(
            "  step=%d  action=%r  reward=%.3f",
            step, action[:60], result.reward,
        )
        steps_taken = step + 1

        if result.done:
            break

    # SwiftSage no_stop=True: when the episode ends with a negative score (e.g. a
    # catastrophic focus action), report the last non-negative score rather than -1.
    # This matches SwiftSage's default evaluation behaviour.
    final_reward = env.reward if env.reward >= 0 else last_nonneg_reward
    if env.reward < 0:
        logger.info(
            "  episode ended with reward=%.3f → reporting last_nonneg_reward=%.3f",
            env.reward, last_nonneg_reward,
        )

    success_metric = final_reward >= 1.0         # for SR reporting only
    success_learn  = final_reward > 0            # for AWM induction
    try:
        agent.finalize_episode(success=success_learn, task=task, reward=final_reward)
    except Exception as exc:
        logger.warning("finalize_episode failed: %s", exc)

    n_wf = _count_workflows(agent.workflow_path)
    graph_nodes, graph_edges = _log_graph_state(agent)
    return EpisodeResult(
        task_idx=task_idx,
        reward=final_reward,
        steps=steps_taken,
        success=success_metric,
        n_workflows=n_wf,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        raw_reward=env.reward,
        step_rewards=step_rewards,
        plan_reliability=agent._preplay_plan_reliability,
        plan_length=len(agent._preplay_plan),
        preplay_plan=list(agent._preplay_plan),
    )


def evaluate(
    agent: AWMAgent,
    env: SciWorldEnv,
    n_episodes: int,
    start_idx: int = 0,
    max_steps: int = 30,
    log_dir: Optional[str] = None,
    task_ids: Optional[list] = None,
) -> None:
    task_list = task_ids if task_ids is not None else list(range(start_idx, start_idx + n_episodes))
    n_total = len(task_list)
    results: list[EpisodeResult] = []
    t0 = time.time()

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(task_list):
        try:
            ep = run_episode(agent, env, idx, max_steps)
        except Exception as exc:
            logger.error("Episode %d failed: %s", idx, exc)
            ep = EpisodeResult(task_idx=idx, reward=0.0, steps=0, success=False, n_workflows=0)
        results.append(ep)

        n_done = len(results)
        sr = sum(r.success for r in results) / n_done
        avg_score = sum(r.reward for r in results) / n_done
        logger.info(
            "[%d/%d] task=%d  success=%s  steps=%d  reward=%.3f  SR=%.3f  avg_score=%.3f  "
            "workflows=%d  graph_nodes=%d  graph_edges=%d",
            i + 1, n_total, idx,
            "✓" if ep.success else "✗",
            ep.steps, ep.reward, sr, avg_score, ep.n_workflows,
            ep.graph_nodes, ep.graph_edges,
        )

        if log_dir:
            mode = "a" if i > 0 else "w"
            with open(Path(log_dir) / "episodes.jsonl", mode) as f:
                f.write(json.dumps(asdict(ep)) + "\n")

    elapsed = time.time() - t0
    sr = sum(r.success for r in results) / n_total
    stats = agent.llm.stats()

    # ── Reliability analysis ──────────────────────────────────────────────────
    # Only episodes with a real preplay plan (plan_length >= 2) have meaningful
    # transition reliability data; single-step and empty plans are excluded.
    rel_eps = [r for r in results if r.plan_length >= 2]
    rel_analysis = _compute_reliability_analysis(results, rel_eps)

    print("\n" + "=" * 55)
    print(f"  Agent:        {agent.AGENT_TYPE}")
    print(f"  Episodes:     {n_total}")
    print(f"  Success rate: {sr:.3f}  ({sr*100:.1f}%)")
    print(f"  Avg steps:    {sum(r.steps for r in results)/n_total:.1f}")
    print(f"  Avg reward:   {sum(r.reward for r in results)/n_total:.3f}")
    print(f"  LLM calls:    {stats['total_calls']}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    if rel_eps:
        thr = rel_analysis["threshold"]
        hi  = rel_analysis["high_reliability"]
        lo  = rel_analysis["low_reliability"]
        print(f"  ── Reliability analysis (plan_length≥2, n={len(rel_eps)}) ──")
        print(f"  Avg plan reliability:  {rel_analysis['avg_plan_reliability']:.3f}")
        print(f"  Threshold (median):    {thr:.3f}")
        print(f"  High-rel  (n={hi['n']:3d}):  SR={hi['sr']:.3f}  avg_score={hi['avg_score']:.3f}  avg_rel={hi['avg_plan_reliability']:.3f}")
        print(f"  Low-rel   (n={lo['n']:3d}):  SR={lo['sr']:.3f}  avg_score={lo['avg_score']:.3f}  avg_rel={lo['avg_plan_reliability']:.3f}")
    print("=" * 55 + "\n")

    if log_dir:
        summary = {
            "agent": agent.AGENT_TYPE,
            "n_episodes": n_total,
            "task_ids": task_list,
            "success_rate": sr,
            "avg_steps": sum(r.steps for r in results) / n_total,
            "avg_reward": sum(r.reward for r in results) / n_total,
            "llm_calls": stats["total_calls"],
            "elapsed_sec": elapsed,
        }
        with open(Path(log_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        with open(Path(log_dir) / "reliability_analysis.json", "w") as f:
            json.dump(rel_analysis, f, indent=2)

        logger.info("Results saved to %s", log_dir)


def _count_workflows(workflow_path: str) -> int:
    import re
    path = Path(workflow_path)
    if not path.exists():
        return 0
    return len(re.findall(r'^Workflow\s+\d+:', path.read_text(), re.MULTILINE))


def _log_graph_state(agent: AWMAgent) -> tuple[int, int]:
    """Log workflow graph structure after each episode. Returns (n_nodes, n_edges)."""
    graph = agent._preplay_graph
    if graph is None:
        return 0, 0

    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

    if n_nodes == 0:
        logger.info("  graph: (empty)")
        return 0, 0

    # Node list
    node_lines = [f"    [{n.node_id}] {n.name}" for n in graph.nodes.values()]
    logger.info("  graph nodes (%d):\n%s", n_nodes, "\n".join(node_lines))

    # Edge list with reliability
    if n_edges > 0:
        edge_lines = []
        for e in graph.edges.values():
            src = graph.nodes.get(e.source_id)
            tgt = graph.nodes.get(e.target_id)
            if src and tgt:
                edge_lines.append(
                    f"    {src.name!r} → {tgt.name!r}  "
                    f"reliability={e.reliability:.2f}  count={e.count}"
                )
        logger.info("  graph edges (%d):\n%s", n_edges, "\n".join(edge_lines))
    else:
        logger.info("  graph edges: (none yet)")

    return n_nodes, n_edges


def _compute_reliability_analysis(
    all_results: list,
    rel_eps: list,
) -> dict:
    """
    Build reliability analysis dict from episode results.

    rel_eps: subset of all_results where plan_length >= 2 (have real transitions).
    High/low split uses the median plan_reliability of rel_eps as threshold.
    Episodes with plan_reliability exactly equal to the threshold go into high group.
    """
    per_episode = [
        {
            "episode_order": i,
            "task_idx": r.task_idx,
            "plan_length": r.plan_length,
            "plan_reliability": r.plan_reliability,
            "preplay_plan": r.preplay_plan or [],
            "success": r.success,
            "reward": r.reward,
        }
        for i, r in enumerate(all_results)
    ]

    if not rel_eps:
        return {
            "per_episode": per_episode,
            "avg_plan_reliability": 0.0,
            "threshold": 0.0,
            "high_reliability": {"n": 0, "sr": 0.0, "avg_score": 0.0, "avg_plan_reliability": 0.0},
            "low_reliability":  {"n": 0, "sr": 0.0, "avg_score": 0.0, "avg_plan_reliability": 0.0},
        }

    rels = sorted(r.plan_reliability for r in rel_eps)
    n = len(rels)
    # median
    threshold = (rels[n // 2 - 1] + rels[n // 2]) / 2 if n % 2 == 0 else rels[n // 2]

    high = [r for r in rel_eps if r.plan_reliability >= threshold]
    low  = [r for r in rel_eps if r.plan_reliability <  threshold]

    def _group_stats(group):
        if not group:
            return {"n": 0, "sr": 0.0, "avg_score": 0.0, "avg_plan_reliability": 0.0}
        return {
            "n": len(group),
            "sr": sum(r.success for r in group) / len(group),
            "avg_score": sum(r.reward for r in group) / len(group),
            "avg_plan_reliability": sum(r.plan_reliability for r in group) / len(group),
        }

    return {
        "per_episode": per_episode,
        "avg_plan_reliability": sum(r.plan_reliability for r in rel_eps) / len(rel_eps),
        "threshold": threshold,
        "high_reliability": _group_stats(high),
        "low_reliability":  _group_stats(low),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int,   default=100)
    parser.add_argument("--start",   type=int,   default=0)
    parser.add_argument("--steps",   type=int,   default=30)
    parser.add_argument("--server",  type=str,   default="http://localhost:36006")
    parser.add_argument("--model",   type=str,   default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--api_base",type=str,   default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str,   default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--workflow_path", type=str, default="workflow/sciworld.txt")
    parser.add_argument("--use_graph", action="store_true",
                        help="Maintain workflow graph (track transitions, update reliability)")
    parser.add_argument("--preplay", action="store_true",
                        help="Use pre-play at inference time (requires --use_graph)")
    parser.add_argument("--no_reliability", action="store_true",
                        help="Ablation: hide edge reliability scores from preplay LLM")
    parser.add_argument("--preplay_graph_path", type=str,
                        default="workflow/preplay_graph_sciworld.json")
    parser.add_argument("--log_dir", type=str,   default="./logs/awm_sciworld")
    parser.add_argument("--task_ids", type=str,  default=None,
                        help="JSON file with list of task indices, or comma-separated integers. "
                             "Overrides --n and --start when provided.")
    args = parser.parse_args()

    # Parse --task_ids: JSON file or comma-separated string
    task_ids = None
    if args.task_ids:
        p = Path(args.task_ids)
        if p.exists() and p.suffix == ".json":
            with p.open() as f:
                task_ids = json.load(f)
        else:
            task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
        logger.info("Using explicit task_ids list (%d episodes)", len(task_ids))

    # pre-play requires the graph; graph can exist without pre-play
    use_graph = args.use_graph or args.preplay
    preplay_graph_path = args.preplay_graph_path if use_graph else None

    llm = LLMClient(LLMConfig(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
    ))
    env   = SciWorldEnv(env_server_base=args.server)
    agent = AWMAgent(
        llm=llm,
        workflow_path=args.workflow_path,
        preplay_graph_path=preplay_graph_path,
        use_preplay=args.preplay,
        use_reliability=not args.no_reliability,
    )

    evaluate(
        agent=agent,
        env=env,
        n_episodes=args.n,
        start_idx=args.start,
        max_steps=args.steps,
        log_dir=args.log_dir,
        task_ids=task_ids,
    )


if __name__ == "__main__":
    main()

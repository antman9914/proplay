"""
τ-Bench preplay (AWM) evaluation pipeline.

Usage:
    python pipeline.py \\
        --env        retail \\
        --steps      30 \\
        --model      gpt-4.1-mini \\
        --api_base   https://api.openai.com/v1 \\
        --graph_path ./graph/retail.json \\
        --preplay \\
        --log_dir    ./logs/awm_retail
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
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# memory_planning_demo deps moved into proplay package
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from llm_client import TauBenchLLMClient, LLMConfig

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
    fail_eps: int = 0
    tool_nodes: int = 0


def run_episode(agent, env, task_idx: int, max_steps: int) -> EpisodeResult:
    reward, trajectory = agent.solve(env, task_idx, max_steps=max_steps)
    success = reward >= 1.0

    try:
        agent.finalize_episode(success=success, task=agent._task, reward=reward)
    except Exception as exc:
        logger.warning("finalize_episode failed: %s", exc)

    n_tool_nodes = (
        len(agent._preplay_graph.nodes) if agent._preplay_graph is not None else 0
    )
    return EpisodeResult(
        task_idx=task_idx,
        reward=reward,
        steps=len(trajectory),
        success=success,
        fail_eps=len(agent._failed_experiences),
        tool_nodes=n_tool_nodes,
    )


def evaluate(
    agent,
    env,
    task_ids: list[int],
    max_steps: int = 30,
    log_dir: Optional[str] = None,
) -> None:
    n_total = len(task_ids)
    results: list[EpisodeResult] = []
    t0 = time.time()

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(task_ids):
        try:
            ep = run_episode(agent, env, idx, max_steps)
        except Exception as exc:
            logger.error("Episode %d failed: %s", idx, exc)
            ep = EpisodeResult(task_idx=idx, reward=0.0, steps=0, success=False)
        results.append(ep)

        n_done = len(results)
        sr = sum(r.success for r in results) / n_done
        logger.info(
            "[%d/%d] task=%d  success=%s  steps=%d  reward=%.1f  SR=%.3f  "
            "fail_eps=%d  tools=%d",
            i + 1, n_total, idx,
            "YES" if ep.success else "NO",
            ep.steps, ep.reward, sr,
            ep.fail_eps, ep.tool_nodes,
        )

        if log_dir:
            mode = "a" if i > 0 else "w"
            with open(Path(log_dir) / "episodes.jsonl", mode) as f:
                f.write(json.dumps(asdict(ep)) + "\n")

    elapsed = time.time() - t0
    sr = sum(r.success for r in results) / n_total
    stats = agent.llm.stats()

    print("\n" + "=" * 55)
    print(f"  Agent:        {agent.AGENT_TYPE}")
    print(f"  Episodes:     {n_total}")
    print(f"  Success rate: {sr:.3f}  ({sr*100:.1f}%)")
    print(f"  Avg steps:    {sum(r.steps for r in results)/n_total:.1f}")
    print(f"  LLM calls:    {stats['total_calls']}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print("=" * 55 + "\n")

    if log_dir:
        summary = {
            "agent": agent.AGENT_TYPE,
            "n_episodes": n_total,
            "task_ids": task_ids,
            "success_rate": sr,
            "avg_steps": sum(r.steps for r in results) / n_total,
            "llm_calls": stats["total_calls"],
            "elapsed_sec": elapsed,
        }
        with open(Path(log_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results saved to %s", log_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run τ-Bench AWM preplay agent.")
    parser.add_argument("--env", type=str, default="retail",
                        choices=["retail", "airline"])
    parser.add_argument("--task_split", type=str, default="test")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_provider", type=str, default="openai")
    parser.add_argument("--graph_path", type=str, default=None,
                        help="Path to tool graph JSON file (created if absent).")
    parser.add_argument("--preplay", action="store_true",
                        help="Enable preplay plan injection.")
    parser.add_argument("--record_mode", type=str, default="plan",
                        choices=["plan", "trajectory"],
                        help="Graph update source: 'plan' (preplay order) or 'trajectory' (actual execution).")
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    # ── Build τ-Bench environment ─────────────────────────────────────────────
    try:
        from tau_bench.envs import get_env
    except ImportError:
        raise SystemExit(
            "tau-bench not installed. Install with:\n"
            "  pip install git+https://github.com/sierra-research/tau-bench"
        )

    env = get_env(
        env_name=args.env,
        user_strategy="llm",
        user_model=args.user_model,
        task_split=args.task_split,
        user_provider=args.user_provider,
    )

    n_tasks = len(env.tasks) if hasattr(env, "tasks") else (args.n or 164)
    n = args.n or n_tasks
    task_ids = list(range(args.start, args.start + n))

    # ── Resolve graph path and clean slate for online run ────────────────────
    graph_path = args.graph_path or f"./graph/{args.env}.json"
    Path(graph_path).unlink(missing_ok=True)

    # ── Build LLM + agent ────────────────────────────────────────────────────
    from agent import AWMAgent

    llm = TauBenchLLMClient(LLMConfig(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
    ))
    agent = AWMAgent(
        llm=llm,
        preplay_graph_path=graph_path,
        site=args.env,
        use_preplay=args.preplay,
        record_mode=args.record_mode,
    )

    log_dir = args.log_dir or f"./logs/awm_{args.env}"
    logger.info(
        "Starting evaluation: env=%s  n=%d  preplay=%s  log_dir=%s",
        args.env, len(task_ids), args.preplay, log_dir,
    )

    evaluate(
        agent=agent,
        env=env,
        task_ids=task_ids,
        max_steps=args.steps,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()

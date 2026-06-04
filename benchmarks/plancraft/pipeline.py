"""
ProPlay evaluation pipeline for PlanCraft.

Usage:
    python pipeline.py \\
        --model gpt-4.1-mini \\
        --api_base https://api.openai.com/v1 \\
        [--workflow_path workflow/proplay.txt] \\
        [--preplay_graph_path workflow/proplay_graph.json] \\
        [--log_dir ./logs/proplay]
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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE / "agents") not in sys.path:
    sys.path.insert(0, str(_HERE / "agents"))
if str(_HERE / "prompts") not in sys.path:
    sys.path.insert(0, str(_HERE / "prompts"))
if str(_HERE / "envs") not in sys.path:
    sys.path.insert(0, str(_HERE / "envs"))

# LLMClient from the parent project utilities
# memory_planning_demo deps moved into proplay package

from proplay.llm import LLMClient, LLMConfig
from router import PlancraftEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    episode_idx: int
    target: str
    reward: float
    steps: int
    success: bool
    n_workflows: int = 0
    plan_reliability: float = 0.0
    plan_length: int = 0
    preplay_plan: list = None


def _make_agent(args, llm):
    from agent import ProPlayAgent
    return ProPlayAgent(
        llm=llm,
        workflow_path=args.workflow_path,
        preplay_graph_path=args.preplay_graph_path,
        use_preplay=args.use_preplay,
        use_reliability=not args.no_reliability,
    )


def run_episode(agent, env: PlancraftEnv, episode_idx: int, max_steps: int) -> EpisodeResult:
    env.reset(episode_idx)
    agent.reset()

    task = env.task_description
    steps_taken = 0

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

        # Store post-action observation in trajectory
        if agent._trajectory:
            agent._trajectory[-1]["result_obs"] = result.observation
            agent._trajectory[-1]["reward"] = result.reward

        logger.info("  step=%d  action=%r  reward=%.3f", step, action[:80], result.reward)
        steps_taken = step + 1

        if result.done:
            break

    final_reward = env.reward
    success = env.success

    try:
        agent.finalize_episode(success=success, task=task, reward=final_reward)
    except Exception as exc:
        logger.warning("finalize_episode failed: %s", exc)

    n_wf = _count_workflows(getattr(agent, 'workflow_path', None))
    plan_rel = getattr(agent, '_preplay_plan_reliability', 0.0)
    plan = list(getattr(agent, '_preplay_plan', []))

    return EpisodeResult(
        episode_idx=episode_idx,
        target=env.target,
        reward=final_reward,
        steps=steps_taken,
        success=success,
        n_workflows=n_wf,
        plan_reliability=plan_rel,
        plan_length=len(plan),
        preplay_plan=plan,
    )


def evaluate(
    agent,
    env: PlancraftEnv,
    max_steps: int = 30,
    log_dir: Optional[str] = None,
) -> None:
    n_total = env.num_episodes()
    results: list[EpisodeResult] = []
    t0 = time.time()

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    for i in range(n_total):
        try:
            ep = run_episode(agent, env, i, max_steps)
        except Exception as exc:
            logger.error("Episode %d failed: %s", i, exc)
            ep = EpisodeResult(
                episode_idx=i, target="?",
                reward=0.0, steps=0, success=False,
            )
        results.append(ep)

        n_done = len(results)
        sr = sum(r.success for r in results) / n_done
        avg_score = sum(r.reward for r in results) / n_done
        logger.info(
            "[%d/%d] ep=%d  target=%r  success=%s  "
            "steps=%d  reward=%.3f  SR=%.3f  avg_score=%.3f  workflows=%d",
            i + 1, n_total, i,
            ep.target,
            "✓" if ep.success else "✗",
            ep.steps, ep.reward, sr, avg_score, ep.n_workflows,
        )

        if log_dir:
            mode = "a" if i > 0 else "w"
            with open(Path(log_dir) / "episodes.jsonl", mode) as f:
                f.write(json.dumps(asdict(ep)) + "\n")

    elapsed = time.time() - t0
    sr = sum(r.success for r in results) / n_total

    try:
        stats = agent.llm.stats()
    except Exception:
        stats = {"total_calls": "N/A"}

    # Reliability analysis (ProPlay only)
    rel_eps = [r for r in results if r.plan_length >= 2]

    print("\n" + "=" * 60)
    print(f"  Agent:              {agent.AGENT_TYPE}")
    print(f"  Episodes:           {n_total}")
    print(f"  Success rate:       {sr:.3f}  ({sr*100:.1f}%)")
    print(f"  Avg steps:          {sum(r.steps for r in results)/n_total:.1f}")
    print(f"  Avg reward:         {sum(r.reward for r in results)/n_total:.3f}")
    print(f"  LLM calls:          {stats.get('total_calls', 'N/A')}")
    print(f"  Elapsed:            {elapsed:.1f}s")

    if rel_eps:
        rels = sorted(r.plan_reliability for r in rel_eps)
        n = len(rels)
        threshold = (rels[n // 2 - 1] + rels[n // 2]) / 2 if n % 2 == 0 else rels[n // 2]
        high = [r for r in rel_eps if r.plan_reliability >= threshold]
        low  = [r for r in rel_eps if r.plan_reliability <  threshold]
        avg_rel = sum(r.plan_reliability for r in rel_eps) / len(rel_eps)
        print(f"  ── Reliability analysis (plan_length≥2, n={len(rel_eps)}) ──")
        print(f"  Avg plan reliability:  {avg_rel:.3f}")
        print(f"  Threshold (median):    {threshold:.3f}")
        if high:
            print(f"  High-rel  (n={len(high):3d}):  SR={sum(r.success for r in high)/len(high):.3f}  avg_rel={sum(r.plan_reliability for r in high)/len(high):.3f}")
        if low:
            print(f"  Low-rel   (n={len(low):3d}):  SR={sum(r.success for r in low)/len(low):.3f}  avg_rel={sum(r.plan_reliability for r in low)/len(low):.3f}")

    print("=" * 60 + "\n")

    if log_dir:
        summary = {
            "agent": agent.AGENT_TYPE,
            "n_total": n_total,
            "success_rate": sr,
            "avg_steps": sum(r.steps for r in results) / n_total,
            "avg_reward": sum(r.reward for r in results) / n_total,
            "llm_calls": stats.get("total_calls", "N/A"),
            "elapsed_sec": elapsed,
        }
        with open(Path(log_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results saved to %s", log_dir)


def _count_workflows(workflow_path: Optional[str]) -> int:
    if not workflow_path:
        return 0
    import re
    path = Path(workflow_path)
    if not path.exists():
        return 0
    return len(re.findall(r'^Workflow\s+\d+:', path.read_text(), re.MULTILINE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  type=str,  default="splits/merged_187_by_complexity.json")
    parser.add_argument("--steps",      type=int,  default=30)
    parser.add_argument("--model",      type=str,  default="gpt-4.1-mini")
    parser.add_argument("--api_base",   type=str,  default="https://api.openai.com/v1")
    parser.add_argument("--api_key",    type=str,  default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max_tokens", type=int,  default=1024)
    # ProPlay-specific
    parser.add_argument("--workflow_path",      type=str, default="workflow/proplay.txt")
    parser.add_argument("--preplay_graph_path", type=str, default="workflow/proplay_graph.json")
    parser.add_argument("--use_preplay",   action="store_true", default=True)
    parser.add_argument("--no_preplay",    action="store_true",
                        help="Disable preplay (ProPlay only — use workflow library without graph)")
    parser.add_argument("--no_reliability", action="store_true",
                        help="Hide edge reliability from preplay LLM (ablation)")
    parser.add_argument("--log_dir",    type=str,  default="./logs/proplay")
    args = parser.parse_args()

    if args.no_preplay:
        args.use_preplay = False

    llm = LLMClient(LLMConfig(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
    ))
    env = PlancraftEnv(data_path=args.data_path, max_steps=args.steps)
    agent = _make_agent(args, llm)

    evaluate(
        agent=agent,
        env=env,
        max_steps=args.steps,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()

"""
TAU-bench environment router for retail and airline domains.

Wraps the standard tau_bench.envs.get_env() interface and exposes
a unified env object compatible with the ProPlay agent:

    env.tasks            — list of Task objects
    env.tools_info       — OpenAI-format tool definitions
    env.reset(idx)       — initialise episode, return initial observation
    env.step(action)     — execute action, return (observation, done)
    env.calculate_reward() — compute final episode reward

Installation
------------
    pip install git+https://github.com/sierra-research/tau-bench

Supported domains: "retail", "airline"
(Telecom domain is excluded from this release.)
"""
from __future__ import annotations

SUPPORTED_DOMAINS = ("retail", "airline")


def load_env(
    domain: str,
    *,
    task_split: str = "test",
    user_model: str = "gpt-4o-mini",
    user_provider: str = "openai",
    user_strategy: str = "llm",
):
    """
    Load a tau-bench environment for the given domain.

    Parameters
    ----------
    domain       : "retail" or "airline"
    task_split   : "train", "test", or "dev"
    user_model   : LLM model string for the user simulator
    user_provider: API provider for user_model
    user_strategy: "llm" (simulated user) or "fixed" (scripted)

    Returns
    -------
    tau_bench Env with .tasks, .reset(), .step(), .calculate_reward()
    """
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(
            f"domain must be one of {SUPPORTED_DOMAINS}, got {domain!r}\n"
            "Note: telecom domain is not included in this release."
        )

    try:
        from tau_bench.envs import get_env
    except ImportError:
        raise SystemExit(
            "tau-bench not installed.\n"
            "Install with:  pip install git+https://github.com/sierra-research/tau-bench"
        )

    return get_env(
        env_name=domain,
        user_strategy=user_strategy,
        user_model=user_model,
        task_split=task_split,
        user_provider=user_provider,
    )

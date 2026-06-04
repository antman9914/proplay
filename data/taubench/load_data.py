"""
TAU-bench data loading utilities for retail and airline domains.

Requires the tau-bench package:
    pip install git+https://github.com/sierra-research/tau-bench

Usage
-----
    from proplay.data.taubench.load_data import load_env, get_task_ids

    env = load_env("retail", task_split="test", user_model="gpt-4o-mini")
    task_ids = get_task_ids(env, n=100)
"""
from __future__ import annotations

import os
from typing import Optional


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
    Load a TAU-bench environment for the given domain.

    Parameters
    ----------
    domain       : "retail" or "airline"
    task_split   : "train", "test", or "dev"
    user_model   : LLM model string for the user simulator
    user_provider: Provider for user_model (e.g. "openai")
    user_strategy: "llm" (simulated user) or "fixed" (scripted user)

    Returns
    -------
    tau_bench Env object with .tasks list and .reset() / .step() interface.
    """
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"domain must be one of {SUPPORTED_DOMAINS}, got {domain!r}")

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


def get_task_ids(env, *, n: Optional[int] = None, start: int = 0) -> list[int]:
    """
    Return a list of task indices to evaluate.

    Parameters
    ----------
    env  : tau_bench Env object (must have a .tasks attribute)
    n    : number of tasks; defaults to all tasks in the split
    start: starting index offset
    """
    n_total = len(env.tasks) if hasattr(env, "tasks") else 0
    n = n or n_total
    return list(range(start, start + n))


def domain_info(domain: str) -> dict:
    """
    Return metadata about a supported domain.

    Returns a dict with keys:
      name         : human-readable domain name
      default_n    : typical number of test tasks
      tools_note   : brief description of available tools
    """
    _INFO = {
        "retail": {
            "name": "Retail",
            "default_n": 500,
            "tools_note": (
                "Order lookup, product search, cancel/return/exchange, "
                "payment method management, address update."
            ),
        },
        "airline": {
            "name": "Airline",
            "default_n": 300,
            "tools_note": (
                "Flight search, booking, seat selection, upgrade, "
                "cancellation, baggage policy lookup."
            ),
        },
    }
    if domain not in _INFO:
        raise ValueError(f"domain must be one of {SUPPORTED_DOMAINS}, got {domain!r}")
    return _INFO[domain]

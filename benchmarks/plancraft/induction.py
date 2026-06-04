"""
Batch recipe induction for PlanCraft (ProPlay).

After each qualifying episode, a compact summary is appended to the episode list.
All summaries + the existing workflow library are passed to the LLM, which:
  1. Updates/creates recipe workflow entries (Workflow N: Name).
  2. Outputs a <trace> of the recipe sequence executed in the latest episode.

Using compact episode summaries (~80-120 tokens each) keeps token cost linear.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from prompts.proplay_prompts import (
    INDUCTION_INSTRUCTION,
    INDUCTION_ONE_SHOT,
    INDUCTION_USER,
)

logger = logging.getLogger(__name__)


def format_examples(episode_summaries: list[str]) -> str:
    """Format all episode summaries for the induction prompt."""
    parts = []
    for i, summary in enumerate(episode_summaries, 1):
        parts.append(f"Past Episode Summary (Episode {i}):\n{summary.strip()}")
    return "\n\n---\n\n".join(parts)


def build_episode_summary(
    task: str,
    trajectory: list[dict],
    reward: float,
) -> str:
    """
    Build a compact episode summary from the trajectory.

    Format:
      Target: <item>
      Steps taken:
        <action>  (<inferred item context>)
      Reward: <reward>
    """
    target_m = re.search(r'type:\s*(\S+)', task)
    target = target_m.group(1) if target_m else task

    lines = [f"Target: {target}", "Steps taken:"]
    for t in trajectory:
        action = t.get("action") or t.get("raw_action", "")
        if not action:
            continue
        lines.append(f"  {action}")
    lines.append(f"Reward: {reward:.1f}")
    return "\n".join(lines)


def run_induction(
    llm_chat_fn,
    episode_summaries: list[str],
    workflow_path: str,
) -> tuple[bool, list[str]]:
    """
    Batch-update the recipe workflow library from all accumulated episode summaries.

    Returns (file_updated, execution_trace) where:
      file_updated    — True if the workflow file was written.
      execution_trace — ordered list of workflow names from the latest episode.
    """
    if not episode_summaries:
        return False, []

    existing_text = _read_existing_workflows(workflow_path)
    examples_text = format_examples(episode_summaries)

    prompt = "\n\n".join([
        INDUCTION_INSTRUCTION,
        INDUCTION_ONE_SHOT,
        INDUCTION_USER.format(
            existing_workflows=existing_text if existing_text else "(none yet)",
            examples=examples_text,
        ),
    ])

    induction_log = Path(workflow_path).parent / "induction_history.log"
    with open(induction_log, "a") as f:
        f.write(f"\n{'='*60}\n{examples_text}\n")

    response = llm_chat_fn([{"role": "user", "content": prompt}])
    logger.info("run_induction: LLM response (first 500 chars): %r", response[:500])

    workflows_text = _extract_workflows_section(response)
    if not workflows_text:
        logger.warning("run_induction: could not extract workflows from response")
        return False, []

    wf_path = Path(workflow_path)
    wf_path.parent.mkdir(parents=True, exist_ok=True)
    wf_path.write_text("## Summary Workflows\n\n" + workflows_text)

    execution_trace = _extract_trace(response, workflows_text)
    if execution_trace:
        logger.info("run_induction: execution trace: %s", execution_trace)
    else:
        logger.info("run_induction: no execution trace found")

    return True, execution_trace


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_existing_workflows(workflow_path: str) -> str:
    path = Path(workflow_path)
    if not path.exists():
        return ""
    content = path.read_text()
    m = re.search(r'##\s*Summary Workflows\s*\n', content, re.I)
    if m:
        return content[m.end():].strip()
    return ""


def _extract_workflows_section(text: str) -> str:
    text_no_trace = re.sub(r'<trace>.*?</trace>', '', text, flags=re.DOTALL)
    lines = text_no_trace.splitlines()
    start = None
    pattern = re.compile(r'Workflow\s+\d+\s*:', re.IGNORECASE)
    for i, line in enumerate(lines):
        if pattern.search(line):
            start = i
            break
    if start is None:
        return ""
    clean = []
    for line in lines[start:]:
        line = re.sub(r'^[#*]+\s*', '', line)
        line = re.sub(r'\s*[*]+$', '', line)
        clean.append(line)
    return "\n".join(clean).strip()


def _extract_trace(response: str, workflows_text: str) -> list[str]:
    trace_m = re.search(r'<trace>(.*?)</trace>', response, re.DOTALL)
    if not trace_m:
        return []

    valid_names: dict[str, str] = {}
    for m in re.finditer(r'Workflow\s+\d+\s*:\s*(.+)', workflows_text):
        canonical = re.sub(r'\*+$', '', m.group(1).strip()).strip()
        valid_names[canonical.lower()] = canonical

    trace: list[str] = []
    seen: set[str] = set()
    for raw_line in trace_m.group(1).splitlines():
        name = re.sub(r'^\s*(\d+[.)]\s*|[-*]\s*)', '', raw_line).strip()
        if not name:
            continue
        canonical = valid_names.get(name.lower())
        if canonical is None:
            logger.info("run_induction: trace entry %r not in library — skipped", name)
            continue
        if canonical not in seen:
            trace.append(canonical)
            seen.add(canonical)
    return trace

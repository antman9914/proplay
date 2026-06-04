"""
Batch workflow induction for τ-Bench (AWM-style).

Logic is identical to sciworld_v2/induce.py. Only the imports differ: prompts
come from the local prompts.py which has τ-Bench-specific induction templates.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from prompts import (
    INDUCTION_INSTRUCTION,
    INDUCTION_ONE_SHOT,
    INDUCTION_USER,
    format_examples,
)

logger = logging.getLogger(__name__)


def run_induction(
    llm_chat_fn,
    episode_summaries: list[str],
    workflow_path: str,
) -> tuple[bool, list[str]]:
    """
    Batch-update the workflow library from all accumulated episode summaries.

    Returns (file_updated, execution_trace) where:
      file_updated     — True if the workflow file was written.
      execution_trace  — ordered list of workflow names matching the latest
                         episode's actual executed steps.
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

    induction_log = Path(workflow_path).with_suffix("").parent / "induction_history.log"
    with open(induction_log, "a") as _f:
        _f.write(f"\n{'='*60}\n{examples_text}\n")
    logger.debug("run_induction: episode summaries written to %s", induction_log)

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
        logger.info("run_induction: execution trace for latest episode: %s", execution_trace)
    else:
        logger.info("run_induction: no execution trace found in response")

    return True, execution_trace


# ── Internal helpers (identical to sciworld_v2/induce.py) ─────────────────────

def _read_existing_workflows(workflow_path: str) -> str:
    path = Path(workflow_path)
    if not path.exists():
        return ""
    content = path.read_text()
    summary_match = re.search(r'##\s*Summary Workflows\s*\n', content, re.I)
    if summary_match:
        return content[summary_match.end():].strip()
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
    trace_match = re.search(r'<trace>(.*?)</trace>', response, re.DOTALL)
    if not trace_match:
        return []

    valid_names: dict[str, str] = {}
    for m in re.finditer(r'Workflow\s+\d+\s*:\s*(.+)', workflows_text):
        canonical = m.group(1).strip()
        canonical = re.sub(r'\*+$', '', canonical).strip()
        valid_names[canonical.lower()] = canonical

    trace: list[str] = []
    seen: set[str] = set()
    for raw_line in trace_match.group(1).splitlines():
        name = re.sub(r'^\s*(\d+[.)]\s*|[-*]\s*)', '', raw_line).strip()
        if not name:
            continue
        key = name.lower()
        canonical = valid_names.get(key)
        if canonical is None:
            logger.info(
                "run_induction: trace entry %r not found in updated library — skipped", name
            )
            continue
        if canonical not in seen:
            trace.append(canonical)
            seen.add(canonical)

    return trace

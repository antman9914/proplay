"""
Pre-play mechanism for recipe-level planning in PlanCraft.

Before the agent acts, a lightweight LLM call traverses the recipe dependency
graph and generates an ordered recipe execution sequence (sub-recipes first,
then the target recipe). The plan is injected into the agent's system prompt.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from workflow_graph import WorkflowGraph


_INSTRUCTION = (Path(__file__).parent / "prompt" / "preplay_instruction.txt").read_text().strip()
_ONE_SHOT    = (Path(__file__).parent / "prompt" / "preplay_one_shot.txt").read_text().strip()
_SYSTEM_MSG  = _INSTRUCTION + "\n\n" + _ONE_SHOT


def run_preplay(
    llm_chat_fn,
    goal: str,              # e.g. "Craft an item of type: wooden_pickaxe"
    site: str,              # e.g. "plancraft"
    graph: WorkflowGraph,
    use_reliability: bool = True,
    workflow_content: dict = None,
    task_embedding=None,
    failed_experiences: List[str] = None,
) -> tuple:
    """
    Run a single LLM call to pre-play a recipe sequence for the task.

    Returns (recipe_names: List[str], plan_text: str).
      recipe_names — ordered list of recipe names (top-level plan entries only).
      plan_text    — full structured plan for agent injection.
    """
    nodes = graph.get_nodes_for_site(site)
    if not nodes:
        return [], ""

    user_msg = _build_user_msg(
        goal, site, graph, nodes,
        use_reliability=use_reliability,
        workflow_content=workflow_content,
        task_embedding=task_embedding,
        failed_experiences=failed_experiences or [],
    )
    messages = [
        {"role": "system", "content": _SYSTEM_MSG},
        {"role": "user",   "content": user_msg},
    ]
    response = llm_chat_fn(messages)
    return _parse_plan(response)


def format_plan_for_injection(
    plan: List[str],
    plan_text: str = None,
    workflow_content: dict = None,
) -> str:
    """
    Format the pre-played plan for injection into the agent's system prompt.
    """
    if not plan:
        return ""

    header = (
        "Suggested recipe plan for this task (from past crafting experience):\n"
        "Follow these steps as a guide; adjust based on your current inventory."
    )

    if plan_text:
        return f"{header}\n\n{plan_text.strip()}"

    # Fallback: reconstruct from recipe names + generic content
    lines = [header]
    for i, recipe_name in enumerate(plan):
        name_key = recipe_name.strip().lower()
        lines.append(f"\nStep {i + 1}: {recipe_name}")
        if workflow_content:
            steps = workflow_content.get(name_key)
            if steps:
                for step_line in steps.strip().splitlines():
                    lines.append(f"  {step_line}")
    return "\n".join(lines)


def record_execution(
    plan: List[str],
    graph: WorkflowGraph,
    site: str,
    reward: float,
    plan_ids: Optional[List[str]] = None,
    task_embedding=None,
) -> None:
    """
    Record the planned recipe transition sequence in the graph after an episode.
    """
    if len(plan) < 2:
        return

    if plan_ids and len(plan_ids) == len(plan):
        for i in range(len(plan_ids) - 1):
            src_id = plan_ids[i]
            tgt_id = plan_ids[i + 1]
            if src_id and tgt_id and src_id in graph.nodes and tgt_id in graph.nodes:
                graph.record_transition(
                    src_id, tgt_id, score=reward, task_embedding=task_embedding
                )
        return

    nodes_by_name = {n.name.lower().strip(): n for n in graph.get_nodes_for_site(site)}
    for i in range(len(plan) - 1):
        src = nodes_by_name.get(plan[i].strip().lower())
        tgt = nodes_by_name.get(plan[i + 1].strip().lower())
        if src and tgt:
            graph.record_transition(
                src.node_id, tgt.node_id, score=reward, task_embedding=task_embedding
            )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_user_msg(
    goal, site, graph, nodes,
    use_reliability: bool = True,
    workflow_content: dict = None,
    task_embedding=None,
    failed_experiences: List[str] = None,
) -> str:
    # Recipe list with full steps when available
    recipe_lines = []
    for node in nodes:
        recipe_lines.append(f"### {node.name}")
        if workflow_content:
            steps = workflow_content.get(node.name.lower().strip())
            if steps:
                recipe_lines.append(steps.strip())
        recipe_lines.append("")
    recipe_section = "\n".join(recipe_lines).strip() or "(no recipes available yet)"

    # Transitions sorted by task-specific reliability descending
    if use_reliability:
        def _score(edge):
            if task_embedding is not None:
                return graph.get_task_reliability(edge, task_embedding)
            return edge.reliability

        scored = [(edge, _score(edge)) for edge in graph.edges.values()]
        scored.sort(key=lambda x: (x[1], x[0].count), reverse=True)
        trans_lines = []
        for edge, score in scored:
            src = graph.nodes.get(edge.source_id)
            tgt = graph.nodes.get(edge.target_id)
            if not (src and tgt and src.site == site):
                continue
            trans_lines.append(
                f"- '{src.name}' → '{tgt.name}' "
                f"(relevance: {score:.2f}, {edge.count} trials)"
            )
        trans_section = "\n".join(trans_lines) or "(no prior recipe sequences recorded)"
    else:
        trans_section = "(reliability scores hidden — ablation mode)"

    # Failed episode context
    fail_section = ""
    if failed_experiences:
        fail_lines = "\n\n".join(failed_experiences)
        fail_section = (
            f"\nPast Episode Experiences\n"
            f"Episodes below contain failed steps. Study them to understand what went wrong "
            f"and design the current plan to avoid those issues.\n"
            f"{fail_lines}"
        )

    footer = (
        "\n\nGenerate a recipe execution plan for this task. "
        "Use the graph as evidence — follow known sequences where they apply, "
        "and order sub-recipes before the recipes that require them."
    )

    return f"""\
Task Goal
{goal}

Available Recipes
{recipe_section}

Known Recipe Transitions (from past experience)
Sorted by relevance to this task (higher = worked well on similar targets).
{trans_section}{fail_section}{footer}

<plan>
"""


def _parse_plan(response: str) -> tuple:
    """
    Extract plan from a <plan>...</plan> block.
    Returns (names, plan_text).
    """
    plan_match = re.search(r'<plan>(.*?)(?:</plan>|$)', response, re.DOTALL)
    if not plan_match:
        return [], ""
    plan_text = plan_match.group(1)
    names = []
    for line in plan_text.splitlines():
        m = re.match(r'^\d+[.)]\s+(.+)', line)
        if m:
            names.append(m.group(1).strip())
    return names, plan_text.strip()

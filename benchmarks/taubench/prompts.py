"""All LLM prompts for the τ-Bench AWM preplay agent."""

# ── System base (τ-Bench customer-service context) ────────────────────────────

SYSTEM_BASE = """\
You are a customer service agent for an online company. Help customers resolve
their requests by using the available tools to look up information and take
actions on their behalf.

Guidelines:
1. Use query tools first (find_user_id, get_order_details, get_product_details,
   etc.) to understand the customer's situation before making any changes.
2. Confirm you have the correct information before executing irreversible actions
   (cancellations, returns, modifications). If the customer's request is
   unambiguous, act directly.
3. Use the respond tool to communicate with the customer when you need more
   information, to confirm an action, or to notify them of the outcome.
4. Be professional, concise, and accurate.
5. When the customer's request is fully resolved, send a brief confirmation
   via the respond tool.
"""


# ── Batch workflow induction (AWM-style) ─────────────────────────────────────

INDUCTION_INSTRUCTION = """\
You are maintaining a library of abstract workflow templates for a τ-Bench
customer service agent.
Given the current workflow library and all accumulated episode summaries, update
the library to reflect the generalizable patterns seen across these episodes.

Each episode summary contains:
- Actions: the complete tool call sequence for the episode (excluding internal
  think steps; respond calls are included for context).
- Successful steps: all API tool calls for episodes the agent completed
  successfully. Derive workflow patterns from these.
- Failed steps: all API tool calls for episodes the agent failed. Ignore these
  in workflow derivation — they show what went wrong, not what to do.

Update rules:
1. ADD a new workflow if episodes demonstrate a task pattern not covered by
   any existing workflow (e.g. a new type of customer request).
2. ADD steps to an existing workflow if episodes reveal consistently missing steps.
3. REWRITE only if existing steps are outright incorrect.
4. Use abstract placeholders (e.g. <order_id>, <item_id>, <user_id>, <amount>).
5. Do not add conditional branches that only apply to a single specific episode.

Output two sections in order:

Section 1 — the complete workflow library (all existing + any new or modified
workflows) in the same numbered format as the input.

Section 2 — an execution trace for the LATEST episode only (the last episode
listed). Identify the ordered sequence of workflows from the UPDATED library
that best describe what the agent actually did. Wrap in <trace> tags, one
workflow name per line.

Rules for the trace:
- Use only workflow names from the updated library.
- List only workflows that match steps the agent actually executed.
- Preserve execution order.\
"""

INDUCTION_ONE_SHOT = """\
## Existing Workflows

Workflow 1: Look Up Customer and Order Information
Purpose: Retrieve customer identity and order details as a prerequisite for any action.
Steps:
1. find_user_id_by_email(<email>) or find_user_id_by_name_zip(<name>, <zip>)
2. get_user_details(<user_id>) to verify account details
3. get_order_details(<order_id>) if an order is mentioned

## Episodes

Episode 1:
Task: Hi, I want to cancel my most recent order. My email is alice@example.com.
Actions: find_user_id_by_email(email='alice@example.com') → get_user_details(user_id='U001') → get_order_details(order_id='W1001') → cancel_order(order_id='W1001') → respond(content='Your order W1001 has been cancelled.')
Successful steps: find_user_id_by_email(email='alice@example.com') → get_user_details(user_id='U001') → get_order_details(order_id='W1001') → cancel_order(order_id='W1001')

## Updated Workflows

Workflow 1: Look Up Customer and Order Information
Purpose: Retrieve customer identity and order details as a prerequisite for any action.
Steps:
1. find_user_id_by_email(<email>) or find_user_id_by_name_zip(<name>, <zip>)
2. get_user_details(<user_id>) to verify account details
3. get_order_details(<order_id>) if an order is mentioned

Workflow 2: Cancel Order
Purpose: Cancel a pending or processing order on behalf of the customer.
Steps:
1. Verify order status via get_order_details(<order_id>)
2. cancel_order(<order_id>) if status is pending or processing
3. respond to the customer with confirmation

<trace>
Look Up Customer and Order Information
Cancel Order
</trace>\
"""

INDUCTION_USER = """\
## Existing Workflows

{existing_workflows}

## Episodes

{examples}

## Updated Workflows\
"""


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_workflow_section(workflow_text: str) -> str:
    if not workflow_text.strip():
        return ""
    return f"Workflow memory from past experience:\n{workflow_text}\n\n"


def format_plan_section(plan: list[str]) -> str:
    """Inject the preplay tool plan into the system prompt."""
    if not plan:
        return ""
    seq = " → ".join(plan)
    return (
        f"Suggested tool sequence based on similar past tasks:\n"
        f"  {seq}\n"
        f"Follow this sequence as a starting guide; adapt if the task requires it."
    )


# ── Tool-graph preplay prompts ────────────────────────────────────────────────

PREPLAY_SYSTEM = """\
You are a planning assistant for a customer service agent.
Given a customer task and historical tool-use data, output a concise plan as an
ordered JSON list of tool names the agent should call.

Rules:
- Use ONLY tool names from the provided list.
- Output exactly one JSON array, e.g. ["tool_a", "tool_b", "tool_c"].
- Omit think and respond — only include data/action API tools.
- Keep the plan to 3-7 tools; skip obvious lookups only if the task clearly doesn't need them.
- If similar past examples exist, prefer sequences that worked before.\
"""

PREPLAY_USER = """\
Task: {task}

Available tools: {tool_names}

Most frequent tool transitions in past successful episodes:
{top_transitions}

Similar past tasks completed successfully:
{success_examples}

Similar past tasks that FAILED (patterns to avoid):
{fail_examples}

Output the planned tool sequence as a JSON array of tool names.\
"""


def format_history(history: list[dict]) -> str:
    """Format recent tool-call history for injection into the agent's context."""
    if not history:
        return "  (none yet)"
    lines = []
    for i, h in enumerate(history):
        action = h.get("action", "")
        result = h.get("result", "")
        lines.append(f"  {i+1}. {action}")
        if result:
            lines.append(f"     → {result[:200]}")
    return "\n".join(lines)


def parse_memory(response: str) -> str:
    """Extract content from <memory>...</memory> tags (not used in τ-Bench)."""
    import re
    m = re.search(r'<memory>(.*?)</memory>', response, re.DOTALL)
    return m.group(1).strip() if m else ""


# Meta-actions excluded from Successful/Failed steps (but kept in Actions).
_SKIP_TOOLS = frozenset({"think", "respond"})


def format_episode_summary(task: str, trajectory: list[dict]) -> str:
    """
    Compact summary of one τ-Bench episode.

    τ-Bench uses binary rewards (0/1) set only at episode end, so mid-episode
    reward splits are not meaningful. The full tool-call sequence is treated
    uniformly: all non-meta steps are Successful steps on success, or Failed
    steps on failure.

    - Actions: all tool calls except think (includes respond for context).
    - Successful/Failed steps: all tool calls except think and respond.
    """
    # Determine episode outcome from any non-zero reward in trajectory.
    episode_reward = max((t.get("reward") or 0.0) for t in trajectory) if trajectory else 0.0
    success = episode_reward > 0

    # Actions line: full sequence minus think (respond stays for readability).
    action_steps = [
        t["action"] for t in trajectory
        if t.get("tool_name", "").lower() != "think"
    ]

    # Successful/Failed steps: substantive API calls only.
    api_steps = [
        t["action"] for t in trajectory
        if t.get("tool_name", "").lower() not in _SKIP_TOOLS
    ]

    task_short = task[:300].replace("\n", " ")
    lines = [f"Task: {task_short}"]
    lines.append("Actions: " + " → ".join(action_steps) if action_steps else "Actions: (none)")
    if success:
        if api_steps:
            lines.append("Successful steps: " + " → ".join(api_steps))
    else:
        if api_steps:
            lines.append("Failed steps: " + " → ".join(api_steps))
    return "\n".join(lines)


def extract_failed_steps_summary(summary: str) -> str:
    """Return a compact 'Task + Failed steps' snippet for preplay context.

    Returns empty string when the summary has no Failed steps line.
    """
    lines = summary.splitlines()
    task_line = next((l for l in lines if l.startswith("Task:")), "")
    failed_line = next((l for l in lines if l.startswith("Failed steps:")), "")
    if not failed_line:
        return ""
    return f"{task_line}\n{failed_line}"


def format_examples(episode_summaries: list[str]) -> str:
    """Join accumulated episode summaries for batch induction."""
    return "\n\n".join(
        f"Episode {i + 1}:\n{s}" for i, s in enumerate(episode_summaries)
    )

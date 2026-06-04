"""All LLM prompts for the ScienceWorld AWM agent."""

# ── Step-level action selection ───────────────────────────────────────────────

SYSTEM_BASE = """\
You are an autonomous agent in ScienceWorld, a text-based science experiment
environment. You will be given science tasks that require interacting with
objects and locations across virtual rooms.

## ScienceWorld Action Reference

Actions follow these patterns — use exact object and location names from the Observation:

  open <door/container>         — open a door or container lid
  go to <location>              — move to an adjacent room through an open door
  look around                   — describe all objects visible in the current room
  look at <object>              — examine a specific object
  look in <container>           — look inside a container
  pick up <object>              — add object to inventory
  put down <object>             — drop object from inventory
  move <object> to <location>   — move object to a room or container
  activate <device>             — turn on a device (stove, sink, stopwatch, etc.)
  deactivate <device>           — turn off a device
  use <tool> on <object>        — apply a tool to an object (e.g. thermometer,
                                  lighter, screwdriver)
  pour <substance> into <container> — transfer liquid/substance
  dunk <container> into <liquid>    — submerge container in liquid
  mix <container>               — stir/mix the substance in a container
  connect <object> to <target>  — connect object's electrical terminal to target;
                                  confirmed with "terminal N on <object> is now
                                  connected to terminal N on <target>"
  disconnect <object>           — disconnect object's electrical terminals
  close <door/container>        — close a door or container lid
  focus on <object>             — signal task intent on an object (used for
                                  task scoring; may also reveal its properties)
  inventory                     — list items currently held by the agent
  read <document>               — read the text of a document or label
  eat <food>                    — consume a food item
  wait / wait1                  — let one time step pass (useful while heating)
  task                          — re-display the current task description

## Response Format

The current Observation is the direct result of your last action.
In your <think>: describe what effect your last action had (success or failure),
then reason about what to do next. If an action did not work, do not repeat it.

After thinking, output one action in <action> tags. Use the Action Reference
patterns above and fill in exact object or location names from the Observation.
Update your working notes in <memory> tags.

<think>
My last action was X. The result shows Y. Therefore I will Z.
</think>
<action>
verb exact-object-name [to exact-target-name]
</action>
<memory>
Key facts found so far: location of objects, device states, measurements.
</memory>

## Concrete Example

Task: Measure the melting point of ice.
Step 1 — Observation: In the hallway you see: the agent, a picture. You also
see: A door to the kitchen (that is closed). A door to the bedroom (that is
closed).
<think>
This is the first step. I need to find ice and a thermometer. I can see a door
to the kitchen. I will open it and go in.
</think>
<action>
open door to kitchen
</action>
<memory>
Hallway: picture, door to kitchen (closed), door to bedroom (closed).
Opening kitchen door next.
</memory>

Step 2 — Observation: The door to the kitchen is now open.
<think>
My last action 'open door to kitchen' succeeded. Now I can enter the kitchen.
</think>
<action>
go to kitchen
</action>
<memory>
Kitchen door open. Moving to kitchen.
</memory>

Step 3 — Observation: You move to the kitchen.
<think>
My last action 'go to kitchen' succeeded — I am now in the kitchen. I should
look around to find what is here.
</think>
<action>
look around
</action>
<memory>
Now in kitchen. Looking around for ice and thermometer.
</memory>

Step 4 — Observation: In the kitchen you see: a fridge, a stove, a thermometer
on the counter, a sink.
<think>
My last action 'look around' succeeded. I can see a thermometer on the counter.
Ice is probably in the fridge. I will pick up the thermometer first.
</think>
<action>
pick up thermometer
</action>
<memory>
In kitchen: fridge, stove, thermometer (counter), sink. Thermometer picked up.
Ice likely in fridge.
</memory>

If the task is fully complete, write:
<action>
done
</action>\
"""

STEP_USER = """\
Task: {task}

Observation:
{observation}

Known room state (objects seen in this room — may be stale if objects moved):
{room_state}

Working notes (your memory from previous steps):
{memory}

Recent history (last {history_len} steps):
{history}

{workflow_section}What do you do next?\
"""

# ── Candidate selection (second LLM call) ────────────────────────────────────

CANDIDATE_SYSTEM = """\
Select the single best matching action from the candidate list that matches the
intended action given the current observation. Output only the chosen action,
copied exactly as written in the list.\
"""

CANDIDATE_USER = """\
Intended action: {intent}

Current observation:
{observation}

Candidate actions:
{candidates}

Which candidate best matches the intended action?\
"""


# ── Incremental workflow induction ───────────────────────────────────────────

INCREMENTAL_INSTRUCTION = """\
You maintain a library of reusable workflows for a science experiment agent.
After each successful episode you are given:
  - The existing workflow library (may be empty on the first episode).
  - One new completed episode (task + action trajectory).

Your job: update the library to incorporate knowledge from the new episode.
Output ONLY the changes needed — do not reprint unchanged workflows.

Output format (use exactly these tags):
  UPDATE: Workflow N: <name>
  <revised full workflow text>

  NEW: Workflow <name>
  <new workflow text>

  NO_CHANGES

Rules:
1. STRONGLY prefer UPDATE over NEW. If the new episode is even loosely related
   to an existing workflow, UPDATE that workflow to make it more general (use
   placeholders, add alternative steps). Do NOT create a new workflow just
   because the task used a different object or location.
2. Use NEW only when the episode contains a genuinely distinct sub-routine with
   no overlap with any existing workflow (different goal, different verb pattern).
3. Never rename, delete, or re-number existing workflows.
4. Each workflow must have at least two steps.
5. Use descriptive placeholders for non-fixed elements
   (e.g. <substance>, <container>, <location>, <heat source>).
6. If the episode adds nothing new, output exactly: NO_CHANGES\
"""

INCREMENTAL_ONE_SHOT = """\
## Existing Workflows

Workflow 1: Navigate to location
1. go to <location>
2. look around

Workflow 2: Heat substance on stove
1. pick up <container>
2. Place <substance> in <container> if needed.
3. move <container> to stove
4. activate stove

## New Episode

Task: Measure the boiling point of water.
Actions:
<think>I need a thermometer and a container of water.</think>
<action>go to kitchen</action>
<action>look around</action>
<action>pick up thermometer</action>
<action>pick up metal pot</action>
<action>pour water into metal pot</action>
<action>move metal pot to stove</action>
<action>activate stove</action>
<action>wait1</action>
<action>use thermometer on metal pot</action>
<action>focus on thermometer</action>

## Changes

UPDATE: Workflow 2: Heat substance on stove
1. pick up <container>
2. Place <substance> in <container> if needed.
3. move <container> to stove
4. activate stove
5. wait1 (repeat until target temperature reached)

NEW: Workflow Measure temperature with thermometer
1. pick up thermometer
2. use thermometer on <object>
3. focus on thermometer to record the reading.\
"""

INCREMENTAL_USER = """\
## Existing Workflows

{existing_workflows}

## New Episode

Task: {task}
Actions:
{trajectory}

## Changes
\
"""


# ── Workflow induction (batch, AWM-style) ────────────────────────────────────

INDUCTION_INSTRUCTION = """\
You are maintaining a library of abstract workflow templates for a science task agent.
Given the current workflow library and all accumulated episode summaries, update the
library to reflect the generalizable patterns seen across these episodes.

Each episode summary contains:
- Actions: the complete action sequence for the episode.
- Successful steps: actions from the productive part of the episode. Reward-gaining
  actions are marked with (+delta); other productive manipulation actions are unmarked.
  This line captures what actually worked.
- Failed steps (optional): manipulation actions attempted after the last reward gain,
  marked with [x]. These actions did not contribute to task progress and should NOT
  be incorporated into workflows.

Derive workflow patterns from Successful steps only — ignore Failed steps entirely.\


Update rules:
1. ADD a new workflow if episodes demonstrate a task pattern not covered by any existing workflow.
2. ADD steps to an existing workflow if episodes reveal steps consistently missing from the general procedure.
3. REWRITE an existing workflow only if it contains steps that are outright incorrect or would fail for common cases.
4. Use abstract placeholders (e.g. <substance>, <container>, <location>) — never hard-code specific values seen in individual episodes.
5. Do not add conditional branches that only apply to a single specific episode's scenario.

Output two sections in order:

Section 1 — the complete workflow library (all existing + any new or modified workflows)
in the same numbered format as the input.

Section 2 — an execution trace for the LATEST episode only (the last episode listed).
Look at its Key actions and identify the ordered sequence of workflows from the UPDATED
library that best describe what the agent actually did. Use only workflow names that
appear verbatim in the updated library. Wrap this section in <trace> tags, one workflow
name per line.

Rules for the trace:
- Use only workflow names from the updated library — no paraphrasing or invented names.
- List only workflows that correspond to steps the agent actually executed (based on Key
  actions), not workflows that were planned but not carried out.
- If a key action does not clearly match any workflow, omit it rather than forcing a match.
- Preserve the order in which the workflows were executed.\
"""

INDUCTION_ONE_SHOT = """\
## Existing Workflows

Workflow 1: Navigate to Location and Explore
Purpose: Move to a specified location and look around to find objects or features.
Steps:
1. If door to `<location>` is closed, open door to `<location>`
2. go to `<location>`
3. look around

## Episodes

Episode 1:
Task: Your task is to find a piece of rubber and focus on it.
Actions: go to hallway → open door to workshop → go to workshop → look around → pick up rubber → focus on rubber
Successful steps: go to workshop (+0.05), pick up rubber, focus on rubber (+0.95)

Episode 2:
Task: Your task is to measure the boiling point of water.
Actions: go to kitchen → pick up thermometer → pick up metal pot → pour water into metal pot → move metal pot to stove → activate stove → use thermometer on metal pot → focus on thermometer → wait1 → wait1 → wait1 → wait1 → wait1 → use thermometer on metal pot → wait1
Successful steps: go to kitchen (+0.08), pick up thermometer, pick up metal pot, pour water into metal pot, move metal pot to stove, activate stove (+0.15), use thermometer on metal pot, focus on thermometer (+0.77)
Failed steps: use thermometer on metal pot [x]

## Updated Workflows

Workflow 1: Navigate to Location and Explore
Purpose: Move to a specified location and look around to find objects or features.
Steps:
1. If door to `<location>` is closed, open door to `<location>`
2. go to `<location>`
3. look around

Workflow 2: Find and Identify Object
Purpose: Locate a target object in the environment and optionally focus on it.
Steps:
1. Navigate to the room where `<object>` is likely located (see Workflow 1)
2. look around to locate `<object>`
3. pick up `<object>`
4. If the task explicitly says to "focus on" the object: focus on `<object>` (the exact target named in the task, NOT unrelated objects)
   Otherwise: do NOT use focus — focusing on the wrong object causes task failure (reward=-1)
   Note: focus is a task-scoring signal, not an information-gathering tool; do not use it to "examine" or "learn about" objects

Workflow 3: Measure Property of Substance
Purpose: Heat a substance to a target state, then use a measurement tool to record its properties.
Steps:
1. pick up thermometer and `<container>`
2. place `<substance>` in `<container>` if needed
3. move `<container>` to `<heat source>`
4. activate `<heat source>`
5. wait1 (repeat until target temperature or state reached)
6. use thermometer on `<container>`
7. focus on thermometer to record the reading

<trace>
Navigate to Location and Explore
Measure Property of Substance
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


def format_history(history: list[dict]) -> str:
    if not history:
        return "  (none yet)"
    lines = []
    for i, h in enumerate(history):
        result = h.get("result_obs", "")
        raw = h.get("raw_action", "")
        executed = h["action"]
        if raw and raw != executed:
            lines.append(f"  {i+1}. intent='{raw}'  →  executed='{executed}'")
        else:
            lines.append(f"  {i+1}. action='{executed}'")
        if result:
            lines.append(f"     result: {result}")
    return "\n".join(lines)


def parse_memory(response: str) -> str:
    """Extract content from <memory>...</memory> tags, or return empty string."""
    import re
    m = re.search(r'<memory>(.*?)</memory>', response, re.DOTALL)
    return m.group(1).strip() if m else ""


_MANIPULATION_VERBS = (
    "pick up", "put down", "pour", "move", "connect", "disconnect",
    "activate", "deactivate", "mix", "dunk", "use", "eat",
    "wait", "fill",
)


def format_episode_summary(task: str, trajectory: list[dict]) -> str:
    """
    Compact deterministic summary of one episode with two-tier key-step markers.

    Successful steps: reward-increasing actions (marked with (+delta)) and
    productive manipulation verbs up to last_gain_idx + 5 (the same follow-
    through window used by _get_induction_trajectory). These are used by the
    induction LLM to update the workflow library.

    Failed steps: unique manipulation verbs attempted after last_gain_idx + 5
    (marked with [x]). These are ignored by induction but surfaced to the
    preplay LLM so it can avoid already-explored dead ends.
    """
    actions = [t["action"] for t in trajectory]

    # Find index of the last step where cumulative reward increased.
    prev_reward = 0.0
    last_gain_idx = -1
    for i, t in enumerate(trajectory):
        r = t.get("reward") or 0.0
        if r > prev_reward:
            last_gain_idx = i
            prev_reward = r

    # Successful trajectory boundary: last reward gain + 5 follow-through steps
    # (mirrors the cutoff used in _get_induction_trajectory).
    # When last_gain_idx is -1 (no reward at all), treat ALL steps as failed —
    # the +5 heuristic only makes sense when there was some partial success.
    if last_gain_idx < 0:
        success_cutoff = -1
    else:
        success_cutoff = min(last_gain_idx + 5, len(trajectory) - 1)

    successful_steps: list[str] = []
    failed_steps: list[str] = []
    key_step_set: set[str] = set()
    manip_seen_success: set[str] = set()
    manip_seen_fail: set[str] = set()
    prev_reward = 0.0

    for i, t in enumerate(trajectory):
        r = t.get("reward") or 0.0
        delta = r - prev_reward
        action = t["action"]
        al = action.lower()

        if delta > 0:
            successful_steps.append(f"{action} (+{delta:.2f})")
            key_step_set.add(action)
            prev_reward = r
        elif r < 0 and i <= success_cutoff:
            # Catastrophic action (reward=-1, wrong focus) within the productive window —
            # always record as failed. Without this, a wrong focus that occurs within
            # last_gain_idx+5 is silently swallowed by the elif below.
            if al.startswith("focus") or any(al.startswith(v) for v in _MANIPULATION_VERBS):
                if action not in manip_seen_fail:
                    failed_steps.append(action)
                    manip_seen_fail.add(action)
        elif i <= success_cutoff:
            # Successful segment: include productive manipulation verbs unmarked.
            if not al.startswith("focus") and any(al.startswith(v) for v in _MANIPULATION_VERBS):
                if action not in key_step_set and action not in manip_seen_success:
                    successful_steps.append(action)
                    manip_seen_success.add(action)
        else:
            # Failed segment: record ALL actions without verb filtering.
            # Collapse consecutive repetitions of the same action (loop compression)
            # so that e.g. 8× "done" or 50× "wait1" don't bloat the summary.
            if failed_steps:
                last = failed_steps[-1]
                base = last.split(" (×")[0] if " (×" in last else last
                if base == action:
                    if " (×" in last:
                        n = int(last.split(" (×")[1].rstrip(")"))
                        failed_steps[-1] = f"{action} (×{n + 1})"
                    else:
                        failed_steps[-1] = f"{action} (×2)"
                else:
                    failed_steps.append(action)
            else:
                failed_steps.append(action)

    lines = [f"Task: {task}"]
    lines.append("Actions: " + " → ".join(actions))
    if successful_steps:
        lines.append("Successful steps: " + ", ".join(successful_steps))
    if failed_steps:
        lines.append("Failed steps: " + " → ".join(failed_steps))
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


def format_example(task: str, trajectory: list[dict]) -> str:
    """Format one (task, trajectory) pair in AWM's Concrete Examples style.

    Steps where reward increased are annotated with reward=+X.XXX so the induction
    LLM can identify which actions actually mattered for task completion.
    """
    lines = [f"Query: {task}", "Actions:"]
    prev_reward = 0.0
    for t in trajectory:
        reward = t.get("reward") or 0.0
        delta = reward - prev_reward
        if t.get("think"):
            lines.append(f"<think>\n{t['think']}\n</think>")
        if delta > 0:
            lines.append(f"<action reward=+{delta:.3f}>\n{t['action']}\n</action>")
        else:
            lines.append(f"<action>\n{t['action']}\n</action>")
        prev_reward = reward
    return "\n".join(lines)



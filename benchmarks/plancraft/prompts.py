"""Prompt templates for the ProPlay agent on PlanCraft."""

SYSTEM_BASE = """\
You are an autonomous crafting agent in PlanCraft, a Minecraft recipe crafting environment.
Your goal is to craft a target item using the materials in your inventory.

## Crafting Table Layout

The crafting table has a 3×3 grid and an output slot:

  Crafting grid: [A1][A2][A3] (row A)
                 [B1][B2][B3] (row B)
                 [C1][C2][C3] (row C)
  Output slot:   [0]  (take crafted items from here)
  Inventory:     [I1]–[I36]  (your 36 personal inventory slots)

## Action Reference

  move: from [Source] to [Target] with quantity N
    — Move N items from Source to Target slot.
    — Source may be [0], [A1]–[C3], or [I1]–[I36].
    — Target may be [A1]–[C3] or [I1]–[I36] (NOT [0]).
    — N must be between 1 and 64.
    — Example: move: from [I3] to [A1] with quantity 1

  smelt: from [Source] to [Target] with quantity N
    — Smelt N items from Source into Target slot.
    — Same slot constraints as move.
    — Example: smelt: from [I2] to [I5] with quantity 1

## Response Format

<think>
Examine the current inventory carefully.
Identify what is needed and what actions to take next.
If an action did not produce the expected result, do not repeat it.
</think>
<action>
move: from [Source] to [Target] with quantity N
</action>

## Crafting Tips

- Shaped recipes: place ingredients in the correct grid pattern.
  Example — wooden pickaxe: planks at [A1][A2][A3] and sticks at [B2][C2].
- Shapeless recipes: ingredients can go in any crafting slot.
- After placing ingredients, the output appears at [0] — take it with:
    move: from [0] to [I1] with quantity N
- Clear the crafting grid between different recipes.
- If you need an intermediate item (e.g. planks before sticks), craft it first,
  move the result to inventory, then proceed with the next recipe.
- Always check the observation for exact slot contents before acting.

## Concrete Example

Task: Craft an item of type: oak_planks

Step 1 — Observation:
Craft an item of type: oak_planks
inventory:
 - oak_log [I1] quantity 1
 - dirt [I5] quantity 3

<think>
I need to craft oak_planks. The recipe requires 1 oak_log anywhere in the crafting grid.
I have oak_log in [I1]. I'll move it to [A1].
</think>
<action>
move: from [I1] to [A1] with quantity 1
</action>

Step 2 — Observation:
Craft an item of type: oak_planks
inventory:
 - oak_log [A1] quantity 1
 - oak_planks [0] quantity 4
 - dirt [I5] quantity 3

<think>
My action placed the oak_log in [A1]. The output slot [0] now shows 4 oak_planks — the recipe worked.
I'll move the oak_planks to my inventory.
</think>
<action>
move: from [0] to [I1] with quantity 4
</action>

(Episode ends — target item acquired.)\
"""


# ── Induction prompts ─────────────────────────────────────────────────────────

INDUCTION_INSTRUCTION = """\
You are a recipe librarian for a Minecraft crafting agent.

Given summaries of past crafting episodes, maintain a workflow library of named recipe procedures.
Each workflow captures HOW to perform a specific crafting recipe in the PlanCraft environment.

Rules for the workflow library:
1. Each workflow entry is named by the item it produces (e.g., "Craft Oak Planks").
2. The body describes the concrete crafting steps: which slot the ingredient goes in, and what the output is.
3. Keep steps abstract enough to generalize across episodes (do not hard-code specific inventory slot positions like [I7]; instead say "move from inventory to crafting slot").
4. If an episode shows a recipe being done differently than in the library, refine the entry.
5. Workflows are numbered sequentially: Workflow 1: Name, Workflow 2: Name, ...
6. After the workflow library, output a <trace> block listing the sequence of workflow names executed in the LATEST episode only.

Output format:
Workflow 1: <Name>
<steps>

Workflow 2: <Name>
<steps>

...

<trace>
1. Workflow Name A
2. Workflow Name B
</trace>\
"""

INDUCTION_ONE_SHOT = """\
## Example

Past Episode Summary (Episode 1):
Target: wooden_pickaxe
Steps taken:
  move: from [I3] to [A1] with quantity 1  (oak_log → crafting)
  move: from [0] to [I1] with quantity 4   (oak_planks ← output)
  move: from [I1] to [A1] with quantity 1  (oak_plank → crafting)
  move: from [I1] to [B1] with quantity 1  (oak_plank → crafting)
  move: from [0] to [I2] with quantity 4   (sticks ← output)
  move: from [I1] to [A1] with quantity 1  (oak_plank → A1)
  move: from [I1] to [A2] with quantity 1  (oak_plank → A2)
  move: from [I1] to [A3] with quantity 1  (oak_plank → A3)
  move: from [I2] to [B2] with quantity 1  (stick → B2)
  move: from [I2] to [C2] with quantity 1  (stick → C2)
  move: from [0] to [I1] with quantity 1   (wooden_pickaxe ← output)
Reward: 1.0

Existing library: (none yet)

---

Workflow 1: Craft Oak Planks
- Place 1 oak_log in any crafting slot (e.g., [A1]).
- Take 4 oak_planks from output [0] to inventory.

Workflow 2: Craft Sticks
- Place 2 planks vertically in the same column (e.g., [A1] and [B1]).
- Take 4 sticks from output [0] to inventory.

Workflow 3: Craft Wooden Pickaxe
- Place 3 planks in the top row [A1][A2][A3].
- Place 2 sticks at [B2] and [C2] (middle and bottom center).
- Take wooden_pickaxe from output [0] to inventory.

<trace>
1. Craft Oak Planks
2. Craft Sticks
3. Craft Wooden Pickaxe
</trace>\
"""

INDUCTION_USER = """\
Existing workflow library:
{existing_workflows}

Past episode summaries (all successful episodes so far):
{examples}

Update the workflow library to incorporate patterns from these episodes.
Then output the <trace> for the LATEST episode only (last summary shown).\
"""

# ── Step-level action selection ───────────────────────────────────────────────

PROPLAY_SYSTEM_TEMPLATE = """\
{base_system}

{plan_section}\
"""

PROPLAY_STEP_USER = """\
Task: {task}

Observation:
{observation}

Working notes (your memory from previous steps):
{memory}

Recent history (last {history_len} steps):
{history}

What do you do next?\
"""

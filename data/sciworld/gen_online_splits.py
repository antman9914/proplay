"""
Generate the online-inference evaluation split for ScienceWorld (shuffled setting).

IMPORTANT: The AgentGym SciWorld server excludes 7 task types from its games list:
    exceptions = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}
    i.e. grow-plant, grow-fruit, inclined-plane-*, mendelian-genetics-*

The server's data_idx is a sequential index into its filtered games list, NOT a
global flat index over all 30 task types. This script replicates the server's
enumeration so that the indices we pass map to the correct tasks.

Split: var 3-14 of each of the 23 included task types (up to 12 per type, ~270 episodes),
then uniformly shuffled with seed=42.

Usage:
    cd data/sciworld
    python gen_online_splits.py
"""
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from scienceworld import ScienceWorldEnv

# ── Replicate server's exceptions filter ───────────────────────────────────────
SERVER_EXCEPTIONS = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}

env = ScienceWorldEnv()
server_starts: dict[str, int] = {}
n_vars: dict[str, int] = {}
cumul = 0
for key, name in env.tasks.items():
    n = env.getMaxVariations(name)
    if key not in SERVER_EXCEPTIONS:
        server_starts[name] = cumul
        n_vars[name] = n
        cumul += n
env.close()

# ── Build episode list (var 3-14 for each of 23 task types) ───────────────────
all_ids: list[int] = []
for name, start in server_starts.items():
    n = n_vars[name]
    for v in range(3, min(15, n)):
        all_ids.append(start + v)

# ── Shuffle with fixed seed ────────────────────────────────────────────────────
shuffled = list(all_ids)
random.seed(42)
random.shuffle(shuffled)

# ── Save ───────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / "splits"
out.mkdir(exist_ok=True)
(out / "online_shuffled_ids.json").write_text(json.dumps(shuffled))

print(f"Saved {len(shuffled)} episodes → splits/online_shuffled_ids.json (shuffled, seed=42)")
print(f"\nTask breakdown ({len(server_starts)} task types):")
for name in server_starts:
    n = n_vars[name]
    count = len(range(3, min(15, n)))
    print(f"  {name:<52}  {count} episodes  (var 3–{min(14, n-1)})")

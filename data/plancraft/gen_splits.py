"""
Generate the PlanCraft evaluation split used by ProPlay.

Source data comes directly from the installed plancraft package —
no separate download needed.

Processing steps:
    1. Load val.small.json and test.small.json from the plancraft package.
    2. Merge (227 total tasks).
    3. Remove tasks marked impossible==True (40 tasks removed → 187 remain).
    4. Sort by complexity_split: easy → medium → hard.
    5. Write splits/merged_187_by_complexity.json.

Usage:
    cd data/plancraft
    python gen_splits.py
"""
from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMPLEXITY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def _load_plancraft_data(filename: str) -> list[dict]:
    """Load a JSON data file bundled inside the plancraft package."""
    try:
        # Python 3.9+ path: importlib.resources.files
        pkg = importlib.resources.files("plancraft").joinpath("data").joinpath(filename)
        return json.loads(pkg.read_text(encoding="utf-8"))
    except Exception:
        # Fallback: locate via package __file__
        import plancraft as _pc
        path = Path(_pc.__file__).parent / "data" / filename
        return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    # ── Load source splits from plancraft package ─────────────────────────────
    val  = _load_plancraft_data("val.small.json")
    test = _load_plancraft_data("test.small.json")

    merged = val + test
    print(f"Loaded from plancraft package: {len(val)} val + {len(test)} test = {len(merged)} total tasks")

    # ── Remove impossible tasks ───────────────────────────────────────────────
    possible = [e for e in merged if not e.get("impossible", False)]
    n_removed = len(merged) - len(possible)
    print(f"Removed {n_removed} impossible tasks → {len(possible)} remain")

    # ── Sort by complexity (easy → medium → hard) ─────────────────────────────
    possible.sort(key=lambda e: COMPLEXITY_ORDER.get(e.get("complexity_split", ""), 99))

    counts: dict[str, int] = {}
    for e in possible:
        c = e.get("complexity_split", "unknown")
        counts[c] = counts.get(c, 0) + 1
    print(f"Complexity distribution: {counts}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = HERE / "splits"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "merged_187_by_complexity.json"
    out_path.write_text(json.dumps(possible, indent=2))
    print(f"Saved {len(possible)} tasks → {out_path}")


if __name__ == "__main__":
    main()

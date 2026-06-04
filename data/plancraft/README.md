# PlanCraft Data

Split files are **not committed** to this repository — generate them with:

```bash
cd data/plancraft
python gen_splits.py
```

This loads `val.small.json` and `test.small.json` directly from the installed
`plancraft` package (no separate download needed), filters out impossible tasks,
and writes `splits/merged_187_by_complexity.json`.

## Generated split

| File | Tasks | Description |
|------|-------|-------------|
| `splits/merged_187_by_complexity.json` | 187 | Solvable tasks (val + test, impossible removed), sorted easy → medium → hard |

## Environment installation

```bash
pip install plancraft
```

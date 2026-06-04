# ScienceWorld Data

The evaluation split is **not committed** to this repository — generate it with:

```bash
pip install agentenv-sciworld scienceworld

cd data/sciworld
python gen_online_splits.py  # → splits/online_shuffled_ids.json
```

## Generated split

| File | Episodes | Description |
|------|----------|-------------|
| `splits/online_shuffled_ids.json` | 270 | 270 test episodes uniformly shuffled (seed=42) |

All indices are server indices — sequential over the 23 included task types,
excluding the 7 server exceptions: grow-plant, grow-fruit, inclined-plane-\*, mendelian-genetics-\*.

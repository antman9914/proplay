# TAU-bench Data

TAU-bench task data is bundled with the tau-bench package itself — no separate download needed.

## Installation

```bash
pip install git+https://github.com/sierra-research/tau-bench
```

## Supported domains

| Domain | Task split sizes | Description |
|--------|-----------------|-------------|
| `retail` | ~500 test tasks | Online retail customer service |
| `airline` | ~300 test tasks | Airline customer service |

**Note:** The telecom domain is not included in this release.

## Loading data

```python
from proplay.data.taubench.load_data import load_env, get_task_ids

env      = load_env("retail", task_split="test")
task_ids = get_task_ids(env)          # all test tasks
task_ids = get_task_ids(env, n=100)   # first 100 tasks
```

See [`load_data.py`](load_data.py) for the full API.

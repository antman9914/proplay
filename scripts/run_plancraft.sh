#!/bin/bash
# Run ProPlay on PlanCraft (Minecraft crafting benchmark).
#
# Prerequisites:
#   pip install plancraft  # or install from the PlanCraft repo
#   export OPENAI_API_KEY=<your_key>
#
# Usage:
#   bash scripts/run_plancraft.sh
set -eo pipefail

PYTHON=${PYTHON:-python}
LOG_DIR=${LOG_DIR:-./logs/plancraft}

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."; exit 1
fi

$PYTHON benchmarks/plancraft/pipeline.py \
    --agent      proplay \
    --data_path  "data/plancraft/splits/merged_187_by_complexity.json" \
    --steps      30 \
    --model      "gpt-4.1-mini" \
    --api_base   "https://api.openai.com/v1" \
    --api_key    "$OPENAI_API_KEY" \
    --max_tokens 1024 \
    --workflow_path      "benchmarks/plancraft/workflow/proplay.txt" \
    --preplay_graph_path "benchmarks/plancraft/workflow/proplay_graph.json" \
    --use_preplay \
    --log_dir    "$LOG_DIR"

echo "Done. Logs saved to $LOG_DIR"

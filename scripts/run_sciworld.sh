#!/bin/bash
# Run ProPlay on ScienceWorld (online shuffled setting).
#
# Prerequisites:
#   1. Install AgentGym SciWorld server:
#      pip install agentenv-sciworld
#   2. Set your API key:
#      export OPENAI_API_KEY=<your_key>
#
# Usage:
#   bash scripts/run_sciworld.sh
set -eo pipefail

PYTHON=${PYTHON:-python}
SCIWORLD_PORT=${SCIWORLD_PORT:-36006}
LOG_DIR=${LOG_DIR:-./logs/sciworld}

# ── Validate env ───────────────────────────────────────────────────────────────
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."; exit 1
fi

# ── Start ScienceWorld server ─────────────────────────────────────────────────
$PYTHON -m uvicorn agentenv_sciworld.server:app \
    --host 0.0.0.0 --port "$SCIWORLD_PORT" &
SCIWORLD_PID=$!

echo "Waiting for ScienceWorld server on port $SCIWORLD_PORT ..."
for i in $(seq 1 30); do
    if ! kill -0 $SCIWORLD_PID 2>/dev/null; then
        echo "ERROR: SciWorld server died."; exit 1
    fi
    if curl -sf "http://localhost:$SCIWORLD_PORT/" > /dev/null 2>&1; then
        echo "SciWorld ready."; break
    fi
    [ $i -eq 30 ] && { echo "ERROR: SciWorld timed out."; kill $SCIWORLD_PID; exit 1; }
    sleep 10
done

# ── Online ProPlay — uniformly shuffled tasks ─────────────────────────────────
$PYTHON benchmarks/sciworld/pipeline.py \
    --steps      100 \
    --server     "http://localhost:$SCIWORLD_PORT" \
    --model      "gpt-4.1-mini" \
    --api_base   "https://api.openai.com/v1" \
    --api_key    "$OPENAI_API_KEY" \
    --max_tokens 1024 \
    --workflow_path      "benchmarks/sciworld/workflow/online.txt" \
    --preplay_graph_path "benchmarks/sciworld/workflow/online_graph.json" \
    --use_graph --preplay \
    --task_ids   "data/sciworld/splits/online_shuffled_ids.json" \
    --log_dir    "$LOG_DIR"

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill $SCIWORLD_PID 2>/dev/null || true
echo "Done. Logs saved to $LOG_DIR"

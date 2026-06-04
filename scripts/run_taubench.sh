#!/bin/bash
# Run ProPlay on TAU-bench (retail or airline domain).
#
# Prerequisites:
#   pip install git+https://github.com/sierra-research/tau-bench
#   export OPENAI_API_KEY=<your_key>
#
# Usage:
#   DOMAIN=retail  bash scripts/run_taubench.sh
#   DOMAIN=airline bash scripts/run_taubench.sh
set -eo pipefail

PYTHON=${PYTHON:-python}
DOMAIN=${DOMAIN:-retail}
LOG_DIR=${LOG_DIR:-./logs/taubench_${DOMAIN}}

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."; exit 1
fi

if [[ "$DOMAIN" != "retail" && "$DOMAIN" != "airline" ]]; then
    echo "ERROR: DOMAIN must be 'retail' or 'airline', got '$DOMAIN'"; exit 1
fi

$PYTHON benchmarks/taubench/pipeline.py \
    --env        "$DOMAIN" \
    --task_split test \
    --steps      30 \
    --model      "gpt-4.1-mini" \
    --api_base   "https://api.openai.com/v1" \
    --api_key    "$OPENAI_API_KEY" \
    --max_tokens 1024 \
    --graph_path "benchmarks/taubench/graph/${DOMAIN}.json" \
    --preplay \
    --log_dir    "$LOG_DIR"

echo "Done. Logs saved to $LOG_DIR"

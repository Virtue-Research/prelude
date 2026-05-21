#!/usr/bin/env bash
#
# Wrapper around policyguard_10_bench.sh to benchmark on the full
# sampled_500_groups_x_10.jsonl dataset.
#
# Defaults:
#   DATASET_PATH=<repo>/sampled_500_groups_x_10.jsonl
#   SAMPLES=<all non-empty lines in DATASET_PATH>
#   WARMUP_SAMPLES=0
#   SAMPLE_MODE=first
#
# Usage:
#   ./benchmark/policyguard_sampled500x10_full_bench.sh
#   ./benchmark/policyguard_sampled500x10_full_bench.sh prelude --gpu
#   ./benchmark/policyguard_sampled500x10_full_bench.sh vllm --gpu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRELUDE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/policyguard_10_bench.sh"

if [ ! -f "$BASE_SCRIPT" ]; then
    echo "Base benchmark script not found: $BASE_SCRIPT" >&2
    exit 1
fi

DATASET_PATH="${DATASET_PATH:-$PRELUDE_DIR/sampled_500_groups_x_10.jsonl}"
if [ ! -f "$DATASET_PATH" ]; then
    echo "Dataset file not found: $DATASET_PATH" >&2
    exit 1
fi

if [ -z "${SAMPLES:-}" ]; then
    SAMPLES="$(
        DATASET_PATH_VALUE="$DATASET_PATH" python3 - <<'PY'
import os

path = os.environ["DATASET_PATH_VALUE"]
count = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            count += 1
print(count)
PY
    )"
fi

export DATASET_URL="file://$DATASET_PATH"
export SAMPLES
export WARMUP_SAMPLES="${WARMUP_SAMPLES:-0}"
export SAMPLE_MODE="${SAMPLE_MODE:-first}"
export RESULTS_DIR="${RESULTS_DIR:-$PRELUDE_DIR/bench_results/policyguard_sampled500x10_full_$(date +%Y%m%d_%H%M%S)}"

exec "$BASE_SCRIPT" "$@"

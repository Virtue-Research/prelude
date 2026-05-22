#!/usr/bin/env bash
#
# Wrapper around policyguard_10_bench.sh to benchmark on the full
# sampled_500_groups_x_10.jsonl dataset.
#
# Defaults:
#   DATASET_PATH=<repo>/sampled_500_groups_x_10.jsonl
#   SAMPLES=<all non-empty lines in DATASET_PATH>
#   WARMUP_SAMPLES=5
#   SAMPLE_MODE=first
#   CONCURRENCY_LIST="1 8 32 64 128"
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
export WARMUP_SAMPLES="${WARMUP_SAMPLES:-5}"
export SAMPLE_MODE="${SAMPLE_MODE:-first}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 8 32 64 128}"
RESULTS_ROOT="${RESULTS_DIR:-$PRELUDE_DIR/bench_results/policyguard_sampled500x10_full_$(date +%Y%m%d_%H%M%S)}"
AGG_CSV_FILE="$RESULTS_ROOT/avg_latency_by_concurrency.csv"

mkdir -p "$RESULTS_ROOT"
echo "concurrency,prelude_avg_latency_ms,vllm_avg_latency_ms,run_dir" > "$AGG_CSV_FILE"

for c in $CONCURRENCY_LIST; do
    case "$c" in
        ''|*[!0-9]*)
            echo "Invalid concurrency value in CONCURRENCY_LIST: $c" >&2
            exit 1
            ;;
    esac

    run_dir="$RESULTS_ROOT/c${c}"
    echo ""
    echo "=== Running concurrency=$c ==="
    CONCURRENCY="$c" RESULTS_DIR="$run_dir" "$BASE_SCRIPT" "$@"

    run_csv="$run_dir/summary.csv"
    if [ ! -f "$run_csv" ]; then
        echo "Missing summary file: $run_csv" >&2
        exit 1
    fi

    RUN_CSV="$run_csv" \
    RUN_DIR="$run_dir" \
    CURRENT_CONCURRENCY="$c" \
    AGG_CSV="$AGG_CSV_FILE" \
    python3 - <<'PY'
import csv
import os

run_csv = os.environ["RUN_CSV"]
run_dir = os.environ["RUN_DIR"]
concurrency = os.environ["CURRENT_CONCURRENCY"]
agg_csv = os.environ["AGG_CSV"]


def normalize_engine(name: str) -> str:
    lower = (name or "").strip().lower()
    if lower == "prelude":
        return "prelude"
    if lower == "vllm":
        return "vllm"
    return ""


def parse_latency(value: str):
    value = (value or "").strip()
    if not value or value.upper() == "N/A":
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return "N/A"


result = {"prelude": "N/A", "vllm": "N/A"}

with open(run_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = normalize_engine(row.get("engine", ""))
        if key and result[key] == "N/A":
            result[key] = parse_latency(row.get("avg_latency_ms", ""))

with open(agg_csv, "a", encoding="utf-8") as f:
    f.write(f"{concurrency},{result['prelude']},{result['vllm']},{run_dir}\n")
PY
done

AGG_CSV="$AGG_CSV_FILE" python3 - <<'PY'
import csv
import os

agg_csv = os.environ["AGG_CSV"]

with open(agg_csv, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("No rows in aggregate csv")

headers = ["Concurrency", "Prelude", "vLLM"]
widths = [len(h) for h in headers]
for row in rows:
    widths[0] = max(widths[0], len(row.get("concurrency", "")))
    widths[1] = max(widths[1], len(row.get("prelude_avg_latency_ms", "")))
    widths[2] = max(widths[2], len(row.get("vllm_avg_latency_ms", "")))

def fmt_line(a, b, c):
    return (
        f"| {a:>{widths[0]}} "
        f"| {b:>{widths[1]}} "
        f"| {c:>{widths[2]}} |"
    )

print("")
print("=== Avg latency by concurrency (ms) ===")
print(fmt_line(headers[0], headers[1], headers[2]))
print(
    f"| {'-' * widths[0]} | {'-' * widths[1]} | {'-' * widths[2]} |"
)
for row in rows:
    print(
        fmt_line(
            row.get("concurrency", ""),
            row.get("prelude_avg_latency_ms", ""),
            row.get("vllm_avg_latency_ms", ""),
        )
    )
print("")
print(f"Aggregate CSV: {agg_csv}")
PY

#!/usr/bin/env bash
#
# Run baseline kernel micro-benchmarks (vLLM, SGLang) in Docker.
#
# Usage:
#   ./benchmark/microbench/run.sh            # run all baselines (uses cache if present)
#   ./benchmark/microbench/run.sh --force    # force re-run even if cached
#
# Results saved to benchmark/microbench/results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

has_cached() {
    ls "${RESULTS_DIR}/$1_"*.json &>/dev/null
}

latest_cached() {
    ls -t "${RESULTS_DIR}/$1_"*.json 2>/dev/null | head -1
}

if [ -f "${SCRIPT_DIR}/baselines/sglang/bench_kernels.py" ]; then
    if [ "$FORCE" = true ] || ! has_cached sglang; then
        echo ">>> Running sglang baseline..."
        docker run --rm --gpus all \
            -v "${SCRIPT_DIR}/baselines/sglang:/bench" \
            -v "${RESULTS_DIR}:/results" \
            lmsysorg/sglang:latest \
            python3 /bench/bench_kernels.py --json "/results/sglang_${TIMESTAMP}.json" \
            || echo "  (sglang skipped — failed or image not available)"
    else
        echo ">>> sglang: using cached $(latest_cached sglang)"
    fi
fi

if [ -f "${SCRIPT_DIR}/baselines/vllm/bench_kernels.py" ]; then
    if [ "$FORCE" = true ] || ! has_cached vllm; then
        echo ">>> Running vllm baseline..."
        docker run --rm --gpus all --entrypoint python3 \
            -v "${SCRIPT_DIR}/baselines/vllm:/bench" \
            -v "${RESULTS_DIR}:/results" \
            vllm/vllm-openai:latest \
            /bench/bench_kernels.py --json "/results/vllm_${TIMESTAMP}.json" \
            || echo "  (vllm skipped — failed or image not available)"
    else
        echo ">>> vllm: using cached $(latest_cached vllm)"
    fi
fi

echo ""
echo "=== Results ==="
ls -t "${RESULTS_DIR}"/*.json 2>/dev/null | head -10 || echo "  (no results)"

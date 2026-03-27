#!/usr/bin/env bash
#
# Run micro-benchmarks via Docker (dev image + mount local source).
#
# Usage:
#   ./benchmark/microbench/run.sh                    # run ours
#   ./benchmark/microbench/run.sh quant              # filter: quant only
#   ./benchmark/microbench/run.sh --all              # ours + baselines (cached)
#   ./benchmark/microbench/run.sh --all --force      # force re-run baselines
#
# Results saved to benchmark/microbench/results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE="prelude-microbench-dev"

mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_BASELINES=false
RUN_OURS=true
FORCE=false
FILTERS=()

for arg in "$@"; do
    case "$arg" in
        --all)       RUN_BASELINES=true ;;
        --baselines) RUN_BASELINES=true; RUN_OURS=false ;;
        --force)     FORCE=true ;;
        *)           FILTERS+=("$arg") ;;
    esac
done

FILTER_ARGS="${FILTERS[*]:-}"

has_cached() {
    ls "${RESULTS_DIR}/$1_"*.json &>/dev/null
}

latest_cached() {
    ls -t "${RESULTS_DIR}/$1_"*.json 2>/dev/null | head -1
}

# ── Ensure dev image exists ──────────────────────────────────────────────
if ! docker image inspect "${IMAGE}" &>/dev/null; then
    echo ">>> Building ${IMAGE}..."
    docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE}" "${PROJECT_ROOT}"
fi

# ── Our benchmarks (always fresh — mount local source) ──────────────────
if [ "$RUN_OURS" = true ]; then
    echo ">>> Running prelude-microbench..."
    docker run --rm \
        -v "${PROJECT_ROOT}:/workspace" \
        -v prelude-bench-target:/workspace/target \
        -v "${RESULTS_DIR}:/results" \
        "${IMAGE}" \
        ${FILTER_ARGS} --json "/results/prelude_${TIMESTAMP}.json"
    echo ">>> Results: ${RESULTS_DIR}/prelude_${TIMESTAMP}.json"
fi

# ── Baselines (cached unless --force) ───────────────────────────────────
if [ "$RUN_BASELINES" = true ]; then

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
fi

echo ""
echo "=== Results ==="
ls -t "${RESULTS_DIR}"/*.json 2>/dev/null | head -10 || echo "  (no results)"

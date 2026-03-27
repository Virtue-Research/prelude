#!/usr/bin/env bash
#
# Run micro-benchmarks. Supports both local and Docker modes.
#
# Usage:
#   ./benchmark/microbench/run.sh                    # run locally (native)
#   ./benchmark/microbench/run.sh forward             # forward benchmark only
#   ./benchmark/microbench/run.sh quant               # quant kernels only
#   ./benchmark/microbench/run.sh --docker            # run via Docker dev image
#   ./benchmark/microbench/run.sh --all               # ours + baselines (Docker)
#   ./benchmark/microbench/run.sh --all --force       # force re-run baselines
#
# Environment variables:
#   GGUF_MODEL=/path/to/model.gguf    GGUF model for llama-bench comparison
#   LLAMA_BENCH=/path/to/llama-bench  llama-bench binary path
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
USE_DOCKER=false
FILTERS=()

for arg in "$@"; do
    case "$arg" in
        --all)       RUN_BASELINES=true; USE_DOCKER=true ;;
        --baselines) RUN_BASELINES=true; RUN_OURS=false; USE_DOCKER=true ;;
        --docker)    USE_DOCKER=true ;;
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

# ── Our benchmarks ────────────────────────────────────────────────────────

if [ "$RUN_OURS" = true ]; then
    if [ "$USE_DOCKER" = true ]; then
        # Docker mode
        if ! docker image inspect "${IMAGE}" &>/dev/null; then
            echo ">>> Building ${IMAGE}..."
            docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE}" "${PROJECT_ROOT}"
        fi

        DOCKER_ARGS=()
        # Mount HuggingFace cache for model access
        if [ -d "$HOME/.cache/huggingface" ]; then
            DOCKER_ARGS+=(-v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro")
        fi
        # Pass through GGUF model if specified
        if [ -n "${GGUF_MODEL:-}" ] && [ -f "${GGUF_MODEL}" ]; then
            DOCKER_ARGS+=(-v "${GGUF_MODEL}:/models/$(basename "${GGUF_MODEL}"):ro")
            DOCKER_ARGS+=(-e "GGUF_MODEL=/models/$(basename "${GGUF_MODEL}")")
        fi

        echo ">>> Running prelude-microbench (Docker)..."
        docker run --rm \
            -v "${PROJECT_ROOT}:/workspace" \
            -v prelude-bench-target:/workspace/target \
            -v "${RESULTS_DIR}:/results" \
            "${DOCKER_ARGS[@]}" \
            "${IMAGE}" \
            ${FILTER_ARGS} --json "/results/prelude_${TIMESTAMP}.json"
        echo ">>> Results: ${RESULTS_DIR}/prelude_${TIMESTAMP}.json"
    else
        # Local mode (native performance with target-cpu=native)
        echo ">>> Running prelude-microbench (local)..."

        EXTRA_ARGS=""
        if [ -n "${GGUF_MODEL:-}" ]; then
            EXTRA_ARGS="--gguf-model ${GGUF_MODEL}"
        fi

        cargo run -p prelude-microbench --release -- \
            ${FILTER_ARGS} ${EXTRA_ARGS} \
            --json "${RESULTS_DIR}/prelude_${TIMESTAMP}.json"
        echo ">>> Results: ${RESULTS_DIR}/prelude_${TIMESTAMP}.json"
    fi
fi

# ── Baselines (cached unless --force) ─────────────────────────────────────

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

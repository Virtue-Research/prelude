#!/usr/bin/env bash
#
# E2E inference benchmark — each engine runs in its own Docker container.
#
# Usage:
#   ./benchmark/e2e/bench.sh prelude              # GPU
#   ./benchmark/e2e/bench.sh prelude --cpu         # CPU
#   ./benchmark/e2e/bench.sh vllm                  # vLLM GPU
#   ./benchmark/e2e/bench.sh vllm --cpu            # vLLM CPU (needs vllm-cpu image)
#   ./benchmark/e2e/bench.sh all                   # all engines, GPU
#   ./benchmark/e2e/bench.sh all --cpu             # all engines, CPU
#
# Environment variables:
#   MODEL           HuggingFace model (default: Qwen/Qwen3-0.6B)
#   GGUF_MODEL      GGUF file path for llama.cpp (required for llama.cpp)
#   INPUT_TOKENS    Prompt length (default: 128)
#   OUTPUT_TOKENS   Generation length (default: 1, i.e. prefill-only)
#   MAX_REQUESTS    Total requests (default: 200)
#   CONCURRENCY     Concurrent requests (default: 1)
#   GPU             GPU id (default: 0)
#   PORT            Server port (default: 8000)
#   RESULTS_DIR     Output directory (default: benchmark/e2e/results)
#   HF_TOKEN        HuggingFace token for gated models
#
# Prerequisites:
#   pip install "genai-bench @ git+https://github.com/rucnyz/genai-bench.git"
#
# CPU images for vLLM/SGLang (build once if needed):
#   vLLM:   cd vllm && docker build -f docker/Dockerfile.cpu -t vllm-cpu --target vllm-openai .
#   SGLang: cd sglang && docker build -f docker/xeon.Dockerfile -t sglang-cpu .
#
# Prelude uses the root Dockerfile (prelude-dev image) for both microbench and e2e.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Config ────────────────────────────────────────────────────────────────

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
GGUF_MODEL="${GGUF_MODEL:-}"
INPUT_TOKENS="${INPUT_TOKENS:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1}"
MAX_REQUESTS="${MAX_REQUESTS:-200}"
CONCURRENCY="${CONCURRENCY:-1}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
HF_TOKEN="${HF_TOKEN:-}"
CONTAINER_NAME="prelude-e2e-bench"
HEALTH_TIMEOUT=180
CPU_MODE=false
HF_CACHE="${HOME}/.cache/huggingface"

ENGINES=()
for arg in "$@"; do
    case "$arg" in
        --cpu)  CPU_MODE=true ;;
        *)      ENGINES+=("$arg") ;;
    esac
done

mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Result file suffix includes device mode
result_tag() {
    local engine="$1"
    if [[ "${CPU_MODE}" == true ]]; then
        echo "${engine}-cpu_${TIMESTAMP}"
    else
        echo "${engine}_${TIMESTAMP}"
    fi
}

# ── Helpers ───────────────────────────────────────────────────────────────

wait_healthy() {
    local url="$1"
    local timeout="${2:-$HEALTH_TIMEOUT}"
    echo "  Waiting for ${url} ..."
    for ((i=0; i<timeout; i++)); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            echo "  Ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: not healthy after ${timeout}s"
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -20
    return 1
}

cleanup() {
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

run_benchmark() {
    local engine="$1"
    local url="http://localhost:${PORT}"
    local out="${RESULTS_DIR}/$(result_tag "${engine}").json"

    local mode="GPU"
    [[ "${CPU_MODE}" == true ]] && mode="CPU"
    echo "  genai-bench [${mode}]: D(${INPUT_TOKENS},${OUTPUT_TOKENS}) × ${MAX_REQUESTS} @ C=${CONCURRENCY}"
    genai-bench benchmark \
        --backend openai \
        --base-url "${url}" \
        --model "${MODEL}" \
        --tokenizer "${MODEL}" \
        --dataset-name random \
        --random-input-len "${INPUT_TOKENS}" \
        --random-output-len "${OUTPUT_TOKENS}" \
        --num-requests "${MAX_REQUESTS}" \
        --max-concurrency "${CONCURRENCY}" \
        --output-file "${out}" \
        2>&1 | tail -30

    echo "  Results: ${out}"
}

# ── Engine launchers ──────────────────────────────────────────────────────

start_prelude() {
    local image="prelude-dev"
    if ! docker image inspect "${image}" &>/dev/null; then
        echo ">>> Building ${image} ..."
        docker build -f "${PROJECT_ROOT}/Dockerfile" -t "${image}" "${PROJECT_ROOT}"
    fi

    local extra="${EXTRA_ARGS:-}"
    [[ "${CPU_MODE}" == true ]] && extra="--device cpu ${extra}"

    local gpu_flag=""
    [[ "${CPU_MODE}" == false ]] && gpu_flag="--gpus device=${GPU}"

    # Mount source + HF cache, build and run server
    # shellcheck disable=SC2086
    docker run -d --name "${CONTAINER_NAME}" \
        ${gpu_flag} \
        -p "${PORT}:8000" \
        -v "${PROJECT_ROOT}:/workspace" \
        -v prelude-e2e-target:/workspace/target \
        -v "${HF_CACHE}:/root/.cache/huggingface:ro" \
        "${image}" \
        bash -c "cd /workspace && cargo build -p prelude-server --release && exec ./target/release/prelude-server --model ${MODEL} --host 0.0.0.0 --port 8000 ${extra}"
}

start_vllm() {
    if [[ "${CPU_MODE}" == true ]]; then
        # CPU: build from vllm/docker/Dockerfile.cpu or use pre-built vllm-cpu
        # Build: cd vllm && docker build -f docker/Dockerfile.cpu -t vllm-cpu --target vllm-openai .
        docker run -d --name "${CONTAINER_NAME}" \
            -p "${PORT}:8000" \
            --ipc=host \
            -v "${HF_CACHE}:/root/.cache/huggingface:ro" \
            -e HF_TOKEN="${HF_TOKEN}" \
            vllm-cpu:latest \
            --model "${MODEL}" \
            --host 0.0.0.0 \
            --port 8000 \
            ${VLLM_EXTRA_ARGS:-}
    else
        # GPU: official image, entrypoint is already `vllm serve`
        # https://docs.vllm.ai/en/latest/deployment/docker.html
        docker run -d --name "${CONTAINER_NAME}" \
            --runtime nvidia --gpus "device=${GPU}" \
            -p "${PORT}:8000" \
            --ipc=host \
            -v "${HF_CACHE}:/root/.cache/huggingface:ro" \
            -e HF_TOKEN="${HF_TOKEN}" \
            vllm/vllm-openai:latest \
            --model "${MODEL}" \
            --host 0.0.0.0 \
            --port 8000 \
            ${VLLM_EXTRA_ARGS:-}
    fi
}

start_sglang() {
    # https://docs.sglang.ai/get_started/install.html
    if [[ "${CPU_MODE}" == true ]]; then
        # CPU: Xeon-optimized image
        # Build: cd sglang && docker build -f docker/xeon.Dockerfile -t sglang-cpu .
        docker run -d --name "${CONTAINER_NAME}" \
            -p "${PORT}:${PORT}" \
            --ipc=host \
            -v "${HF_CACHE}:/root/.cache/huggingface:ro" \
            -e HF_TOKEN="${HF_TOKEN}" \
            sglang-cpu:latest \
            python3 -m sglang.launch_server \
            --model-path "${MODEL}" \
            --host 0.0.0.0 \
            --port "${PORT}" \
            ${SGLANG_EXTRA_ARGS:-}
    else
        # GPU: official image
        docker run -d --name "${CONTAINER_NAME}" \
            --gpus "device=${GPU}" \
            --shm-size 32g \
            -p "${PORT}:${PORT}" \
            --ipc=host \
            -v "${HF_CACHE}:/root/.cache/huggingface:ro" \
            -e HF_TOKEN="${HF_TOKEN}" \
            lmsysorg/sglang:latest \
            python3 -m sglang.launch_server \
            --model-path "${MODEL}" \
            --host 0.0.0.0 \
            --port "${PORT}" \
            ${SGLANG_EXTRA_ARGS:-}
    fi
}

start_vllm_rs() {
    local image="vllm-rs-e2e"
    local vllm_rs_dir="${VLLM_RS_DIR:-${PROJECT_ROOT}/../vllm.rs}"
    if ! docker image inspect "${image}" &>/dev/null; then
        if [[ ! -d "${vllm_rs_dir}" ]]; then
            echo "ERROR: vllm.rs source not found at ${vllm_rs_dir}"
            echo "  Set VLLM_RS_DIR or clone: git clone https://github.com/yuzhounie/vllm.rs ../vllm.rs"
            return 1
        fi
        echo ">>> Building ${image} (this takes a while) ..."
        docker build -f "${SCRIPT_DIR}/Dockerfile.vllm-rs" -t "${image}" "${vllm_rs_dir}"
    fi

    # vllm.rs is GPU-only
    if [[ "${CPU_MODE}" == true ]]; then
        echo "  SKIP: vllm.rs does not support CPU mode"
        return 1
    fi

    docker run -d --name "${CONTAINER_NAME}" \
        --gpus "device=${GPU}" \
        -p "${PORT}:8000" \
        --ipc=host \
        -v "${HF_CACHE}:/data:ro" \
        -e HF_TOKEN="${HF_TOKEN}" \
        "${image}" \
        --model "${MODEL}" \
        --host 0.0.0.0 \
        --port 8000
}

start_llama_cpp() {
    local image="llama-cpp-e2e"
    if ! docker image inspect "${image}" &>/dev/null; then
        echo ">>> Building ${image} ..."
        docker build -f "${SCRIPT_DIR}/Dockerfile.llama-cpp" -t "${image}" "${SCRIPT_DIR}"
    fi
    if [[ -z "${GGUF_MODEL}" ]]; then
        echo "ERROR: GGUF_MODEL not set. Example:"
        echo "  GGUF_MODEL=/path/to/Qwen3-0.6B-BF16.gguf ./benchmark/e2e/bench.sh llama.cpp"
        return 1
    fi
    local model_dir
    model_dir="$(cd "$(dirname "${GGUF_MODEL}")" && pwd)"
    local model_file
    model_file="$(basename "${GGUF_MODEL}")"

    local ngl=99
    local gpu_flag="--gpus device=${GPU}"
    if [[ "${CPU_MODE}" == true ]]; then
        ngl=0
        gpu_flag=""
    fi

    # shellcheck disable=SC2086
    docker run -d --name "${CONTAINER_NAME}" \
        ${gpu_flag} \
        -p "${PORT}:8000" \
        -v "${model_dir}:/models:ro" \
        -e MODEL="/models/${model_file}" \
        -e N_GPU_LAYERS="${ngl}" \
        "${image}"
}

# ── Run one engine ────────────────────────────────────────────────────────

bench_engine() {
    local engine="$1"
    local mode="GPU"
    [[ "${CPU_MODE}" == true ]] && mode="CPU"
    echo ""
    echo "=== ${engine} (${mode}) ==="
    cleanup

    case "${engine}" in
        prelude)    start_prelude ;;
        vllm)       start_vllm ;;
        vllm.rs)    start_vllm_rs ;;
        sglang)     start_sglang ;;
        llama.cpp)  start_llama_cpp ;;
        *)
            echo "Unknown engine: ${engine}"
            echo "Available: prelude, vllm, vllm.rs, sglang, llama.cpp"
            return 1
            ;;
    esac

    if wait_healthy "http://localhost:${PORT}/health"; then
        run_benchmark "${engine}"
    fi

    cleanup
}

# ── Main ──────────────────────────────────────────────────────────────────

if [[ ${#ENGINES[@]} -eq 0 ]]; then
    echo "Usage: $0 <engine|all> [engine2 ...] [--cpu]"
    echo ""
    echo "Engines: prelude, vllm, vllm.rs, sglang, llama.cpp"
    echo "  all = all engines"
    echo ""
    echo "Flags:"
    echo "  --cpu    CPU mode (no GPU, auto-selects CPU images where needed)"
    echo ""
    echo "Examples:"
    echo "  $0 prelude                             # Prelude GPU"
    echo "  $0 prelude --cpu                       # Prelude CPU"
    echo "  $0 all --cpu                           # All engines CPU"
    echo "  INPUT_TOKENS=512 $0 prelude vllm       # Compare prelude vs vllm"
    echo ""
    echo "Environment:"
    echo "  MODEL=${MODEL}  GPU=${GPU}  PORT=${PORT}"
    echo "  INPUT_TOKENS=${INPUT_TOKENS}  OUTPUT_TOKENS=${OUTPUT_TOKENS}"
    echo "  MAX_REQUESTS=${MAX_REQUESTS}  CONCURRENCY=${CONCURRENCY}"
    exit 0
fi

trap cleanup EXIT

for engine in "${ENGINES[@]}"; do
    if [[ "${engine}" == "all" ]]; then
        for e in prelude vllm vllm.rs sglang llama.cpp; do
            bench_engine "${e}" || true
        done
    else
        bench_engine "${engine}"
    fi
done

echo ""
echo "=== Results ==="
ls -t "${RESULTS_DIR}"/*.json 2>/dev/null | head -10 || echo "  (no results)"

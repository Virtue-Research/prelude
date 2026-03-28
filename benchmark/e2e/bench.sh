#!/usr/bin/env bash
#
# E2E inference benchmark — each engine runs in its own Docker container.
#
# Usage:
#   ./benchmark/e2e/bench.sh prelude --local       # GPU, local binary (no Docker)
#   ./benchmark/e2e/bench.sh prelude --local --cpu # CPU, local binary
#   ./benchmark/e2e/bench.sh prelude               # GPU, Docker
#   ./benchmark/e2e/bench.sh vllm                  # vLLM GPU
#   ./benchmark/e2e/bench.sh all --gpu             # all engines, GPU
#   ./benchmark/e2e/bench.sh all --cpu             # all engines, CPU
#
# Environment variables:
#   MODEL           HuggingFace model (default: Qwen/Qwen3-0.6B)
#   INPUT_TOKENS    Prompt length (default: 128)
#   OUTPUT_TOKENS   Generation length (default: 1, i.e. prefill-only)
#   MAX_REQUESTS    Total requests (default: 200)
#   CONCURRENCY     Concurrent requests (default: 1)
#   GPU             GPU id (default: 0)
#   PORT            Server port (default: 8000)
#   RESULTS_DIR     Output directory (default: benchmark/e2e/results)
#   HF_TOKEN        HuggingFace token for gated models
#   LLAMA_CPP_CONVERT  Path to convert_hf_to_gguf.py (auto-detected)
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
GGUF_CACHE="${HOME}/.cache/prelude/gguf"

DEVICE_FILTER=""  # "", "cpu", or "gpu"
LOCAL_PRELUDE=false
ENGINES=()
for arg in "$@"; do
    case "$arg" in
        --cpu)    DEVICE_FILTER=cpu ;;
        --gpu)    DEVICE_FILTER=gpu ;;
        --local)  LOCAL_PRELUDE=true ;;
        *)        ENGINES+=("$arg") ;;
    esac
done

mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="${RESULTS_DIR}/summary_${TIMESTAMP}.csv"
TRAFFIC="D(${INPUT_TOKENS},${OUTPUT_TOKENS})"

# Hardware info
CPU_NAME=$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')
GPU_NAME=""; GPU_COUNT=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)
fi

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

STARTUP_ELAPSED=0
LOCAL_PID=""

wait_healthy() {
    local url="$1"
    local timeout="${2:-$HEALTH_TIMEOUT}"
    echo "  Waiting for ${url} ..."
    for ((i=0; i<timeout; i++)); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            STARTUP_ELAPSED=$i
            echo "  Ready (${i}s)"
            return 0
        fi
        # Check if the process/container died
        if [[ -n "${LOCAL_PID}" ]]; then
            if ! kill -0 "${LOCAL_PID}" 2>/dev/null; then
                echo "  ERROR: local process ${LOCAL_PID} died"
                return 1
            fi
        fi
        sleep 1
    done
    STARTUP_ELAPSED=$timeout
    echo "  ERROR: not healthy after ${timeout}s"
    if [[ -n "${LOCAL_PID}" ]]; then
        echo "  (local process ${LOCAL_PID})"
    else
        docker logs "${CONTAINER_NAME}" 2>&1 | tail -20
    fi
    return 1
}

cleanup() {
    if [[ -n "${LOCAL_PID}" ]]; then
        kill "${LOCAL_PID}" 2>/dev/null || true
        wait "${LOCAL_PID}" 2>/dev/null || true
        LOCAL_PID=""
    fi
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    # Kill anything still listening on our port (stale processes from prior runs)
    local pid
    pid=$(lsof -ti tcp:"${PORT}" 2>/dev/null) && kill "$pid" 2>/dev/null && sleep 1 || true
}

run_benchmark() {
    local engine="$1"
    local url="http://localhost:${PORT}"
    local tag
    tag="$(result_tag "${engine}")"

    local mode="GPU"
    [[ "${CPU_MODE}" == true ]] && mode="CPU"
    echo "  genai-bench [${mode}]: ${TRAFFIC} × ${MAX_REQUESTS} @ C=${CONCURRENCY}"
    genai-bench benchmark \
        --api-backend openai \
        --api-base "${url}" \
        --api-key "dummy" \
        --api-model-name "${MODEL}" \
        --model-tokenizer "${MODEL}" \
        --task text-to-text \
        --traffic-scenario "${TRAFFIC}" \
        --num-concurrency "${CONCURRENCY}" \
        --max-requests-per-run "${MAX_REQUESTS}" \
        --max-time-per-run 10 \
        --experiment-base-dir "${RESULTS_DIR}" \
        --experiment-folder-name "${tag}" \
        2>&1 | tee "${RESULTS_DIR}/${tag}.log" | tail -30

    # Extract metrics from genai-bench JSON into CSV
    local json_file
    json_file=$(find "${RESULTS_DIR}/${tag}" -name '*.json' -not -name 'experiment_metadata.json' 2>/dev/null | head -1)
    if [[ -n "${json_file}" ]] && [[ -f "${json_file}" ]]; then
        python3 "${SCRIPT_DIR}/bench_utils.py" extract-metrics \
            --json-file "${json_file}" --engine "${engine}" --device "${mode}" \
            --startup-s "${STARTUP_ELAPSED:-0}" --csv-file "${CSV_FILE}"
    else
        echo "  WARNING: no result JSON found"
        echo "${engine},${mode},${STARTUP_ELAPSED:-0},N/A,N/A,N/A,N/A,N/A,N/A" >> "${CSV_FILE}"
    fi
}

# ── Engine launchers ──────────────────────────────────────────────────────

start_prelude() {
    local extra="${EXTRA_ARGS:-}"
    [[ "${CPU_MODE}" == true ]] && extra="--device cpu ${extra}"

    if [[ "${LOCAL_PRELUDE}" == true ]]; then
        # ── Local mode: use pre-built binary directly ──
        local bin="${PROJECT_ROOT}/target/release/prelude-server"
        if [[ ! -x "${bin}" ]]; then
            echo "  Building prelude-server ..."
            (cd "${PROJECT_ROOT}" && cargo build -p prelude-server --release --features full) \
                || { echo "  ERROR: cargo build failed"; return 1; }
        fi

        local cuda_env=""
        [[ "${CPU_MODE}" == false ]] && cuda_env="CUDA_VISIBLE_DEVICES=${GPU}"

        # shellcheck disable=SC2086
        env ${cuda_env} "${bin}" \
            --model "${MODEL}" --host 0.0.0.0 --port "${PORT}" \
            ${extra} &
        LOCAL_PID=$!
        echo "  Local PID: ${LOCAL_PID}"
        return 0
    fi

    # ── Docker mode: pre-built image with binary baked in ──
    local image="prelude"
    if ! docker image inspect "${image}" &>/dev/null; then
        echo ">>> Building ${image} (this takes ~20min on first build) ..."
        docker build -f "${PROJECT_ROOT}/Dockerfile" -t "${image}" "${PROJECT_ROOT}"
    fi

    local gpu_flag=""
    [[ "${CPU_MODE}" == false ]] && gpu_flag="--gpus device=${GPU}"

    # shellcheck disable=SC2086
    docker run -d --name "${CONTAINER_NAME}" \
        ${gpu_flag} \
        -p "${PORT}:8000" \
        -v "${HF_CACHE}:/root/.cache/huggingface:ro" \
        "${image}" \
        --model "${MODEL}" ${extra}
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

## Build the shared "others" image (llama.cpp + vllm.rs, both git cloned inside)
ensure_others_image() {
    local image="prelude-others"
    if docker image inspect "${image}" &>/dev/null; then
        return 0
    fi
    echo ">>> Building ${image} (llama.cpp + vllm.rs, this takes a while) ..."
    docker build -f "${SCRIPT_DIR}/Dockerfile.others" -t "${image}" "${SCRIPT_DIR}"
}

start_vllm_rs() {
    ensure_others_image || return 1

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
        prelude-others \
        vllm-rs-server \
        --model "${MODEL}" \
        --host 0.0.0.0 \
        --port 8000
}

## Resolve GGUF model: auto-convert from HF safetensors if not cached.
## Result path: ~/.cache/prelude/gguf/<org>--<name>/model-BF16.gguf
resolve_gguf() {
    local model_safe="${MODEL//\//__}"  # Qwen/Qwen3-0.6B → Qwen__Qwen3-0.6B
    local gguf_dir="${GGUF_CACHE}/${model_safe}"
    local gguf_file="${gguf_dir}/model-BF16.gguf"

    if [[ -f "${gguf_file}" ]]; then
        echo "${gguf_file}"
        return 0
    fi

    # Find HF model directory
    local hf_model_dir=""
    local hf_safe="${MODEL//\//--}"  # Qwen/Qwen3-0.6B → Qwen--Qwen3-0.6B
    local hf_base="${HF_CACHE}/hub/models--${hf_safe}/snapshots"
    if [[ -d "${hf_base}" ]]; then
        for snap in "${hf_base}"/*/; do
            if [[ -f "${snap}/config.json" ]]; then
                hf_model_dir="${snap}"
                break
            fi
        done
    fi

    if [[ -z "${hf_model_dir}" ]]; then
        echo "  Downloading ${MODEL} ..." >&2
        huggingface-cli download "${MODEL}" --local-dir "${gguf_dir}/hf_tmp" >&2
        hf_model_dir="${gguf_dir}/hf_tmp"
    fi

    # Convert using llama.cpp's convert script
    local convert_script="${LLAMA_CPP_CONVERT:-}"
    if [[ -z "${convert_script}" ]]; then
        # Common locations
        for p in \
            /opt/llama.cpp/convert_hf_to_gguf.py \
            ../llama.cpp/convert_hf_to_gguf.py \
            /home/yuzhounie/src/llama.cpp/convert_hf_to_gguf.py; do
            if [[ -f "${p}" ]]; then
                convert_script="${p}"
                break
            fi
        done
    fi

    if [[ -z "${convert_script}" ]]; then
        echo "ERROR: convert_hf_to_gguf.py not found. Set LLAMA_CPP_CONVERT=/path/to/convert_hf_to_gguf.py" >&2
        return 1
    fi

    mkdir -p "${gguf_dir}"
    echo "  Converting ${MODEL} → GGUF (BF16) ..." >&2
    python3 "${convert_script}" "${hf_model_dir}" \
        --outfile "${gguf_file}" \
        --outtype bf16 >&2 || { echo "ERROR: GGUF conversion failed" >&2; return 1; }

    # Clean up temp download if we did one
    [[ -d "${gguf_dir}/hf_tmp" ]] && rm -rf "${gguf_dir}/hf_tmp"

    echo "${gguf_file}"
}

start_llama_cpp() {
    ensure_others_image || return 1

    local gguf_file
    gguf_file="$(resolve_gguf)" || return 1

    local model_dir
    model_dir="$(dirname "${gguf_file}")"
    local model_file
    model_file="$(basename "${gguf_file}")"

    local ngl=99
    local gpu_flag="--gpus device=${GPU}"
    if [[ "${CPU_MODE}" == true ]]; then
        ngl=0
        gpu_flag=""
    fi

    echo "  GGUF: ${gguf_file}"

    # shellcheck disable=SC2086
    docker run -d --name "${CONTAINER_NAME}" \
        ${gpu_flag} \
        -p "${PORT}:8000" \
        -v "${model_dir}:/models:ro" \
        prelude-others \
        llama-server \
        -m "/models/${model_file}" \
        --host 0.0.0.0 \
        --port 8000 \
        -ngl "${ngl}"
}

# ── Run one engine in one mode ─────────────────────────────────────────────

# GPU-only engines (skip CPU mode)
is_gpu_only() {
    [[ "$1" == "vllm.rs" ]]
}

bench_engine_mode() {
    local engine="$1" mode="$2"  # mode = GPU or CPU
    CPU_MODE=false
    [[ "${mode}" == "CPU" ]] && CPU_MODE=true

    if is_gpu_only "${engine}" && [[ "${CPU_MODE}" == true ]]; then
        echo "  SKIP: ${engine} does not support CPU mode"
        return 0
    fi

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

# Run an engine in the appropriate mode(s) based on --cpu/--gpu filter
bench_engine() {
    local engine="$1"
    if [[ "${DEVICE_FILTER}" == "cpu" ]]; then
        bench_engine_mode "${engine}" CPU
    elif [[ "${DEVICE_FILTER}" == "gpu" ]]; then
        bench_engine_mode "${engine}" GPU
    else
        # No filter: run both GPU and CPU
        bench_engine_mode "${engine}" GPU || true
        bench_engine_mode "${engine}" CPU || true
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────

echo "engine,device,startup_s,ttft_s,tpot_s,e2e_latency_s,input_tps,output_tps,rpm" > "${CSV_FILE}"

if [[ ${#ENGINES[@]} -eq 0 ]]; then
    echo "Usage: $0 <engine|all> [engine2 ...] [--cpu] [--gpu]"
    echo ""
    echo "Engines: prelude, vllm, vllm.rs, sglang, llama.cpp"
    echo "  all = all engines"
    echo ""
    echo "Flags:"
    echo "  --cpu    CPU only"
    echo "  --gpu    GPU only"
    echo "  --local  Run prelude locally (no Docker)"
    echo "  (none)   Both GPU and CPU"
    echo ""
    echo "Examples:"
    echo "  $0 prelude --local --gpu                # Prelude local, GPU"
    echo "  $0 all --gpu                            # All engines, GPU only"
    echo "  $0 prelude --cpu                        # Prelude CPU only"
    echo "  INPUT_TOKENS=512 $0 prelude vllm        # Compare prelude vs vllm"
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

# ── Summary table ─────────────────────────────────────────────────────────

python3 "${SCRIPT_DIR}/bench_utils.py" print-summary \
    --csv-file "${CSV_FILE}" --model "${MODEL}" --traffic "${TRAFFIC}" \
    --input-tokens "${INPUT_TOKENS}" --output-tokens "${OUTPUT_TOKENS}" \
    --concurrency "${CONCURRENCY}" --max-requests "${MAX_REQUESTS}" \
    --cpu-name "${CPU_NAME:-}" --gpu-name "${GPU_NAME:-}" --gpu-count "${GPU_COUNT}"

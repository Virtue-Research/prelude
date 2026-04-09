#!/usr/bin/env bash
# bench.sh — Benchmark Prelude vs vLLM / SGLang / vllm.rs / llama.cpp
#
# Usage:
#   ./benchmark/bench.sh                   # all engines
#   ./benchmark/bench.sh prelude --gpu     # Prelude GPU only
#   ./benchmark/bench.sh sglang --gpu      # SGLang GPU only
#   ./benchmark/bench.sh --gpu             # all GPU engines
#   ./benchmark/bench.sh --cpu             # all CPU engines
#
# Environment:
#   MODEL  INPUT_TOKENS  OUTPUT_TOKENS  CONCURRENCY  MAX_REQUESTS  CUDA_VISIBLE_DEVICES

set -uo pipefail  # No -e: individual engine failures should not abort the whole run
trap '' PIPE      # Ignore SIGPIPE (Docker commands can trigger it when piped)

# ── Config ──

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
INPUT_TOKENS="${INPUT_TOKENS:-32}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
CONCURRENCY="${CONCURRENCY:-1}"
MAX_REQUESTS="${MAX_REQUESTS:-10}"
MAX_TIME_MIN="${MAX_TIME_MIN:-20}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRELUDE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRELUDE_BIN="$PRELUDE_DIR/target/release/prelude-server"
VLLM_RS_BIN="${VLLM_RS_DIR:-$PRELUDE_DIR/../vllm.rs}/target/release/vllm-rs"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PRELUDE_DIR/../llama.cpp}"
LLAMA_CPP_BIN="${LLAMA_CPP_BIN:-$LLAMA_CPP_DIR/build/bin/llama-server}"
GGUF_MODEL="${GGUF_MODEL:-$LLAMA_CPP_DIR/models/$(basename "$MODEL")-BF16.gguf}"
GGUF_MODEL_ID="${GGUF_MODEL_ID:-$MODEL}"

TRAFFIC="D(${INPUT_TOKENS},${OUTPUT_TOKENS})"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$PRELUDE_DIR/bench_results/$TIMESTAMP}"
CSV_FILE="$RESULTS_DIR/summary.csv"

HAS_GPU=false; GPU_NAME=""; GPU_COUNT=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
fi
CPU_NAME=$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')

# ── Engine registry ──
# label|display|port|gpu_only|health_path|timeout|type(native|docker)

declare -A ENGINES
ENGINES=(
    [prelude]="prelude|Prelude|8099|no|/health|180|native"
    [prelude-gguf]="prelude-gguf|Prelude-GGUF|8098|no|/health|180|native"
    [vllm.rs]="vllm-rs|vLLM.rs|8002|yes|/v1/models|180|native"
    [vllm]="vllm|vLLM|8003|yes|/v1/models|300|docker"
    [vllm-cpu]="vllm-cpu|vLLM-CPU|8005|no|/v1/models|300|docker"
    [sglang]="sglang|SGLang|8004|yes|/v1/models|300|docker"
    [sglang-cpu]="sglang-cpu|SGLang-CPU|8006|no|/v1/models|300|docker"
    [llama.cpp]="llama-cpp|llama.cpp|8007|no|/health|120|native"
)

# Docker image per engine (only for docker-type engines)
declare -A DOCKER_IMAGES
DOCKER_IMAGES=(
    [vllm]="vllm/vllm-openai:latest-cu130"
    [vllm-cpu]="vllm/vllm-openai:latest"
    [sglang]="lmsysorg/sglang:latest-cu130"
    [sglang-cpu]="lmsysorg/sglang:latest"
)

# ── Helpers ──

log()  { echo -e "\033[1;34m[bench]\033[0m $*"; }
err()  { echo -e "\033[1;31m[bench ERROR]\033[0m $*" >&2; }
warn() { echo -e "\033[1;33m[bench WARN]\033[0m $*"; }

STARTUP_ELAPSED=0

wait_for_server() {
    local url="$1" name="$2" timeout="${3:-180}" pid="${4:-}" elapsed=0
    log "Waiting for $name at $url ..."
    while ! curl -sf --max-time 2 "$url" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed + 2))
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            err "$name process exited prematurely (pid=$pid)"
            STARTUP_ELAPSED="$elapsed"; return 1
        fi
        if [ "$elapsed" -ge "$timeout" ]; then
            err "$name did not start within ${timeout}s"
            STARTUP_ELAPSED="$elapsed"; return 1
        fi
    done
    STARTUP_ELAPSED="$elapsed"
    log "$name ready (${elapsed}s)"
}

kill_server() {
    local pid="$1" name="$2" container="${3:-}"
    [ -n "$container" ] && { docker stop "$container" 2>/dev/null || true; docker rm -f "$container" 2>/dev/null || true; }
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "Stopping $name (pid=$pid)"
        kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    fi
}

check_engine() {
    local engine="$1"
    case "$engine" in
        prelude)
            [ -f "$PRELUDE_BIN" ] || { echo "binary not built ($PRELUDE_BIN)"; return 1; } ;;
        prelude-gguf)
            [ -f "$PRELUDE_BIN" ] || { echo "binary not built ($PRELUDE_BIN)"; return 1; }
            [ -f "$GGUF_MODEL" ] || { echo "GGUF not found ($GGUF_MODEL)"; return 1; } ;;
        vllm.rs)
            [ -f "$VLLM_RS_BIN" ] || { echo "binary not built ($VLLM_RS_BIN)"; return 1; } ;;
        llama.cpp)
            [ -f "$LLAMA_CPP_BIN" ] || { echo "llama-server not found ($LLAMA_CPP_BIN)"; return 1; }
            [ -f "$GGUF_MODEL" ] || { echo "GGUF not found ($GGUF_MODEL)"; return 1; } ;;
        vllm|vllm-cpu|sglang|sglang-cpu)
            local img="${DOCKER_IMAGES[$engine]}"
            docker image inspect "$img" >/dev/null 2>&1 || { echo "Docker image not found: $img (docker pull $img)"; return 1; } ;;
    esac
    return 0
}

start_engine() {
    local engine="$1" port="$2"
    local hf_cache="${HOME}/.cache/huggingface"
    local cvd="${CUDA_VISIBLE_DEVICES:-0}"
    case "$engine" in
        prelude)
            env PRELUDE_DEVICE="$DEVICE" RUST_LOG="${RUST_LOG:-warn}" "$PRELUDE_BIN" \
                --host 0.0.0.0 --port "$port" --model "$MODEL" --dtype bf16 & ;;
        prelude-gguf)
            env PRELUDE_DEVICE=cpu RUST_LOG="${RUST_LOG:-warn}" "$PRELUDE_BIN" \
                --host 0.0.0.0 --port "$port" --model "$GGUF_MODEL_ID" & ;;
        vllm.rs)
            "$VLLM_RS_BIN" --m "$MODEL" --server --port "$port" & ;;
        vllm)
            docker run --rm --name vllm-bench --network=host --gpus all --ipc=host \
                -v "$hf_cache:/root/.cache/huggingface" -e "CUDA_VISIBLE_DEVICES=$cvd" \
                "$img" --model "$MODEL" --port "$port" --host 0.0.0.0 & ;;
        vllm-cpu)
            docker run --rm --name vllm-cpu-bench --network=host \
                -v "$hf_cache:/root/.cache/huggingface" \
                "$img" --model "$MODEL" --port "$port" --host 0.0.0.0 --device cpu & ;;
        sglang)
            docker run --rm --name sglang-bench --network=host --gpus all --ipc=host --shm-size 32g \
                -v "$hf_cache:/root/.cache/huggingface" -e "CUDA_VISIBLE_DEVICES=$cvd" \
                "$img" python3 -m sglang.launch_server \
                    --model-path "$MODEL" --port "$port" --host 0.0.0.0 & ;;
        sglang-cpu)
            docker run --rm --name sglang-cpu-bench --network=host \
                -v "$hf_cache:/root/.cache/huggingface" \
                "$img" python3 -m sglang.launch_server \
                    --model-path "$MODEL" --port "$port" --host 0.0.0.0 \
                    --device cpu --disable-overlap-schedule & ;;
        llama.cpp)
            "$LLAMA_CPP_BIN" -m "$GGUF_MODEL" \
                --host 0.0.0.0 --port "$port" --ctx-size 4096 --jinja --reasoning-budget 0 & ;;
    esac
}

# Docker container name for an engine (empty for native engines)
container_name() {
    case "$1" in
        vllm) echo "vllm-bench" ;; vllm-cpu) echo "vllm-cpu-bench" ;;
        sglang) echo "sglang-bench" ;; sglang-cpu) echo "sglang-cpu-bench" ;;
        *) echo "" ;;
    esac
}

# ── Generic engine runner ──

run_engine() {
    local engine="$1"; DEVICE="${2:-gpu}"
    IFS='|' read -r label display port gpu_only health_path timeout _type <<< "${ENGINES[$engine]}"

    [ "$gpu_only" = "yes" ] && [ "$DEVICE" = "cpu" ] && return
    local reason; reason=$(check_engine "$engine") || { warn "Skipping $display: $reason"; return; }

    log "Starting $display (device=$DEVICE) on port $port ..."
    start_engine "$engine" "$port"
    local pid=$! container; container=$(container_name "$engine")

    if wait_for_server "http://localhost:${port}${health_path}" "$display" "$timeout" "$pid"; then
        local startup_s="$STARTUP_ELAPSED" run_name="${label}-${DEVICE}"
        rm -rf "$RESULTS_DIR/$run_name"

        log "Running genai-bench: $run_name (traffic=$TRAFFIC concurrency=$CONCURRENCY)"
        genai-bench benchmark \
            --api-backend vllm --api-base "http://localhost:${port}" --api-key "none" \
            --api-model-name "$MODEL" --model-tokenizer "$MODEL" --task text-to-text \
            --max-time-per-run "$MAX_TIME_MIN" --max-requests-per-run "$MAX_REQUESTS" \
            --num-concurrency "$CONCURRENCY" --traffic-scenario "$TRAFFIC" \
            --server-engine "vLLM" --experiment-folder-name "$run_name" \
            --experiment-base-dir "$RESULTS_DIR" 2>&1 | tee "$RESULTS_DIR/${run_name}.log" || true

        local json_file
        json_file=$(find "$RESULTS_DIR/$run_name" -name '*.json' -not -name 'experiment_metadata.json' 2>/dev/null | head -1)
        if [ -n "$json_file" ] && [ -f "$json_file" ]; then
            python3 "$SCRIPT_DIR/bench_utils.py" extract-metrics \
                --json-file "$json_file" --engine "$display" --device "$DEVICE" \
                --startup-s "$startup_s" --csv-file "$CSV_FILE"
        else
            warn "No results JSON found for $run_name"
            echo "$display,$DEVICE,$startup_s,N/A,N/A,N/A,N/A,N/A,N/A" >> "$CSV_FILE"
        fi
    fi

    kill_server "$pid" "$display" "$container"
    sleep 2
}

# ── Main ──

command -v genai-bench &>/dev/null || { err "genai-bench not found (pip install genai-bench)"; exit 1; }

# Clean up leftover Docker containers
for c in vllm-bench vllm-cpu-bench sglang-bench sglang-cpu-bench; do
    docker rm -f "$c" 2>/dev/null || true
done

mkdir -p "$RESULTS_DIR"
echo "engine,device,startup_s,ttft_s,tpot_s,e2e_latency_s,input_tps,output_tps,rpm" > "$CSV_FILE"

# Parse args
FILTER=""; TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --cpu) FILTER="cpu" ;; --gpu) FILTER="gpu" ;;
        -*) err "Unknown flag: $arg"; exit 1 ;;
        *) TARGETS+=("$arg") ;;
    esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("all")

log "Config: model=$MODEL traffic=$TRAFFIC concurrency=$CONCURRENCY gpu=$HAS_GPU"
[ -n "$CPU_NAME" ] && log "CPU: $CPU_NAME"
[ -n "$GPU_NAME" ] && log "GPU: ${GPU_COUNT}x $GPU_NAME"
echo ""

# Dispatch
GPU_ENGINES=(prelude vllm.rs vllm sglang)
CPU_ENGINES=(prelude prelude-gguf llama.cpp vllm-cpu sglang-cpu)

run_single_engine() {
    local target="$1"
    if [ -z "${ENGINES[$target]+x}" ]; then
        err "Unknown engine: $target"
        echo "Available: ${!ENGINES[*]}"
        return 1
    fi
    IFS='|' read -r _ _ _ gpu_only _ <<< "${ENGINES[$target]}"
    if [ "$FILTER" = "gpu" ] || { [ "$FILTER" = "" ] && [ "$gpu_only" = "yes" ]; }; then
        [ "$HAS_GPU" = true ] && { [ "$target" = "prelude" ] && run_engine "$target" "cuda:0" || run_engine "$target" gpu; } \
            || warn "$target requires GPU"
    elif [ "$FILTER" = "cpu" ] || { [ "$FILTER" = "" ] && [ "$gpu_only" = "no" ]; }; then
        run_engine "$target" cpu
    else
        # No filter, engine supports both
        run_engine "$target" cpu
        [ "$HAS_GPU" = true ] && { [ "$target" = "prelude" ] && run_engine "$target" "cuda:0" || run_engine "$target" gpu; }
    fi
}

if [ "${TARGETS[0]}" = "all" ]; then
    [ "$FILTER" != "gpu" ] && for e in "${CPU_ENGINES[@]}"; do run_engine "$e" cpu; done
    [ "$FILTER" != "cpu" ] && [ "$HAS_GPU" = true ] && for e in "${GPU_ENGINES[@]}"; do
        [ "$e" = "prelude" ] && run_engine "$e" "cuda:0" || run_engine "$e" gpu
    done
else
    for target in "${TARGETS[@]}"; do
        run_single_engine "$target"
    done
fi

python3 "$SCRIPT_DIR/bench_utils.py" print-summary \
    --csv-file "$CSV_FILE" --model "$MODEL" --traffic "$TRAFFIC" \
    --input-tokens "$INPUT_TOKENS" --output-tokens "$OUTPUT_TOKENS" \
    --concurrency "$CONCURRENCY" --max-requests "$MAX_REQUESTS" \
    --cpu-name "$CPU_NAME" --gpu-name "$GPU_NAME" --gpu-count "$GPU_COUNT"

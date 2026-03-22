#!/usr/bin/env bash
# bench.sh — Benchmark Prelude vs vllm.rs / vLLM / SGLang using genai-bench
#
# Usage:
#   ./scripts/bench.sh                   # benchmark all engines
#   ./scripts/bench.sh prelude         # Prelude only
#   ./scripts/bench.sh vllm.rs           # vllm.rs only
#   ./scripts/bench.sh vllm              # vLLM (GPU) only
#   ./scripts/bench.sh vllm-cpu          # vLLM (CPU) only
#   ./scripts/bench.sh sglang            # SGLang only
#   ./scripts/bench.sh llama.cpp         # llama.cpp (CPU) only
#   ./scripts/bench.sh --cpu             # all CPU engines only
#   ./scripts/bench.sh --gpu             # all GPU engines only
#   ./scripts/bench.sh prelude --cpu   # Prelude CPU only
#
# Environment overrides:
#   MODEL=Qwen/Qwen3-0.6B  INPUT_TOKENS=64  OUTPUT_TOKENS=64  MAX_REQUESTS=20 ./scripts/bench.sh

set -euo pipefail

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
VLLM_RS_DIR="${VLLM_RS_DIR:-$PRELUDE_DIR/../vllm.rs}"
VLLM_RS_BIN="$VLLM_RS_DIR/target/release/vllm-rs"
VLLM_CPU_VENV="${VLLM_CPU_VENV:-$PRELUDE_DIR/../vllm-cpu/.venv}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PRELUDE_DIR/../llama.cpp}"
LLAMA_CPP_BIN="${LLAMA_CPP_BIN:-$LLAMA_CPP_DIR/build/bin/llama-server}"
GGUF_MODEL="${GGUF_MODEL:-$LLAMA_CPP_DIR/models/$(basename "$MODEL")-BF16.gguf}"
GGUF_MODEL_ID="${GGUF_MODEL_ID:-$MODEL}"  # HF repo for tokenizer (e.g. unsloth/Qwen3.5-0.8B)

TRAFFIC="D(${INPUT_TOKENS},${OUTPUT_TOKENS})"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$PRELUDE_DIR/bench_results/$TIMESTAMP}"
CSV_FILE="$RESULTS_DIR/summary.csv"

HAS_GPU=false
GPU_NAME=""
GPU_COUNT=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk 'NR==1' | xargs)
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | awk 'NR==1')
fi
CPU_NAME=$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')

# ── Engine registry ──
# Each engine: label|display_name|port|gpu_only|health_path|timeout

declare -A ENGINES
ENGINES=(
    [prelude]="prelude|Prelude|8099|no|/health|180"
    [prelude-gguf]="prelude-gguf|Prelude-GGUF|8098|no|/health|180"
    [vllm.rs]="vllm-rs|vLLM.rs|8002|yes|/v1/models|180"
    [vllm]="vllm|vLLM|8003|yes|/v1/models|300"
    [vllm-cpu]="vllm-cpu|vLLM-CPU|8005|no|/v1/models|300"
    [sglang]="sglang|SGLang|8004|yes|/v1/models|300"
    [sglang-cpu]="sglang-cpu|SGLang-CPU|8006|no|/v1/models|300"
    [llama.cpp]="llama-cpp|llama.cpp|8007|no|/health|120"
)

# ── Helpers ──

log()  { echo -e "\033[1;34m[bench]\033[0m $*"; }
err()  { echo -e "\033[1;31m[bench ERROR]\033[0m $*" >&2; }
warn() { echo -e "\033[1;33m[bench WARN]\033[0m $*"; }

STARTUP_ELAPSED=0  # set by wait_for_server for the caller to read

wait_for_server() {
    local url="$1" name="$2" timeout="${3:-180}" pid="${4:-}" elapsed=0
    log "Waiting for $name at $url ..."
    while ! curl -sf --max-time 2 "$url" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        # If we know the server PID, check it hasn't already exited
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            err "$name process exited prematurely (pid=$pid)"
            STARTUP_ELAPSED="$elapsed"
            return 1
        fi
        if [ "$elapsed" -ge "$timeout" ]; then
            err "$name did not start within ${timeout}s"
            STARTUP_ELAPSED="$elapsed"
            return 1
        fi
    done
    STARTUP_ELAPSED="$elapsed"
    log "$name ready (${elapsed}s)"
}

kill_server() {
    local pid="$1" name="$2"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "Stopping $name (pid=$pid)"
        # Stop Docker container gracefully (suppresses SGLang shutdown traceback)
        docker stop sglang-cpu-bench 2>/dev/null || true
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    # Clean up Docker container if applicable
    docker rm -f sglang-cpu-bench 2>/dev/null || true
}

start_engine() {
    local engine="$1" port="$2"
    case "$engine" in
        prelude)
            env PRELUDE_DEVICE="$DEVICE" RUST_LOG="${RUST_LOG:-warn}" "$PRELUDE_BIN" \
                --host 0.0.0.0 --port "$port" --model "$MODEL" --dtype bf16 &
            ;;
        prelude-gguf)
            env PRELUDE_DEVICE=cpu RUST_LOG="${RUST_LOG:-warn}" "$PRELUDE_BIN" \
                --host 0.0.0.0 --port "$port" --model "$GGUF_MODEL_ID" &
            ;;
        vllm.rs)
            "$VLLM_RS_BIN" --m "$MODEL" --server --port "$port" &
            ;;
        vllm)
            python3 -m vllm.entrypoints.openai.api_server \
                --model "$MODEL" --port "$port" --host 0.0.0.0 &
            ;;
        vllm-cpu)
            "$VLLM_CPU_VENV/bin/python3" -m vllm.entrypoints.openai.api_server \
                --model "$MODEL" --port "$port" --host 0.0.0.0 &
            ;;
        sglang)
            python3 -m sglang.launch_server \
                --model-path "$MODEL" --port "$port" --host 0.0.0.0 &
            ;;
        sglang-cpu)
            # Mount local sglang source for debug logging
            local sglang_src="${SGLANG_SRC:-}"
            local sglang_mount=""
            if [ -n "$sglang_src" ] && [ -d "$sglang_src/python/sglang" ]; then
                sglang_mount="-v ${sglang_src}/python/sglang:/opt/.venv/lib/python3.12/site-packages/sglang"
                log "Mounting local sglang from $sglang_src"
            fi
            docker run --rm --name sglang-cpu-bench --network=host \
                -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
                $sglang_mount \
                sglang-cpu:latest \
                bash -c 'source /opt/.venv/bin/activate && python -m sglang.launch_server \
                    --model-path "'"$MODEL"'" --port "'"$port"'" --host 0.0.0.0 \
                    --device cpu --disable-overlap-schedule' &
            ;;
        llama.cpp)
            "$LLAMA_CPP_BIN" -m "$GGUF_MODEL" \
                --host 0.0.0.0 --port "$port" --ctx-size 4096 \
                --jinja --reasoning-budget 0 &
            ;;
    esac
}

# ── Generic engine runner ──

run_engine() {
    local engine="$1"
    DEVICE="${2:-gpu}"

    IFS='|' read -r label display port gpu_only health_path timeout \
        <<< "${ENGINES[$engine]}"

    # GPU guard
    if [ "$gpu_only" = "yes" ] && [ "$DEVICE" = "cpu" ]; then
        warn "Skipping $display (cpu): requires GPU"
        return
    fi

    # Dependency check (unified via bench_utils.py)
    # NOTE: || true prevents set -e from aborting on non-zero exit
    local check_reason
    if ! check_reason=$(python3 "$SCRIPT_DIR/bench_utils.py" check-engine \
        --engine "$engine" \
        --prelude-bin "$PRELUDE_BIN" \
        --vllm-rs-bin "$VLLM_RS_BIN" \
        --vllm-cpu-venv "$VLLM_CPU_VENV" \
        --llama-cpp-bin "$LLAMA_CPP_BIN" \
        --gguf-model "$GGUF_MODEL" 2>/dev/null); then
        warn "Skipping $display: $check_reason"
        return
    fi

    log "Starting $display (device=$DEVICE) on port $port ..."
    start_engine "$engine" "$port"
    local pid=$!

    if wait_for_server "http://localhost:${port}${health_path}" "$display" "$timeout" "$pid"; then
        local startup_s="$STARTUP_ELAPSED"
        local run_name="${label}-${DEVICE}"
        local log_file="$RESULTS_DIR/${run_name}.log"

        # Clean stale results from previous runs so extract-metrics picks the right file
        rm -rf "$RESULTS_DIR/$run_name"

        log "Running genai-bench: $run_name (traffic=$TRAFFIC concurrency=$CONCURRENCY)"
        genai-bench benchmark \
            --api-backend vllm \
            --api-base "http://localhost:${port}" \
            --api-key "none" \
            --api-model-name "$MODEL" \
            --model-tokenizer "$MODEL" \
            --task text-to-text \
            --max-time-per-run "$MAX_TIME_MIN" \
            --max-requests-per-run "$MAX_REQUESTS" \
            --num-concurrency "$CONCURRENCY" \
            --traffic-scenario "$TRAFFIC" \
            --server-engine "vLLM" \
            --experiment-folder-name "$run_name" \
            --experiment-base-dir "$RESULTS_DIR" \
            2>&1 | tee "$log_file" || true

        # Extract metrics
        local json_file
        json_file=$(find "$RESULTS_DIR/$run_name" -name '*.json' \
            -not -name 'experiment_metadata.json' 2>/dev/null | head -1)

        if [ -n "$json_file" ] && [ -f "$json_file" ]; then
            python3 "$SCRIPT_DIR/bench_utils.py" extract-metrics \
                --json-file "$json_file" --engine "$display" --device "$DEVICE" \
                --startup-s "$startup_s" --csv-file "$CSV_FILE"
        else
            warn "No results JSON found in $RESULTS_DIR/$run_name"
            echo "$display,$DEVICE,$startup_s,N/A,N/A,N/A,N/A,N/A,N/A" >> "$CSV_FILE"
        fi
    fi

    kill_server "$pid" "$display"
    sleep 2
}

# ── Main ──

if ! command -v genai-bench &>/dev/null; then
    err "genai-bench not found (pip install genai-bench)"
    exit 1
fi

# Clean up leftover processes and containers from previous runs
docker rm -f sglang-cpu-bench 2>/dev/null || true
for spec in "${ENGINES[@]}"; do
    IFS='|' read -r _ _ port _ <<< "$spec"
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "Killing existing processes on port $port"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
done

mkdir -p "$RESULTS_DIR"
echo "engine,device,startup_s,ttft_s,tpot_s,e2e_latency_s,input_tps,output_tps,rpm" > "$CSV_FILE"

# Parse --cpu / --gpu flags and positional target
FILTER=""
target="all"
for arg in "$@"; do
    case "$arg" in
        --cpu) FILTER="cpu" ;;
        --gpu) FILTER="gpu" ;;
        -*) err "Unknown flag: $arg"; exit 1 ;;
        *) target="$arg" ;;
    esac
done
log "Config: model=$MODEL traffic=$TRAFFIC concurrency=$CONCURRENCY gpu=$HAS_GPU"
[ -n "$CPU_NAME" ] && log "CPU: $CPU_NAME"
[ -n "$GPU_NAME" ] && log "GPU: ${GPU_COUNT}x $GPU_NAME"
echo ""

case "$target" in
    prelude)
        [ "$FILTER" != "gpu" ] && run_engine prelude cpu
        [ "$FILTER" != "cpu" ] && [ "$HAS_GPU" = true ] && run_engine prelude "cuda:0"
        ;;
    vllm-cpu)
        [ "$FILTER" = "gpu" ] && warn "vllm-cpu is CPU-only, skipping with --gpu" || run_engine vllm-cpu cpu
        ;;
    sglang-cpu)
        [ "$FILTER" = "gpu" ] && warn "sglang-cpu is CPU-only, skipping with --gpu" || run_engine sglang-cpu cpu
        ;;
    prelude-gguf)
        [ "$FILTER" = "gpu" ] && warn "prelude-gguf is CPU-only, skipping with --gpu" || run_engine prelude-gguf cpu
        ;;
    llama.cpp)
        [ "$FILTER" = "gpu" ] && warn "llama.cpp is CPU-only, skipping with --gpu" || run_engine llama.cpp cpu
        ;;
    vllm.rs|vllm|sglang)
        [ "$FILTER" = "cpu" ] && warn "$target is GPU-only, skipping with --cpu" || \
            { [ "$HAS_GPU" = true ] && run_engine "$target" gpu || warn "$target requires GPU"; }
        ;;
    all)
        if [ "$FILTER" != "gpu" ]; then
            run_engine prelude cpu
            run_engine prelude-gguf cpu
            run_engine llama.cpp cpu
            run_engine vllm-cpu cpu
            run_engine sglang-cpu cpu
        fi
        if [ "$FILTER" != "cpu" ] && [ "$HAS_GPU" = true ]; then
            run_engine prelude "cuda:0"
            run_engine vllm.rs gpu
            run_engine vllm gpu
            run_engine sglang gpu
        fi
        ;;
    *)
        err "Unknown target: $target"
        echo "Usage: $0 [prelude|prelude-gguf|llama.cpp|vllm.rs|vllm|vllm-cpu|sglang|sglang-cpu|all] [--cpu|--gpu]"
        exit 1
        ;;
esac

python3 "$SCRIPT_DIR/bench_utils.py" print-summary \
    --csv-file "$CSV_FILE" --model "$MODEL" --traffic "$TRAFFIC" \
    --input-tokens "$INPUT_TOKENS" --output-tokens "$OUTPUT_TOKENS" \
    --concurrency "$CONCURRENCY" --max-requests "$MAX_REQUESTS" --has-gpu "$HAS_GPU" \
    --cpu-name "$CPU_NAME" --gpu-name "$GPU_NAME" --gpu-count "$GPU_COUNT"

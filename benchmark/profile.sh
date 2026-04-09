#!/usr/bin/env bash
# profile.sh — CUDA kernel profiling with Nsight Systems
#
# Captures GPU kernel traces during inference and prints per-kernel breakdown.
# Mirrors bench.sh engine registry — same engine names, start logic, Docker images.
#
# Usage (same env vars as bench.sh — just swap bench.sh → profile.sh):
#   ./benchmark/profile.sh prelude --gpu                   # Profile Prelude
#   ./benchmark/profile.sh prelude vllm sglang --gpu       # Profile all three
#   ./benchmark/profile.sh prelude --gpu --no-cuda-graph   # Individual kernel visibility
#   ./benchmark/profile.sh stats output.nsys-rep           # Re-analyze existing report
#
# Example (identical env as bench.sh):
#   CUDA_VISIBLE_DEVICES=2 INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
#     MODEL=Qwen/Qwen3-8B ./benchmark/profile.sh prelude vllm sglang --gpu
#
# Environment (same as bench.sh):
#   MODEL  INPUT_TOKENS  OUTPUT_TOKENS  MAX_REQUESTS  CONCURRENCY  CUDA_VISIBLE_DEVICES
#   WARMUP  (profile-specific, default: 5)

set -uo pipefail
trap '' PIPE

# ── Config (same env vars as bench.sh) ──

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
INPUT_TOKENS="${INPUT_TOKENS:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
MAX_REQUESTS="${MAX_REQUESTS:-20}"
CONCURRENCY="${CONCURRENCY:-1}"
WARMUP="${WARMUP:-5}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRELUDE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRELUDE_BIN="$PRELUDE_DIR/target/release/prelude-server"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$PRELUDE_DIR/bench_results/profile_$TIMESTAMP}"

# ── Engine registry (same as bench.sh) ──
# label|display|port|gpu_only|health_path|timeout|type

declare -A ENGINES
ENGINES=(
    [prelude]="prelude|Prelude|8099|no|/health|180|native"
    [vllm]="vllm|vLLM|8003|yes|/v1/models|300|docker"
    [sglang]="sglang|SGLang|8004|yes|/v1/models|300|docker"
)

declare -A DOCKER_IMAGES
DOCKER_IMAGES=(
    [vllm]="vllm/vllm-openai:latest-cu130"
    [sglang]="lmsysorg/sglang:latest-cu130"
)

# ── Helpers (from bench.sh) ──

log()  { echo -e "\033[1;34m[profile]\033[0m $*"; }
err()  { echo -e "\033[1;31m[profile ERROR]\033[0m $*" >&2; }

wait_for_server() {
    local url="$1" name="$2" timeout="${3:-180}" pid="${4:-}" elapsed=0
    log "Waiting for $name at $url ..."
    while ! curl -sf --max-time 2 "$url" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed + 2))
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            err "$name process exited prematurely (pid=$pid)"; return 1
        fi
        if [ "$elapsed" -ge "$timeout" ]; then
            err "$name did not start within ${timeout}s"; return 1
        fi
    done
    log "$name ready (${elapsed}s)"
}

kill_server() {
    local pid="$1" name="$2" container="${3:-}"
    if [ -n "$container" ]; then
        # Send SIGINT to nsys (PID 1) inside the container — nsys flushes
        # the .nsys-rep on SIGINT but may ignore SIGTERM.
        log "Stopping $name container ($container)..."
        docker kill -s SIGINT "$container" 2>/dev/null || true
        # Wait for nsys to flush and container to exit
        docker wait "$container" 2>/dev/null || sleep 15
        docker rm -f "$container" 2>/dev/null || true
    fi
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "Stopping $name (pid=$pid)"
        kill -INT "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    fi
}

container_name() {
    case "$1" in
        vllm) echo "vllm-profile" ;; sglang) echo "sglang-profile" ;;
        *) echo "" ;;
    esac
}

check_engine() {
    local engine="$1"
    case "$engine" in
        prelude) [ -f "$PRELUDE_BIN" ] || { echo "binary not built ($PRELUDE_BIN)"; return 1; } ;;
        vllm|sglang) ;;
    esac
    return 0
}

start_engine_under_nsys() {
    local engine="$1" port="$2" output="$3"
    local hf_cache="$HOME/.cache/huggingface"
    local cvd="${CUDA_VISIBLE_DEVICES:-0}"
    local img="${DOCKER_IMAGES[$engine]:-}"
    # Mount host nsys into container (images don't ship nsys).
    # The `nsys` on PATH is a wrapper; find the real installation directory.
    local nsys_bin nsys_dir
    nsys_bin=$(ls -1t /opt/nvidia/nsight-systems/*/target-linux-x64/nsys 2>/dev/null | head -1)
    [ -x "$nsys_bin" ] || { err "Cannot find nsys binary under /opt/nvidia/nsight-systems/"; return; }
    nsys_dir="$(dirname "$(dirname "$nsys_bin")")"

    case "$engine" in
        prelude)
            local nsys_flags=(-o "$output" --trace=cuda,nvtx --force-overwrite=true)
            local extra_env=(RUST_LOG="${RUST_LOG:-warn}")
            [ "$NO_CUDA_GRAPH" = true ] && extra_env+=(PRELUDE_CUDA_GRAPH_MAX_BS=0)
            env "${extra_env[@]}" nsys profile "${nsys_flags[@]}" \
                "$PRELUDE_BIN" --host 0.0.0.0 --port "$port" --model "$MODEL" --dtype bf16 &
            ;;
        vllm)
            # nsys runs INSIDE the container; host nsys mounted in via volume
            docker run --rm --name "$(container_name vllm)" \
                --network=host --gpus all --ipc=host \
                --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE \
                --entrypoint "$nsys_bin" \
                -v "$nsys_dir:$nsys_dir:ro" \
                -v "$hf_cache:/root/.cache/huggingface" \
                -v "$RESULTS_DIR:/profiles" \
                -e "CUDA_VISIBLE_DEVICES=$cvd" \
                "$img" \
                profile -o "/profiles/$engine" --trace=cuda,nvtx --force-overwrite=true \
                    vllm serve "$MODEL" --port "$port" --host 0.0.0.0 &
            ;;
        sglang)
            docker run --rm --name "$(container_name sglang)" \
                --network=host --gpus all --ipc=host --shm-size 32g \
                --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE \
                --entrypoint "$nsys_bin" \
                -v "$nsys_dir:$nsys_dir:ro" \
                -v "$hf_cache:/root/.cache/huggingface" \
                -v "$RESULTS_DIR:/profiles" \
                -e "CUDA_VISIBLE_DEVICES=$cvd" \
                "$img" \
                profile -o "/profiles/$engine" --trace=cuda,nvtx --force-overwrite=true \
                    python3 -m sglang.launch_server \
                    --model-path "$MODEL" --port "$port" --host 0.0.0.0 &
            ;;
    esac
}

# ── Stats ──

print_stats() {
    local report="$1" name="${2:-}"
    [ -f "$report" ] || { err "Not found: $report"; return 1; }

    echo ""
    [ -n "$name" ] && log "=== $name ==="
    log "CUDA Kernel Summary (sorted by total GPU time)"
    echo "────────────────────────────────────────────────────────────"
    nsys stats --report cuda_gpu_kern_sum "$report" 2>/dev/null || true

    echo ""
    log "CUDA Memory Operations"
    echo "────────────────────────────────────────────────────────────"
    nsys stats --report cuda_gpu_mem_size_sum "$report" 2>/dev/null || true

    echo ""
    log "Report: $report"
    log "GUI:    nsys-ui $report"
}

# ── Request sender ──

send_requests() {
    local port="$1" n="$2" pids=()
    for i in $(seq 1 "$n"); do
        curl -s "http://localhost:$port/v1/completions" \
            -H "Content-Type: application/json" -d "$BODY" > /dev/null &
        pids+=($!)
        if (( i % CONCURRENCY == 0 )); then
            wait "${pids[@]}" 2>/dev/null || true
            pids=()
        fi
    done
    [ ${#pids[@]} -gt 0 ] && { wait "${pids[@]}" 2>/dev/null || true; }
}

# ── Profile single engine ──

profile_engine() {
    local engine="$1"
    IFS='|' read -r label display port gpu_only health_path timeout _type <<< "${ENGINES[$engine]}"

    local reason; reason=$(check_engine "$engine") || { err "Skipping $display: $reason"; return; }

    local img="${DOCKER_IMAGES[$engine]:-}"
    if [ -n "$img" ]; then
        log "Pulling $img ..."
        docker pull "$img" || { err "Skipping $display: docker pull failed"; return; }
    fi

    local output="$RESULTS_DIR/$label"
    log "Starting $display under nsys on port $port ..."
    start_engine_under_nsys "$engine" "$port" "$output"
    local pid=$! container; container=$(container_name "$engine")

    if wait_for_server "http://localhost:${port}${health_path}" "$display" "$timeout" "$pid"; then
        log "Warmup ($WARMUP requests)..."
        send_requests "$port" "$WARMUP"

        log "Profiling $display ($MAX_REQUESTS requests, in=$INPUT_TOKENS out=$OUTPUT_TOKENS c=$CONCURRENCY)..."
        send_requests "$port" "$MAX_REQUESTS"
    fi

    kill_server "$pid" "$display" "$container"
    sleep 2

    # Find and print report
    local report=""
    for f in "$output.nsys-rep" "$output"; do
        [ -f "$f" ] && { report="$f"; break; }
    done
    [ -n "$report" ] && print_stats "$report" "$display"
}

# ── Main ──

command -v nsys &>/dev/null || { err "nsys not found (install NVIDIA Nsight Systems from CUDA toolkit)"; exit 1; }

# Clean up leftover Docker containers
for c in vllm-profile sglang-profile; do
    docker rm -f "$c" 2>/dev/null || true
done

# Parse args
NO_CUDA_GRAPH=false
TARGETS=()
STATS_FILE=""
MODE="profile"

for arg in "$@"; do
    case "$arg" in
        --no-cuda-graph) NO_CUDA_GRAPH=true ;;
        --gpu)           ;; # accepted for bench.sh compatibility, GPU is default
        --cu12)
            DOCKER_IMAGES[vllm]="vllm/vllm-openai:latest"
            DOCKER_IMAGES[sglang]="lmsysorg/sglang:latest"
            ;;
        stats)           MODE="stats" ;;
        *.nsys-rep)      STATS_FILE="$arg" ;;
        -h|--help)       sed -n '2,/^[^#]/{ /^#/s/^# \?//p }' "$0"; exit 0 ;;
        *)
            if [ -n "${ENGINES[$arg]+x}" ]; then
                TARGETS+=("$arg")
            else
                err "Unknown engine or flag: $arg"; exit 1
            fi
            ;;
    esac
done

if [ "$MODE" = "stats" ]; then
    [ -z "$STATS_FILE" ] && { err "Usage: $0 stats <file.nsys-rep>"; exit 1; }
    print_stats "$STATS_FILE"
    exit 0
fi

[ ${#TARGETS[@]} -eq 0 ] && { err "Usage: $0 <prelude|vllm|sglang> ... [--no-cuda-graph] [--gpu] [--cu12]"; exit 1; }

# Generate prompt (~INPUT_TOKENS tokens)
PROMPT=$(yes hello 2>/dev/null | head -n "$INPUT_TOKENS" | tr '\n' ' ')
BODY="{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":$OUTPUT_TOKENS,\"temperature\":0}"

mkdir -p "$RESULTS_DIR"

log "Config: model=$MODEL in=$INPUT_TOKENS out=$OUTPUT_TOKENS requests=$MAX_REQUESTS concurrency=$CONCURRENCY"
[ "$NO_CUDA_GRAPH" = true ] && log "CUDA graphs disabled (kernel-level visibility)"
echo ""

for target in "${TARGETS[@]}"; do
    profile_engine "$target"
    echo ""
done

log "All reports in: $RESULTS_DIR"

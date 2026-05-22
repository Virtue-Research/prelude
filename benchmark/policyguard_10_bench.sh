#!/usr/bin/env bash
# policyguard_first10_bench.sh
#
# Benchmark Prelude/vLLM with the first 10 samples from PolicyGuardData.
# Defaults:
#   MODEL=Virtue-AI-HUB/topicguard-qwen3-15b-bf16-272k-compact-nouserprefix
#   DATASET_URL=https://huggingface.co/datasets/Virtue-AI-HUB/PolicyGuardData/blob/main/jingyang/labeled_272k_20260513_compact_nouserprefix_chatml.jsonl
#   MAX_TOKENS=3
#
# Usage:
#   ./benchmark/policyguard_first10_bench.sh                 # prelude + vllm on GPU (if available)
#   ./benchmark/policyguard_first10_bench.sh prelude --gpu
#   ./benchmark/policyguard_first10_bench.sh vllm --gpu
#   ./benchmark/policyguard_first10_bench.sh prelude --cpu

set -uo pipefail
trap '' PIPE

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRELUDE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-Virtue-AI-HUB/topicguard-qwen3-15b-bf16-272k-compact-nouserprefix}"
DATASET_URL="${DATASET_URL:-https://huggingface.co/datasets/Virtue-AI-HUB/PolicyGuardData/blob/main/jingyang/labeled_272k_20260513_compact_nouserprefix_chatml.jsonl}"
SAMPLES="${SAMPLES:-10}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-5}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
RANDOM_SEED="${RANDOM_SEED:-20260521}"
MAX_TOKENS="${MAX_TOKENS:-3}"
TEMPERATURE="${TEMPERATURE:-0}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-120}"
HF_TOKEN="$(cat /scratch/xueying/.cache/huggingface/token)"
MODEL_PATH="${MODEL_PATH:-}"
PRELUDE_EXTRA_ARGS="${PRELUDE_EXTRA_ARGS:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$PRELUDE_DIR/bench_results/policyguard_first10_$TIMESTAMP}"
CSV_FILE="$RESULTS_DIR/summary.csv"

PRELUDE_BIN="${PRELUDE_BIN:-$PRELUDE_DIR/target/release/prelude-server}"
HF_DOCKER_CACHE="${HF_DOCKER_CACHE:-${HOME}/.cache/huggingface-docker}"

HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
fi

declare -A ENGINES
ENGINES=(
    [prelude]="Prelude|8099|no|/health|240|native"
    [vllm]="vLLM|8003|yes|/v1/models|420|docker"
)

declare -A DOCKER_IMAGES
DOCKER_IMAGES=(
    [vllm]="vllm/vllm-openai:latest-cu130"
)

log()  { echo -e "\033[1;34m[policyguard]\033[0m $*"; }
warn() { echo -e "\033[1;33m[policyguard WARN]\033[0m $*"; }
err()  { echo -e "\033[1;31m[policyguard ERROR]\033[0m $*" >&2; }

STARTUP_ELAPSED=0

append_failure_row() {
    local display="$1" device="$2" startup_s="${3:-0}"
    echo "${display},${device},${startup_s},0,0,N/A,N/A,N/A,${SAMPLES}" >> "$CSV_FILE"
}

select_idle_gpu() {
    local max_mem_mb="${IDLE_GPU_MAX_MEM_MB:-1024}"
    local max_util_pct="${IDLE_GPU_MAX_UTIL_PCT:-10}"
    local idx mem util chosen=""

    while IFS=',' read -r idx mem util; do
        idx="$(echo "$idx" | xargs)"
        mem="$(echo "$mem" | xargs)"
        util="$(echo "$util" | xargs)"
        [ -z "$idx" ] && continue
        [ -z "$mem" ] && continue
        [ -z "$util" ] && continue
        if [ "$mem" -le "$max_mem_mb" ] && [ "$util" -le "$max_util_pct" ]; then
            chosen="$idx"
            break
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)

    if [ -n "$chosen" ]; then
        echo "$chosen"
        return 0
    fi
    return 1
}

wait_for_server() {
    local url="$1" name="$2" timeout="${3:-180}" pid="${4:-}" elapsed=0
    log "Waiting for $name at $url ..."
    while ! curl -sf --max-time 2 "$url" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            err "$name exited early (pid=$pid)"
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

container_name() {
    case "$1" in
        vllm) echo "vllm-policyguard-bench" ;;
        *) echo "" ;;
    esac
}

kill_server() {
    local pid="$1" name="$2" container="${3:-}"
    if [ -n "$container" ]; then
        docker stop "$container" 2>/dev/null || true
        docker rm -f "$container" 2>/dev/null || true
    fi
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "Stopping $name (pid=$pid)"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

check_engine() {
    local engine="$1"
    case "$engine" in
        prelude)
            [ -f "$PRELUDE_BIN" ] || { echo "prelude binary not found: $PRELUDE_BIN"; return 1; }
            ;;
        vllm)
            ;;
        *)
            echo "unknown engine: $engine"
            return 1
            ;;
    esac
    return 0
}

start_engine() {
    local engine="$1" port="$2" device="$3" img="$4"
    local cvd="${CUDA_VISIBLE_DEVICES:-0}"
    mkdir -p "$HF_DOCKER_CACHE"
    local model_path_args=()
    if [ -n "$MODEL_PATH" ]; then
        model_path_args=(--model-path "$MODEL_PATH")
    fi

    case "$engine" in
        prelude)
            if [ -n "$HF_TOKEN" ]; then
                # shellcheck disable=SC2086
                env PRELUDE_DEVICE="$device" RUST_LOG="${RUST_LOG:-warn}" HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" "$PRELUDE_BIN" \
                    --host 0.0.0.0 --port "$port" --model "$MODEL" --dtype bf16 "${model_path_args[@]}" ${PRELUDE_EXTRA_ARGS} &
            else
                # shellcheck disable=SC2086
                env PRELUDE_DEVICE="$device" RUST_LOG="${RUST_LOG:-warn}" "$PRELUDE_BIN" \
                    --host 0.0.0.0 --port "$port" --model "$MODEL" --dtype bf16 "${model_path_args[@]}" ${PRELUDE_EXTRA_ARGS} &
            fi
            ;;
        vllm)
            if [ -n "$HF_TOKEN" ]; then
                # shellcheck disable=SC2086
                docker run --rm --name vllm-policyguard-bench --network=host --gpus all --ipc=host \
                    -v "$HF_DOCKER_CACHE:/root/.cache/huggingface" \
                    -e "CUDA_VISIBLE_DEVICES=$cvd" -e "HF_TOKEN=$HF_TOKEN" -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
                    "$img" --model "$MODEL" --port "$port" --host 0.0.0.0 ${VLLM_EXTRA_ARGS} &
            else
                # shellcheck disable=SC2086
                docker run --rm --name vllm-policyguard-bench --network=host --gpus all --ipc=host \
                    -v "$HF_DOCKER_CACHE:/root/.cache/huggingface" -e "CUDA_VISIBLE_DEVICES=$cvd" \
                    "$img" --model "$MODEL" --port "$port" --host 0.0.0.0 ${VLLM_EXTRA_ARGS} &
            fi
            ;;
    esac
}

run_policyguard_eval() {
    local api_base="$1" run_name="$2" display="$3" device="$4" startup_s="$5"
    local run_dir="$RESULTS_DIR/$run_name"
    local out_jsonl="$run_dir/predictions.jsonl"
    local stats_json="$run_dir/stats.json"

    mkdir -p "$run_dir"
    log "Running first ${SAMPLES} samples: $display ($device), max_tokens=$MAX_TOKENS"
    if [ "$WARMUP_SAMPLES" -gt 0 ]; then
        log "Warmup samples: $WARMUP_SAMPLES (not counted in summary)"
    fi

    API_BASE="$api_base" \
    MODEL_NAME="$MODEL" \
    DATASET_URL_VALUE="$DATASET_URL" \
    OUTPUT_JSONL="$out_jsonl" \
    STATS_JSON="$stats_json" \
    MAX_TOKENS_VALUE="$MAX_TOKENS" \
    TEMPERATURE_VALUE="$TEMPERATURE" \
    SAMPLES_VALUE="$SAMPLES" \
    WARMUP_SAMPLES_VALUE="$WARMUP_SAMPLES" \
    SAMPLE_MODE_VALUE="$SAMPLE_MODE" \
    RANDOM_SEED_VALUE="$RANDOM_SEED" \
    REQUEST_TIMEOUT_S_VALUE="$REQUEST_TIMEOUT_S" \
    HF_TOKEN_VALUE="$HF_TOKEN" \
    python3 - <<'PY'
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request


def normalize_dataset_url(url: str) -> str:
    if "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def load_jsonl_rows(url: str, n: int, sample_mode: str, random_seed: int):
    headers = {"User-Agent": "policyguard-bench"}
    hf_token = os.environ.get("HF_TOKEN_VALUE", "")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    req = urllib.request.Request(normalize_dataset_url(url), headers=headers)
    if sample_mode not in {"first", "random"}:
        raise ValueError(f"unsupported SAMPLE_MODE={sample_mode!r}, expected 'first' or 'random'")

    rng = random.Random(random_seed)
    rows = []
    seen = 0
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            item = json.loads(line)
            if sample_mode == "first":
                if len(rows) < n:
                    rows.append(item)
                else:
                    break
            else:
                # Reservoir sampling: uniform random sample without loading full file.
                seen += 1
                if len(rows) < n:
                    rows.append(item)
                else:
                    j = rng.randint(1, seen)
                    if j <= n:
                        rows[j - 1] = item
    return rows


def to_messages(item):
    if isinstance(item, dict) and isinstance(item.get("messages"), list) and item["messages"]:
        return item["messages"]
    if isinstance(item, dict):
        for key in ("prompt", "input", "text", "question"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return [{"role": "user", "content": val}]
    return [{"role": "user", "content": json.dumps(item, ensure_ascii=False)}]


def call_chat(api_base, model, messages, max_tokens, temperature, timeout_s):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - start) * 1000.0
    choices = data.get("choices", [])
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
    return data, text, latency_ms


api_base = os.environ["API_BASE"].rstrip("/")
model = os.environ["MODEL_NAME"]
dataset_url = os.environ["DATASET_URL_VALUE"]
output_jsonl = os.environ["OUTPUT_JSONL"]
stats_json = os.environ["STATS_JSON"]
max_tokens = int(os.environ["MAX_TOKENS_VALUE"])
temperature = float(os.environ["TEMPERATURE_VALUE"])
samples = int(os.environ["SAMPLES_VALUE"])
warmup_samples = int(os.environ.get("WARMUP_SAMPLES_VALUE", "0"))
sample_mode = os.environ.get("SAMPLE_MODE_VALUE", "first")
random_seed = int(os.environ.get("RANDOM_SEED_VALUE", "42"))
timeout_s = int(os.environ["REQUEST_TIMEOUT_S_VALUE"])

rows = load_jsonl_rows(
    dataset_url,
    samples + warmup_samples,
    sample_mode=sample_mode,
    random_seed=random_seed,
)
warmup_rows = rows[:warmup_samples]
eval_rows = rows[warmup_samples : warmup_samples + samples]
latencies = []
successes = 0
errors = 0

# Warmup requests are sent but excluded from final stats/output file.
for row in warmup_rows:
    messages = to_messages(row)
    try:
        call_chat(
            api_base=api_base,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    except Exception:
        # Warmup failures should not fail the run; evaluation phase remains authoritative.
        pass

with open(output_jsonl, "w", encoding="utf-8") as fout:
    for idx, row in enumerate(eval_rows):
        messages = to_messages(row)
        record = {
            "index": idx,
            "request_messages": messages,
        }
        try:
            raw_resp, output_text, latency_ms = call_chat(
                api_base=api_base,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            latencies.append(latency_ms)
            successes += 1
            record.update(
                {
                    "ok": True,
                    "latency_ms": latency_ms,
                    "output_text": output_text,
                    "raw_response": raw_resp,
                }
            )
        except urllib.error.HTTPError as e:
            errors += 1
            record.update(
                {
                    "ok": False,
                    "error_type": "HTTPError",
                    "error_message": f"{e.code} {e.reason}",
                }
            )
        except Exception as e:
            errors += 1
            record.update(
                {
                    "ok": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if latencies:
    avg_ms = statistics.fmean(latencies)
    p50_ms = statistics.median(latencies)
    p95_ms = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
else:
    avg_ms = p50_ms = p95_ms = None

stats = {
    "samples_requested": samples,
    "sample_mode": sample_mode,
    "random_seed": random_seed,
    "warmup_requested": warmup_samples,
    "warmup_loaded": len(warmup_rows),
    "samples_loaded": len(eval_rows),
    "successes": successes,
    "errors": errors,
    "avg_latency_ms": avg_ms,
    "p50_latency_ms": p50_ms,
    "p95_latency_ms": p95_ms,
}

with open(stats_json, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
PY

    STATS_JSON="$stats_json" \
    CSV_FILE="$CSV_FILE" \
    ENGINE_DISPLAY="$display" \
    DEVICE_NAME="$device" \
    STARTUP_S="$startup_s" \
    python3 - <<'PY'
import json
import os

stats_file = os.environ["STATS_JSON"]
csv_file = os.environ["CSV_FILE"]
engine = os.environ["ENGINE_DISPLAY"]
device = os.environ["DEVICE_NAME"]
startup_s = os.environ["STARTUP_S"]

with open(stats_file, "r", encoding="utf-8") as f:
    stats = json.load(f)

def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

row = ",".join(
    [
        engine,
        device,
        str(startup_s),
        str(stats.get("samples_loaded", 0)),
        str(stats.get("successes", 0)),
        fmt(stats.get("avg_latency_ms")),
        fmt(stats.get("p50_latency_ms")),
        fmt(stats.get("p95_latency_ms")),
        str(stats.get("errors", 0)),
    ]
)

with open(csv_file, "a", encoding="utf-8") as f:
    f.write(row + "\n")
PY
}

run_engine() {
    local engine="$1" device="$2"
    IFS='|' read -r display port gpu_only health_path timeout _type <<< "${ENGINES[$engine]}"
    if [ "$gpu_only" = "yes" ] && [ "$device" = "cpu" ]; then
        warn "Skipping $display on CPU: GPU only"
        return
    fi

    local reason
    reason=$(check_engine "$engine") || {
        warn "Skipping $display: $reason"
        append_failure_row "$display" "$device" 0
        return
    }

    local img="${DOCKER_IMAGES[$engine]:-}"
    if [ -n "$img" ]; then
        log "Pulling $img ..."
        docker pull "$img" || {
            warn "Skipping $display: docker pull failed"
            append_failure_row "$display" "$device" 0
            return
        }
    fi

    local run_name="${engine}-${device}"
    local pid container
    container="$(container_name "$engine")"

    log "Starting $display (device=$device) on port $port ..."
    start_engine "$engine" "$port" "$device" "$img"
    pid=$!

    if wait_for_server "http://localhost:${port}${health_path}" "$display" "$timeout" "$pid"; then
        if ! run_policyguard_eval "http://localhost:${port}" "$run_name" "$display" "$device" "$STARTUP_ELAPSED"; then
            warn "$display eval failed"
            append_failure_row "$display" "$device" "$STARTUP_ELAPSED"
        fi
    else
        append_failure_row "$display" "$device" "$STARTUP_ELAPSED"
    fi

    kill_server "$pid" "$display" "$container"
    sleep 2
}

print_sample_preview() {
    DATASET_URL_VALUE="$DATASET_URL" \
    SAMPLES_VALUE="$SAMPLES" \
    WARMUP_SAMPLES_VALUE="$WARMUP_SAMPLES" \
    SAMPLE_MODE_VALUE="$SAMPLE_MODE" \
    RANDOM_SEED_VALUE="$RANDOM_SEED" \
    HF_TOKEN_VALUE="$HF_TOKEN" \
    python3 - <<'PY'
import json
import os
import random
import urllib.request


def normalize_dataset_url(url: str) -> str:
    if "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def load_jsonl_rows(url: str, n: int, sample_mode: str, random_seed: int):
    headers = {"User-Agent": "policyguard-bench-preview"}
    hf_token = os.environ.get("HF_TOKEN_VALUE", "")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    req = urllib.request.Request(normalize_dataset_url(url), headers=headers)
    if sample_mode not in {"first", "random"}:
        raise ValueError(f"unsupported SAMPLE_MODE={sample_mode!r}, expected 'first' or 'random'")

    rng = random.Random(random_seed)
    rows = []
    seen = 0
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            item = json.loads(line)
            if sample_mode == "first":
                if len(rows) < n:
                    rows.append(item)
                else:
                    break
            else:
                seen += 1
                if len(rows) < n:
                    rows.append(item)
                else:
                    j = rng.randint(1, seen)
                    if j <= n:
                        rows[j - 1] = item
    return rows


def one_line(s: str, n: int = 140) -> str:
    s = " ".join(s.split())
    return s[:n] + ("..." if len(s) > n else "")


samples = int(os.environ["SAMPLES_VALUE"])
warmup = int(os.environ["WARMUP_SAMPLES_VALUE"])
sample_mode = os.environ.get("SAMPLE_MODE_VALUE", "random")
seed = int(os.environ.get("RANDOM_SEED_VALUE", "20260521"))
dataset_url = os.environ["DATASET_URL_VALUE"]

rows = load_jsonl_rows(dataset_url, samples + warmup, sample_mode=sample_mode, random_seed=seed)
eval_rows = rows[warmup : warmup + samples]

print("")
print("=== Sample Preview (evaluation set) ===")
print(f"mode={sample_mode} seed={seed} warmup={warmup} eval={len(eval_rows)}")
for i, row in enumerate(eval_rows, start=1):
    msgs = row.get("messages", [])
    user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    assistant = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    label = assistant.split("|", 1)[0] if assistant else ""
    print(f"{i:02d}\t{label}\t{one_line(user)}")
PY
}

print_summary() {
    CSV_FILE_VALUE="$CSV_FILE" python3 - <<'PY'
import csv
import os

csv_file = os.environ["CSV_FILE_VALUE"]
print("")
print("=== PolicyGuard First10 Summary ===")
with open(csv_file, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("No rows in summary.csv")
    raise SystemExit(0)

headers = [
    "engine",
    "device",
    "startup_s",
    "samples",
    "successes",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "errors",
]
print(",".join(headers))
for row in rows:
    print(",".join([row.get(h, "") for h in headers]))
PY
}

command -v curl &>/dev/null || { err "curl not found"; exit 1; }
command -v python3 &>/dev/null || { err "python3 not found"; exit 1; }

mkdir -p "$RESULTS_DIR"
echo "engine,device,startup_s,samples,successes,avg_latency_ms,p50_latency_ms,p95_latency_ms,errors" > "$CSV_FILE"

for c in vllm-policyguard-bench; do
    docker rm -f "$c" 2>/dev/null || true
done

FILTER="gpu"
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --cpu) FILTER="cpu" ;;
        --gpu) FILTER="gpu" ;;
        --cu12) DOCKER_IMAGES[vllm]="vllm/vllm-openai:latest" ;;
        -*) err "Unknown flag: $arg"; exit 1 ;;
        *) TARGETS+=("$arg") ;;
    esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("all")

if [ "$FILTER" != "cpu" ]; then
    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
        echo "no idle gpu"
        exit 1
    fi
    IDLE_GPU="$(select_idle_gpu)" || {
        echo "no idle gpu"
        exit 1
    }
    export CUDA_VISIBLE_DEVICES="$IDLE_GPU"
fi

log "Model: $MODEL"
log "Dataset: $DATASET_URL"
log "Samples: $SAMPLES (+warmup=$WARMUP_SAMPLES), max_tokens=$MAX_TOKENS, temperature=$TEMPERATURE"
log "Sample mode: $SAMPLE_MODE (seed=$RANDOM_SEED)"
if [ "$FILTER" != "cpu" ]; then
    log "Selected idle GPU: $CUDA_VISIBLE_DEVICES"
fi
if [ -n "$HF_TOKEN" ]; then
    log "HF token: detected"
else
    warn "HF token not set (HF_TOKEN/HUGGING_FACE_HUB_TOKEN). Private/gated model or dataset may fail with 401."
fi
log "Results dir: $RESULTS_DIR"
print_sample_preview

run_single() {
    local target="$1"
    if [ -z "${ENGINES[$target]+x}" ]; then
        err "Unknown engine: $target"
        echo "Available: ${!ENGINES[*]}"
        return 1
    fi

    if [ "$FILTER" = "cpu" ]; then
        run_engine "$target" cpu
    else
        [ "$target" = "prelude" ] && run_engine "$target" "cuda:0" || run_engine "$target" gpu
    fi
}

if [ "${TARGETS[0]}" = "all" ]; then
    if [ "$FILTER" = "cpu" ]; then
        run_engine prelude cpu
    else
        run_engine prelude "cuda:0"
        run_engine vllm gpu
    fi
else
    for t in "${TARGETS[@]}"; do
        run_single "$t"
    done
fi

print_summary
log "Done. Summary CSV: $CSV_FILE"

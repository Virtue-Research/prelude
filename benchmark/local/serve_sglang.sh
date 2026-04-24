#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUNTIME_DIR="${RUNTIME_DIR:-$ROOT_DIR/bench_results/runtime}"
PID_FILE="$RUNTIME_DIR/sglang.pid"
LOG_FILE="$RUNTIME_DIR/sglang.log"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
HOST="${HOST:-0.0.0.0}"
START_TIMEOUT_S="${START_TIMEOUT_S:-180}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

mkdir -p "$RUNTIME_DIR"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "sglang already running (pid=$existing_pid). log=$LOG_FILE"
    exit 0
  fi
fi

cmd=(python3 -m sglang.launch_server
  --model-path "$MODEL"
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "$SGLANG_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $SGLANG_EXTRA_ARGS )
  cmd+=("${extra_args[@]}")
fi

echo "Starting SGLang"
echo "  model: $MODEL"
echo "  host:  $HOST"
echo "  port:  $PORT"
echo "  gpu:   $GPU"

CUDA_VISIBLE_DEVICES="$GPU" nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &
pid="$!"
echo "$pid" > "$PID_FILE"

for ((i=0; i<START_TIMEOUT_S; i++)); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || \
     curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "SGLang ready: http://localhost:${PORT} (pid=$pid)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: SGLang process exited early. log=$LOG_FILE"
    exit 1
  fi
  sleep 1
done

echo "ERROR: SGLang did not become healthy in ${START_TIMEOUT_S}s. log=$LOG_FILE"
exit 1

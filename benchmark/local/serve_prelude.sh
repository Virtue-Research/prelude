#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUNTIME_DIR="${RUNTIME_DIR:-$ROOT_DIR/bench_results/runtime}"
PID_FILE="$RUNTIME_DIR/prelude.pid"
LOG_FILE="$RUNTIME_DIR/prelude.log"
BIN="${PRELUDE_BIN:-$ROOT_DIR/target/release/prelude-server}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
HOST="${HOST:-0.0.0.0}"
START_TIMEOUT_S="${START_TIMEOUT_S:-120}"
PRELUDE_EXTRA_ARGS="${PRELUDE_EXTRA_ARGS:-}"

mkdir -p "$RUNTIME_DIR"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "prelude already running (pid=$existing_pid). log=$LOG_FILE"
    exit 0
  fi
fi

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: prelude binary not found: $BIN"
  echo "Build first: cargo build -p prelude-server --release --features cuda"
  exit 1
fi

echo "Starting prelude-server"
echo "  bin:   $BIN"
echo "  model: $MODEL"
echo "  host:  $HOST"
echo "  port:  $PORT"
echo "  gpu:   $GPU"

cmd=(
  "$BIN"
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "$PRELUDE_EXTRA_ARGS" ]]; then
  # Intentionally split on shell words for flags like:
  # "--max-num-batched-tokens 16384 --max-batch-wait-ms 2"
  # shellcheck disable=SC2206
  extra_args=( $PRELUDE_EXTRA_ARGS )
  cmd+=("${extra_args[@]}")
fi

CUDA_VISIBLE_DEVICES="$GPU" nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

for ((i=0; i<START_TIMEOUT_S; i++)); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "prelude ready: http://localhost:${PORT} (pid=$pid)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: prelude process exited early. log=$LOG_FILE"
    exit 1
  fi
  sleep 1
done

echo "ERROR: prelude did not become healthy in ${START_TIMEOUT_S}s. log=$LOG_FILE"
exit 1

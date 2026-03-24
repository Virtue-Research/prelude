#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUNTIME_DIR="${RUNTIME_DIR:-$ROOT_DIR/bench_results/runtime}"
PID_FILE="$RUNTIME_DIR/vllm.pid"
LOG_FILE="$RUNTIME_DIR/vllm.log"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
HOST="${HOST:-0.0.0.0}"
MODE="${MODE:-completion}"   # completion | classify | embed
START_TIMEOUT_S="${START_TIMEOUT_S:-180}"
VLLM_SERVE_SCRIPT="${VLLM_SERVE_SCRIPT:-}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

mkdir -p "$RUNTIME_DIR"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "vllm already running (pid=$existing_pid). log=$LOG_FILE"
    exit 0
  fi
fi

if [[ -n "$VLLM_SERVE_SCRIPT" ]] && [[ ! -f "$VLLM_SERVE_SCRIPT" ]]; then
  echo "ERROR: VLLM_SERVE_SCRIPT does not exist: $VLLM_SERVE_SCRIPT"
  exit 1
fi

cmd=(python3 -m vllm.entrypoints.openai.api_server)
if [[ -n "$VLLM_SERVE_SCRIPT" ]]; then
  cmd=(python3 "$VLLM_SERVE_SCRIPT")
fi

cmd+=(
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
)

if [[ "$MODE" == "classify" ]]; then
  cmd+=(
    --convert classify
    --pooler-config '{"pooling_type":"LAST","use_activation":false}'
  )
fi

if [[ "$VLLM_TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
  # Intentionally split on shell words for flags like: "--dtype bfloat16 --gpu-memory-utilization 0.85"
  # shellcheck disable=SC2206
  extra_args=( $VLLM_EXTRA_ARGS )
  cmd+=("${extra_args[@]}")
fi

echo "Starting vLLM"
echo "  mode:  $MODE"
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
    echo "vLLM ready: http://localhost:${PORT} (pid=$pid)"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: vLLM process exited early. log=$LOG_FILE"
    exit 1
  fi
  sleep 1
done

echo "ERROR: vLLM did not become healthy in ${START_TIMEOUT_S}s. log=$LOG_FILE"
exit 1

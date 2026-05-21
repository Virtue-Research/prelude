#!/usr/bin/env bash
set -euo pipefail

# High-performance template: full context and default compile/warmup flow.
# Usage: bash scratch_runs/launch_vllm_perf.sh

CUDA_VISIBLE_DEVICES=4 \
PYTHONUNBUFFERED=1 \
HF_HOME=/scratch/xueying/.cache \
nohup taskset -c 0-47,96-143 numactl --cpunodebind=1 --membind=1 \
  /scratch/xueying/miniforge3/envs/vllm-clean/bin/vllm serve \
  Virtue-AI-HUB/topicguard-qwen3-15b-bf16-272k-compact-nouserprefix \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.9 \
  --served-model-name topicguard \
  > scratch_runs/vllm_topicguard_perf.log 2>&1 &

echo "Launched PERF template in background."
echo "Log: scratch_runs/vllm_topicguard_perf.log"

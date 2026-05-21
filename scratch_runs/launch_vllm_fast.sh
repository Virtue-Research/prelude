#!/usr/bin/env bash
set -euo pipefail

# Fast startup template: lower max context and eager mode.
# Usage: bash scratch_runs/launch_vllm_fast.sh

CUDA_VISIBLE_DEVICES=4 \
PYTHONUNBUFFERED=1 \
HF_HOME=/scratch/xueying/.cache \
nohup taskset -c 0-47,96-143 numactl --cpunodebind=1 --membind=1 \
  /scratch/xueying/miniforge3/envs/vllm-clean/bin/vllm serve \
  Virtue-AI-HUB/topicguard-qwen3-15b-bf16-272k-compact-nouserprefix \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --served-model-name topicguard \
  --enforce-eager \
  > scratch_runs/vllm_topicguard_fast.log 2>&1 &

echo "Launched FAST template in background."
echo "Log: scratch_runs/vllm_topicguard_fast.log"

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
  # > scratch_runs/vllm_topicguard_gpu3.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 \
# PYTHONUNBUFFERED=1 \
# HF_HOME=/scratch/panmz/.cache \
# nohup taskset -c 0-47,96-143 numactl --cpunodebind=0 --membind=0 \
#   /scratch/xueying/miniforge3/envs/vllm/bin/vllm serve \
#   Virtue-AI-HUB/topicguard-qwen3-15b-bf16-272k-compact-nouserprefix \
#   --host 0.0.0.0 --port 8000 \
#   --dtype bfloat16 \
#   --max-model-len 262144 \
#   --gpu-memory-utilization 0.9 \
#   --served-model-name topicguard
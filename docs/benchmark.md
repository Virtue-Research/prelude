# Benchmark Guide

## Prerequisites

- Docker (for SGLang, vLLM baselines)
- CUDA GPU + NVIDIA Container Toolkit (`nvidia-ctk`)
- Python 3.12 + uv (for genai-bench client)

## Setup

### 1. Build Prelude

```bash
# GPU (recommended)
cargo build -p prelude-server --release --features cuda

# CPU only
cargo build -p prelude-server --release
```

### 2. Install genai-bench (benchmark client)

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install "genai-bench @ git+https://github.com/rucnyz/genai-bench.git"
```

### 3. Pull baseline Docker images

```bash
docker pull lmsysorg/sglang:latest
docker pull vllm/vllm-openai:latest
```
For B300:
```
docker pull lmsysorg/sglang:latest-cu130
docker tag lmsysorg/sglang:latest-cu130 lmsysorg/sglang:latest
```

### 4. Verify

```bash
source .venv/bin/activate
genai-bench --version
./target/release/prelude-server --help
```

## Run Benchmarks

All engines use the same `bench.sh` script. SGLang and vLLM run in Docker containers automatically.

```bash
source .venv/bin/activate

# Single engine
CUDA_VISIBLE_DEVICES=2 ./benchmark/bench.sh prelude --gpu
CUDA_VISIBLE_DEVICES=2 ./benchmark/bench.sh sglang --gpu
CUDA_VISIBLE_DEVICES=2 ./benchmark/bench.sh vllm --gpu

# All GPU engines at once
CUDA_VISIBLE_DEVICES=2 ./benchmark/bench.sh --gpu

# All CPU engines
./benchmark/bench.sh --cpu
```

### Prefill-Only (128 in, 1 out)

```bash
CUDA_VISIBLE_DEVICES=2 INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 \
  ./benchmark/bench.sh --gpu
```

### Decode (32 in, 32 out)

```bash
CUDA_VISIBLE_DEVICES=2 INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh --gpu
```

### Decode (128 in, 32 out)

```bash
CUDA_VISIBLE_DEVICES=2 INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 \
  ./benchmark/bench.sh --gpu
```

### Custom

```bash
MODEL=Qwen/Qwen3-0.6B INPUT_TOKENS=64 OUTPUT_TOKENS=64 \
  CONCURRENCY=1 MAX_REQUESTS=50 ./benchmark/bench.sh prelude --gpu
```

## Accuracy Tests

```bash
# GPU
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant gpu \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B

# CPU F32
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-f32 \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B --timeout 600

# CPU BF16
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-bf16 \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B
```

## Notes

- SGLang and vLLM run inside Docker — no pip install needed
- `CUDA_VISIBLE_DEVICES` is passed into the Docker container automatically
- HuggingFace model cache (`~/.cache/huggingface`) is mounted into containers
- CPU benchmarks are slow — use small traffic like `D(16,16)` and few requests
- Results are saved to `bench_results/<timestamp>/summary.csv`
- Always use an idle GPU — check `nvidia-smi` before running

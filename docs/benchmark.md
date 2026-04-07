# Benchmark Guide

## Prerequisites

- Docker (for SGLang, vLLM baselines)
- CUDA GPU + NVIDIA Container Toolkit (`nvidia-ctk`)
- Python 3.12 + uv (for genai-bench client only)

## Setup

### 1. Build Prelude

```bash
# GPU (recommended): FlashInfer + FA4 + DeepGEMM + CUTLASS + oneDNN
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm,cutlass-gemm
```

### 2. Install genai-bench (benchmark client)

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install "genai-bench @ git+https://github.com/rucnyz/genai-bench.git"
```

### 3. Pull baseline Docker images

```bash
docker pull lmsysorg/sglang:latest     # SGLang (GPU + CPU)
docker pull vllm/vllm-openai:latest    # vLLM (GPU + CPU)
```

### 4. (Optional) Native baselines

```bash
# vllm.rs — Rust inference engine
cd .. && git clone https://github.com/yuzhounie/vllm.rs && cd vllm.rs
cargo build --release --bin vllm-rs --bin runner --features "cuda,nccl,flashinfer,graph,cutlass"
cd ../prelude

# llama.cpp — GGUF CPU inference
mkdir -p ../llama.cpp/{bin,models} && cd ../llama.cpp
TAG=$(curl -sL https://api.github.com/repos/ggerganov/llama.cpp/releases/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
curl -sL "https://github.com/ggerganov/llama.cpp/releases/download/${TAG}/llama-${TAG}-bin-ubuntu-x64.tar.gz" \
  | tar xz --strip-components=1 -C bin
huggingface-cli download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf --local-dir models
cd ../prelude
```

### 5. Verify

```bash
source .venv/bin/activate
genai-bench --version
./target/release/prelude-server --help
docker run --rm vllm/vllm-openai:latest --help
docker run --rm lmsysorg/sglang:latest python3 -c "import sglang; print('sglang', sglang.__version__)"
```

## Run Benchmarks

All engines use the same `bench.sh` script. SGLang and vLLM run in Docker containers automatically.

```bash
source .venv/bin/activate

# Single engine
CUDA_VISIBLE_DEVICES=1 ./benchmark/bench.sh prelude --gpu
CUDA_VISIBLE_DEVICES=1 ./benchmark/bench.sh sglang --gpu
CUDA_VISIBLE_DEVICES=1 ./benchmark/bench.sh vllm --gpu

# All GPU engines at once
CUDA_VISIBLE_DEVICES=1 ./benchmark/bench.sh --gpu

# All CPU engines
./benchmark/bench.sh --cpu
```

### Prefill-Only (128 in, 1 out)

```bash
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=1 CUDA_VISIBLE_DEVICES=1 \
  ./benchmark/bench.sh --gpu

INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=100 CONCURRENCY=4 CUDA_VISIBLE_DEVICES=1 \
  ./benchmark/bench.sh --gpu
```

### Decode (32 in, 32 out)

```bash
INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 CUDA_VISIBLE_DEVICES=1 \
  ./benchmark/bench.sh --gpu
```

### Decode (128 in, 32 out)

```bash
INPUT_TOKENS=128 OUTPUT_TOKENS=32 MAX_REQUESTS=400 CONCURRENCY=4 CUDA_VISIBLE_DEVICES=1 \
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

# CPU BF16 (requires --features onednn)
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-bf16 \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B
```

## GEMM Microbenchmark

```bash
# All backends (dispatch vs CUTLASS vs cuBLAS) across all models
CUDA_VISIBLE_DEVICES=1 cargo run -p prelude-core --bin gpu_ops_bench --release \
    --features flashinfer-v4,cutlass-gemm,deepgemm,bench-cublas,onednn

# Filter by model
... -- 8B

# Correctness verification
... -- --verify 8B
```

## Notes

- SGLang and vLLM run inside Docker — no pip install needed, no dependency conflicts
- `CUDA_VISIBLE_DEVICES` is passed into the Docker container automatically
- HuggingFace model cache (`~/.cache/huggingface`) is mounted into containers
- CPU benchmarks are slow. Use small traffic like `D(16,16)` and few requests
- Results are saved to `bench_results/<timestamp>/summary.csv`

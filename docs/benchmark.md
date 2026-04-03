# Benchmark Guide

## Environment Setup (GPU Server)

### 1. Clone and build Prelude

```bash
git clone https://github.com/Virtue-Research/prelude
cd prelude

# GPU (recommended): FlashInfer + FA4 + DeepGEMM + oneDNN
cargo build -p prelude-server --release --features flashinfer-v4,onednn,deepgemm
```

### 2. Install Python dependencies

Python 3.12 is recommended

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate

uv pip install "genai-bench @ git+https://github.com/rucnyz/genai-bench.git" # contain fixing for prefill-only e2e test
uv pip install transformers torch requests numpy
```

### Accuracy Tests

- cpu-f32

```shell
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-f32 \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B --timeout 600

python tests/accuracy/test_ppl.py \
    --binary target/release/prelude-server --model Qwen/Qwen3-0.6B --device cpu --dtype f32 --max-chunks 100
```

- cpu-bf16

```shell
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant cpu-bf16 \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B

python tests/accuracy/test_ppl.py \
    --binary target/release/prelude-server --model Qwen/Qwen3-0.6B --device cpu --dtype bf16 --max-chunks 100
```

- gpu

```shell
# logit
.venv/bin/python tests/accuracy/run_accuracy_test.py --variant gpu \
    --server prelude --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B
# PPL
python tests/accuracy/test_ppl.py \
    --binary target/release/prelude-server --model Qwen/Qwen3-0.6B
```

#### Tokenizer Latency

```bash
cargo run -p prelude-core --bin tokenizer_bench --release --features hf_tokenizer -- --model Qwen/Qwen3-0.6B
```
### microbench

```shell
cd crates/prelude-flash-attn-v4
CUDA_VISIBLE_DEVICES=1 cargo bench --bench fa4_vs_fa3
```
### End2End Latency (Docker)

Each engine runs in its own Docker container. Benchmark client (`genai-bench`) runs on the host.

```bash
pip install "genai-bench @ git+https://github.com/rucnyz/genai-bench.git"
```

#### Prefill only (128 tokens)

```bash
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=1 ./benchmark/e2e/bench.sh prelude
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=4 ./benchmark/e2e/bench.sh prelude
```

#### Decode (32 in, 32 out)

```bash
INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=100 CONCURRENCY=1 ./benchmark/e2e/bench.sh prelude
```

#### Compare all engines

```bash
# All engines sequentially
./benchmark/e2e/bench.sh all

# Single engine
./benchmark/e2e/bench.sh prelude
./benchmark/e2e/bench.sh vllm
./benchmark/e2e/bench.sh sglang
GGUF_MODEL=/path/to/model.gguf ./benchmark/e2e/bench.sh llama.cpp

# Custom parameters
MODEL=Qwen/Qwen3-0.6B INPUT_TOKENS=64 OUTPUT_TOKENS=64 \
  CONCURRENCY=4 MAX_REQUESTS=200 ./benchmark/e2e/bench.sh all
```

#### Docker images used

| Engine | Image | Notes |
|--------|-------|-------|
| prelude | `prelude-e2e` (built from `Dockerfile.prelude`) | GPU + CPU |
| vllm | `vllm/vllm-openai:latest` | Official image |
| sglang | `lmsysorg/sglang:latest` | Official image |
| llama.cpp | `llama-cpp-e2e` (built from `Dockerfile.llama-cpp`) | Needs GGUF model |

#### Non-Docker benchmarks

The `benchmark/` directory also has direct serve scripts for host-based benchmarking:

```bash
# Start server manually
./benchmark/serve_prelude.sh   # Prelude
./benchmark/serve_vllm.sh      # vLLM
./benchmark/serve_sglang.sh    # SGLang

# Run benchmark.py against any OpenAI-compatible server
python benchmark/benchmark.py complete --url http://localhost:8000 --model Qwen/Qwen3-0.6B
```

## Notes

- CPU benchmarks are slow. Use small traffic like `D(16,16)` and few requests.
- vLLM and SGLang official images require GPU.
- llama.cpp requires a GGUF model file. Download with:
  `huggingface-cli download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf --local-dir models`
- Custom fused CUDA kernels are compiled to PTX at build time and loaded at runtime via cudarc.

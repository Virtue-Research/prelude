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
### End2End Latency

Calculate E2E latency using [genai-bench](https://github.com/sgl-project/genai-bench)

#### Prefill only (128 tokens)

```bash
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=1 ./benchmark/bench.sh prelude --cpu
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=4 ./benchmark/bench.sh prelude --cpu
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=1 ./benchmark/bench.sh prelude --gpu
INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=4 ./benchmark/bench.sh prelude --gpu
```

#### Decode only (32 in, 32 out)

```bash
INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=100 CONCURRENCY=1 ./benchmark/bench.sh prelude --cpu
INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=100 CONCURRENCY=4 ./benchmark/bench.sh prelude --cpu
INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=100 CONCURRENCY=1 ./benchmark/bench.sh prelude --gpu
INPUT_TOKENS=32 OUTPUT_TOKENS=32 MAX_REQUESTS=100 CONCURRENCY=4 ./benchmark/bench.sh prelude --gpu
```

## Compare with Other Engines

Compare Prelude against vllm.rs, vLLM, and SGLang.

```shell
uv pip install vllm
uv pip install sglang --upgrade
uv pip install numpy==2.2
```

### 3a. (Optional) Build vllm.rs from source

Clone as a sibling directory next to `prelude` (the bench script expects `../vllm.rs` by default):

```shell
cd ..
git clone https://github.com/yuzhounie/vllm.rs
cd vllm.rs
cargo build --release --bin vllm-rs --bin runner --features "cuda,nccl,flashinfer,graph,cutlass"
```

### 3b. (Optional) Install vllm for CPU-only

Use a separate venv to avoid overwriting the GPU version (the bench script expects `../vllm-cpu/.venv` by default):

```shell
mkdir -p ../vllm-cpu && cd ../vllm-cpu
uv venv .venv --python 3.12
source .venv/bin/activate
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --torch-backend cpu
deactivate
cd ../prelude
```

### 3c. (Optional) Set up llama.cpp

Download prebuilt binary and BF16 GGUF model:

```shell
mkdir -p ../llama.cpp/{bin,models} && cd ../llama.cpp

# Download latest prebuilt binary
TAG=$(curl -sL https://api.github.com/repos/ggerganov/llama.cpp/releases/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
curl -sL "https://github.com/ggerganov/llama.cpp/releases/download/${TAG}/llama-${TAG}-bin-ubuntu-x64.tar.gz" \
  | tar xz --strip-components=1 -C bin

# Download BF16 GGUF model (1.2 GB)
huggingface-cli download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf --local-dir models

cd ../prelude
```

### 3d. (Optional) Build SGLang CPU Docker image

```shell
git clone https://github.com/sgl-project/sglang.git ../sglang-cpu
cd ../sglang-cpu/docker
docker build -t sglang-cpu:latest -f xeon.Dockerfile .
cd ../../prelude
```

### 4. Verify installation

```bash
# All should succeed
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import sglang; print('sglang', sglang.__version__)"
../vllm.rs/target/release/vllm-rs --help
../llama.cpp/bin/llama-server --help
genai-bench --version
./target/release/prelude-server --help
```

## Quick Start

```bash
# Benchmark all engines (Prelude CPU/GPU + vllm.rs + vLLM + SGLang)
./benchmark/bench.sh
# for prefill only
RESULTS_DIR=./prefill_bench INPUT_TOKENS=128 OUTPUT_TOKENS=1 MAX_REQUESTS=500 CONCURRENCY=4 ./benchmark/bench.sh
 
# Single engine
./benchmark/bench.sh prelude
./benchmark/bench.sh llama.cpp
./benchmark/bench.sh vllm.rs
./benchmark/bench.sh vllm
./benchmark/bench.sh vllm-cpu
./benchmark/bench.sh sglang
./benchmark/bench.sh sglang-cpu

# Custom paths (defaults: ../vllm.rs, ../vllm-cpu/.venv, ../llama.cpp)
VLLM_RS_DIR=/path/to/vllm.rs ./benchmark/bench.sh vllm.rs
VLLM_CPU_VENV=/path/to/venv ./benchmark/bench.sh vllm-cpu
LLAMA_CPP_DIR=/path/to/llama.cpp ./benchmark/bench.sh llama.cpp
GGUF_MODEL=/path/to/model.gguf ./benchmark/bench.sh llama.cpp

# Other Custom parameters
MODEL=Qwen/Qwen3-0.6B INPUT_TOKENS=64 OUTPUT_TOKENS=64 \
  CONCURRENCY=1 MAX_REQUESTS=50 ./benchmark/bench.sh
```

> You may need to manually install [nccl](https://developer.nvidia.com/nccl/nccl-download) to use vllm.rs

## Notes

- CPU benchmarks are slow. Use small traffic like `D(16,16)` and few requests.
- `--api-backend vllm` works for any OpenAI-compatible server.
- vllm.rs requires CUDA or Metal; vLLM and SGLang require GPU; Prelude supports both CPU and GPU.
- The bench script reads metrics from genai-bench's JSON output, not log parsing.
- NCCL is required by vllm.rs. If not installed system-wide, PyTorch bundles it — add the path to `LD_LIBRARY_PATH`.
- Custom fused CUDA kernels are compiled to PTX at build time and loaded at runtime via cudarc. No separate compilation
  step is needed.

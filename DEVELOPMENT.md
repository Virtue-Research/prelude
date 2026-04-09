# Development

## Docker (recommended for multi-machine)

One Dockerfile, works with or without GPU.

```bash
# Build image (once per machine, or after Dockerfile changes)
docker build -t prelude-dev .

# Interactive development
docker run --gpus all -it -v $(pwd):/workspace prelude-dev bash

# Run CPU benchmarks (no GPU needed)
docker run -v $(pwd):/workspace prelude-dev \
  cargo run -p prelude-cpu --bin cpu_ops_bench --release -- quant

# Run tests
docker run -v $(pwd):/workspace prelude-dev \
  cargo test -p prelude-core
```

`--gpus all` is ignored on machines without NVIDIA runtime — same image works everywhere.

The image includes: Rust stable, CUDA 12.8 toolkit, cmake, llama.cpp (for benchmark comparison), Claude Code.

## Local development (without Docker)

### Prerequisites

| Dependency    | Version | Required for       |
|---------------|---------|---------------------|
| Rust (stable) | 1.85+   | All builds          |
| CMake         | >= 3.18 | oneDNN (CPU builds) |
| CUDA Toolkit  | >= 12.x | GPU builds          |
| Python 3      | 3.10+   | Accuracy tests      |
| PyTorch       | 2.x     | Accuracy tests (HF reference) |

oneDNN is auto-downloaded and statically linked on first build.

### Python

```shell
uv venv .venv --python 3.12 --seed
source .venv/bin/activate

uv pip install transformers torch requests numpy cmake nvidia-cutlass datasets
```

### Build

```bash
# CPU only (default)
cargo build -p prelude-core --release

# Full stack (server + all GPU backends)
cargo build -p prelude-server --release --features full

# Check everything compiles
cargo check --workspace --all-targets
```

## Testing

### Running tests

```bash
# ── Core library (PyTorch reference) ────────────────────────
cargo test -p prelude-core

# ── CPU kernels (AVX-512, oneDNN, quantized) ─────────────────
cargo test -p prelude-cpu

# ── GPU kernel correctness ───────────────────────────────────
cargo test -p prelude-cutlass-gemm --release
cargo test -p prelude-deepgemm --release
cargo test -p prelude-flashinfer --release
cargo test -p prelude-flash-attn-v4 --release
cargo test -p prelude-quant-gemm --release
cargo test -p prelude-cula --release
```

### Accuracy (WikiText PPL)

Validates inference precision against HuggingFace transformers, same method as
[vLLM's generation_ppl_test](https://github.com/vllm-project/vllm/tree/main/tests/models/language/generation_ppl_test).
Pass criteria: server PPL within 1% of HF (one-sided — only fails if server is worse).

```bash
# Full test (computes HF reference PPL, takes a few minutes)
CUDA_VISIBLE_DEVICES=2 python3 tests/accuracy/test_ppl.py \
    --binary target/release/prelude-server \
    --model Qwen/Qwen3-8B

# Skip HF computation with known reference PPL
CUDA_VISIBLE_DEVICES=2 python3 tests/accuracy/test_ppl.py \
    --binary target/release/prelude-server \
    --model Qwen/Qwen3-0.6B --hf-ppl 23.864

# Connect to a running server
python3 tests/accuracy/test_ppl.py \
    --server http://localhost:8000 \
    --model Qwen/Qwen3-0.6B --hf-ppl 23.864
```

Known HF reference PPL (BF16, WikiText-2 test, stride=1024):
| Model | HF PPL |
|-------|--------|
| Qwen/Qwen3-0.6B | 23.864 |

### Benchmark

```bash
# E2E server benchmark (decode throughput)
CUDA_VISIBLE_DEVICES=2 MODEL=Qwen/Qwen3-8B INPUT_TOKENS=128 OUTPUT_TOKENS=32 \
    MAX_REQUESTS=400 CONCURRENCY=4 ./benchmark/bench.sh prelude --gpu

# CPU kernel benchmarks (quantized, GEMM, attention, etc.)
cargo run -p prelude-cpu --bin cpu_ops_bench --release
cargo run -p prelude-cpu --bin cpu_ops_bench --release -- quant
cargo run -p prelude-cpu --bin cpu_ops_bench --release -- gemm

# GPU kernel benchmarks
cargo run -p prelude-cutlass-gemm --example bench_kernel --release
cargo run -p prelude-deepgemm --example bench_kernel --release
cargo run -p prelude-flashinfer --example bench_kernel --release
cargo run -p prelude-flash-attn-v4 --example bench_kernel --release
cargo run -p prelude-quant-gemm --example bench_kernel --release

# Fused ops (silu_mul_concat vs fused_silu_mul, model shapes)
cargo run -p prelude-cuda --bin fused_ops_test --release
```

## Debugging

### Log levels

Server uses `tracing` (writes to stderr). Control via `RUST_LOG`:

```bash
# Default: info level for server + core
RUST_LOG=prelude_server=info,prelude_core=info

# Debug a specific module
RUST_LOG=prelude_cuda::ops::gemm=debug ./target/release/prelude-server ...

# Trace everything (very verbose)
RUST_LOG=debug ./target/release/prelude-server ...
```

### Runtime env vars

| Variable | Default | Effect |
|----------|---------|--------|
| `PRELUDE_SYNC_TIMING` | `false` | Enable GPU synchronization for per-step timing |
| `PRELUDE_CUDA_GRAPH_MAX_BS` | `32` | Max batch size for CUDA graph capture. Set to `0` to disable CUDA graphs. |
| `PRELUDE_FORCE_VARLEN_PREFILL` | `false` | Force varlen (non-paged) prefill path |
| `PRELUDE_FUSED_KV_CACHE_WRITE` | `0` | Enable fused KV cache write kernel |
| `PRELUDE_PAGED_ATTN_BLOCKS` | auto | Override number of KV cache blocks (0 = auto from GPU memory) |
| `PRELUDE_PAGED_BLOCK_SIZE` | auto | Override KV cache block size (default: 128 with FlashInfer) |
| `PRELUDE_PREFIX_CACHE_BLOCKS` | `0` | Number of blocks reserved for prefix cache |
| `PRELUDE_DEFAULT_TEMPERATURE` | `0.7` | Default sampling temperature |
| `PRELUDE_DEFAULT_MAX_TOKENS` | `4096` | Default max new tokens per request |

### GPU profiling

```bash
# NVTX markers (requires --features nvtx at build time)
cargo build -p prelude-server --release --features full,nvtx
nsys profile ./target/release/prelude-server --model Qwen/Qwen3-0.6B

# nsys on a running server (attach by PID)
nsys profile -p <PID> --trace=cuda --duration=10

# Disable CUDA graphs for individual kernel visibility
PRELUDE_CUDA_GRAPH_MAX_BS=0 ./target/release/prelude-server ...
```

## Feature flags

### prelude-server

| Feature | Effect |
|---------|--------|
| `cpu`   | CPU inference (default). Enables oneDNN BF16 GEMM + AVX-512 fused kernels. |
| `cuda`  | CUDA inference. All GPU kernel crates are compiled: FlashInfer, FA4, CUTLASS, DeepGEMM, quant-gemm, cuLA. |
| `full`  | `cpu` + `cuda`. Recommended for GPU machines. |

```bash
cargo build -p prelude-server --release --features full   # GPU
cargo build -p prelude-server --release                    # CPU only (default)
```

### prelude-core

| Feature | Effect |
|---------|--------|
| `cuda`  | Enable CUDA tensor backend (candle-core/cuda). Implied by server's `cuda`. |
| `hf_tokenizer` | HuggingFace tokenizer support via `tokenizers` crate. |
| `nvtx`  | NVIDIA NVTX profiling markers. Build with `--features nvtx` to enable. |
| `test-cuda` | Enable NVIDIA GPU in unit tests. |
| `test-amd`  | (planned) AMD GPU in tests. |
| `test-metal` | (planned) Apple GPU in tests. |
| `test-tpu`  | (planned) TPU in tests. |

### prelude-cpu

| Feature | Effect |
|---------|--------|
| `onednn` | oneDNN BF16/F32 GEMM (default). Auto-downloads and statically links oneDNN. |

GPU kernel crates (deepgemm, cutlass-gemm, flashinfer, fa4, quant-gemm, cula)
have no user-facing feature flags — they are always compiled when `cuda` is enabled.

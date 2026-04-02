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
  cargo run -p prelude-core --bin cpu_ops_bench --release -- quant

# Run tests
docker run -v $(pwd):/workspace prelude-dev \
  cargo test -p prelude-core --lib
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

oneDNN is auto-downloaded and statically linked on first build.
### Python

```shell
uv venv .venv --python 3.12 --seed
source .venv/bin/activate

uv pip install transformers torch requests numpy cmake
```


### Build

```bash
# CPU only (default)
cargo build -p prelude-core --release

# full stack
cargo build -p prelude-server --release --features full

# Check everything compiles
cargo check --workspace --all-targets
```

### Test

```bash
# CPU library tests
cargo test -p prelude-core --lib

# Specific module
cargo test -p prelude-core --lib -- quant
cargo test -p prelude-core --lib -- linear

# GPU kernel correctness tests
cargo test -p prelude-cutlass-gemm --release
cargo test -p prelude-deepgemm --release
cargo test -p prelude-flashinfer --release
cargo test -p prelude-flash-attn-v4 --release
cargo test -p prelude-quant-gemm --release
```

### Benchmark

```bash
# CPU kernel benchmarks (quantized, GEMM, attention, etc.)
cargo run -p prelude-core --bin cpu_ops_bench --release
cargo run -p prelude-core --bin cpu_ops_bench --release -- quant
cargo run -p prelude-core --bin cpu_ops_bench --release -- gemm

# GPU kernel benchmarks
cargo run -p prelude-cutlass-gemm --bin bench_kernel --release
cargo run -p prelude-deepgemm --bin bench_kernel --release
cargo run -p prelude-flashinfer --bin bench_kernel --release
cargo run -p prelude-flash-attn-v4 --bin bench_kernel --release

# Quant
cargo run -p prelude-quant-gemm --bin bench_kernel --release
```

## Feature flags

| Feature        | Effect                              |
|----------------|--------------------------------------|
| `cuda`         | GPU support (includes cutlass-gemm)  |
| `deepgemm`     | DeepGEMM SM90+ BF16 fast path       |
| `flash-attn-v4`| FlashAttention v4 (CuTeDSL)         |
| `flashinfer`   | FlashInfer attention backend         |
| `flashinfer-v4`| FlashInfer + FA4 combined            |
| `paged-attn`   | Paged KV cache attention             |
| `hf_tokenizer` | HuggingFace tokenizer support        |

CPU features (oneDNN, quantized kernels) are always compiled — no feature flag needed.

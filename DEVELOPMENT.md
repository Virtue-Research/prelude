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
| Python 3      | 3.10+   | PyTorch reference tests |
| PyTorch       | 2.x     | PyTorch reference tests |

oneDNN is auto-downloaded and statically linked on first build.

### Python

```shell
uv venv .venv --python 3.12 --seed
source .venv/bin/activate

uv pip install transformers torch requests numpy cmake nvidia-cutlass
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

### Architecture

Tests are organized in two layers:

1. **prelude-core** (`tests/tensor_ops.rs`, 97 tests): Correctness tests with **PyTorch reference**. Covers all TensorOps primitives (unary, binary, reduce, matmul, cast, etc.) and composed ops (rmsnorm, softmax, attention patterns). Tested across F32/BF16/F16 dtypes.

2. **prelude-cpu** (`src/ops/*/tests`, 169 tests): CPU kernel tests. Validates AVX-512/oneDNN optimized kernels against scalar reference implementations. Covers rmsnorm, attention, RoPE, SiLU×Mul, quantized GEMM.

prelude-cpu is a dev-dependency of prelude-core, so `cargo test -p prelude-core` automatically links prelude-cpu's high-level ops (AVX-512 rmsnorm, attention, etc.) via `#[ctor]` auto-registration. The PyTorch reference tests cover the full stack from model ops down to device kernels.

### Dual TensorOps backends

prelude-core has two parallel TensorOps implementations (temporary, for A/B validation):

- **CubeCL backend** (`cubecl_backend/`): CubeCL runtime, `Storage::CubeCL`. Default.
- **Device backend** (`device_backend/`): Pure Rust, `Storage::Device`. Reference implementation.

Controlled by `PRELUDE_TENSOR_BACKEND` env var:

```bash
# Test with CubeCL backend (default)
cargo test -p prelude-core

# Test with Device backend
PRELUDE_TENSOR_BACKEND=device cargo test -p prelude-core

# Test both backends (CI should run both)
PRELUDE_TENSOR_BACKEND=cubecl cargo test -p prelude-core
PRELUDE_TENSOR_BACKEND=device cargo test -p prelude-core
```

Both backends must pass the same 97 PyTorch-validated tests.

### Device selection

CPU tests always run. Device tests enabled via features:

```bash
# CPU only (default)
cargo test -p prelude-core

# CPU + NVIDIA GPU
cargo test -p prelude-core --features test-cuda

# CPU + AMD GPU (planned)
cargo test -p prelude-core --features test-amd

# CPU + Apple GPU (planned)
cargo test -p prelude-core --features test-metal

# CPU + TPU (planned)
cargo test -p prelude-core --features test-tpu

# All available devices
cargo test -p prelude-core --features test-cuda,test-amd,test-metal,test-tpu
```

### Running tests

```bash
# ── Core library (PyTorch reference, both backends) ──────────
cargo test -p prelude-core                                  # CubeCL backend, CPU
PRELUDE_TENSOR_BACKEND=device cargo test -p prelude-core    # Device backend, CPU

# ── CPU kernels (AVX-512, oneDNN, quantized) ─────────────────
cargo test -p prelude-cpu

# ── GPU kernel correctness ───────────────────────────────────
cargo test -p prelude-cutlass-gemm --release
cargo test -p prelude-deepgemm --release
cargo test -p prelude-flashinfer --release
cargo test -p prelude-flash-attn-v4 --release
cargo test -p prelude-quant-gemm --release
```

### Benchmark

```bash
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
| `test-cuda`    | Enable NVIDIA GPU in tests           |
| `test-amd`     | Enable AMD GPU in tests (planned)    |
| `test-metal`   | Enable Apple GPU in tests (planned)  |
| `test-tpu`     | Enable TPU in tests (planned)        |

CPU features (oneDNN, quantized kernels) are always compiled — no feature flag needed.

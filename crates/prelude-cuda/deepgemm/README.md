# prelude-deepgemm

DeepGEMM BF16 GEMM integration for Prelude. Based on [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).

Replaces cuBLAS for BF16 matmul — no cuBLAS dependency, statically linked.

## Performance (H200, SM90)

```
Shape                    cuBLAS(us) DeepGEMM(us)    Ratio
----------------------------------------------------------
M=1     N=4096  K=4096         13.9         11.5    0.83x  ← decode 17% faster
M=4     N=4096  K=4096         22.2         11.4    0.51x  ← decode 2x faster
M=32    N=4096  K=4096         13.9          9.0    0.65x  ← decode 35% faster
M=128   N=4096  K=4096         24.8         11.4    0.46x  ← 2x faster
M=256   N=4096  K=4096         18.9         18.1    0.96x
M=512   N=4096  K=4096         26.0         29.7    1.14x
M=1024  N=4096  K=4096         47.2         46.6    0.99x
M=2048  N=4096  K=4096         89.5         88.9    0.99x
```

## How It Works

```
Build time:
  nvcc compiles deepgemm_wrapper.cu (AOT kernel instantiations + heuristic + TMA)
  → nvcc --lib → libdeepgemm.a (static archive)
  CUTLASS headers auto-cloned from GitHub (sparse checkout, include/ only)

Runtime:
  select_config(M, N, K, num_sms)  → picks optimal (block_m, block_n, stages)
  make_2d_tma(ptr, dims, ...)      → creates CUtensorMap descriptors
  cudaLaunchKernel(kernel, ...)     → launches pre-compiled kernel variant
```

## Build

```bash
# As part of server (feature flag)
cargo build -p prelude-server --release --features deepgemm

# Standalone correctness test
CUDA_VISIBLE_DEVICES=1 cargo run --manifest-path crates/prelude-deepgemm/Cargo.toml \
    --example correctness_test --release

# GPU GEMM benchmark (vs candle/cuBLAS)
CUDA_VISIBLE_DEVICES=1 cargo bench -p prelude-core --bench gpu_ops_bench --features deepgemm
```

Requires: CUDA Toolkit 12.3+ with nvcc, SM90 (Hopper) GPU.

## Architecture

- **Kernel**: `vendor/deep_gemm/impls/sm90_bf16_gemm.cuh` — warp-specialized persistent kernel with TMA
- **Wrapper**: `src/deepgemm_wrapper.cu` — AOT kernel instantiations + C heuristic + TMA descriptor creation
- **FFI**: `src/lib.rs` — Rust interface: `bf16_gemm(A, B, D, M, N, K, stream)`
- **Heuristic**: Translated from DeepGEMM's `get_best_config` — selects optimal tile/stage/multicast per shape

## Pre-compiled Kernel Variants

Each variant is a specific template instantiation of `sm90_bf16_gemm_impl<...>`:

| block_m | block_n | stages | math_threads | multicast | use case |
|---------|---------|--------|-------------|-----------|----------|
| 16 | 32 | 32 | 128 | 1 | decode M=1~16 |
| 32 | 32 | 28 | 128 | 1 | decode M=17~32 |
| 64 | 64 | 13 | 128 | 1 | M=64~128 |
| 64 | 128 | 8 | 128 | 1 | M=128~256 |
| 64 | 256 | 4 | 128 | 2 | M=256~512, multicast |
| 128 | 256 | 3 | 256 | 2 | M=512+ prefill |
| 16/32/64/128 | 96/176 | varies | varies | varies | N=11008 shapes |

## Vendored Code

`vendor/deep_gemm/` contains kernel headers from DeepGEMM (MIT license).
Only `impls/` and `common/` are needed — no JIT, no Python, no torch dependency.
CUTLASS/CuTe headers are auto-cloned at build time.

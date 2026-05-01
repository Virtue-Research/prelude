# cutlass-gemm

CUTLASS 3.x BF16/F32/FP8 GEMM CUDA kernels for SM80+, compiled directly
from [NVIDIA cutlass](https://github.com/NVIDIA/cutlass) templates. No
cuBLAS dependency, no libtorch, statically linked.

## Scope

A minimal general-purpose GEMM kernel set that:

- Covers **BF16** (SM80+), **F32/TF32** (SM80+), and **FP8 E4M3** (SM90+)
- Supports **non-batched** and **strided-batched** layouts
- Runs on **SM80** (Ampere), **SM90** (Hopper), and **SM100** (Blackwell)
- Compiles as a **fat binary** containing all three arches, with runtime
  dispatch via a simple capability check

It exists as a cuBLAS fallback for cases where DeepGEMM doesn't handle
the shape (e.g. unusual M / batched layouts / FP8 paths not yet covered
by DeepGEMM).

| | |
|---|---|
| Upstream | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) (headers only) |
| Compile model | Direct `nvcc` compilation of `csrc/cutlass_wrapper.cu` + `csrc/naive_gemm.cu` at build time |
| Dispatch | Single `gemm_dispatch(...)` entry point — handles arch selection |
| Runtime deps | cudart_static, stdc++ — no cuBLAS, no libtorch |

## Features

```toml
[dependencies]
cutlass-gemm = "0.1"
```

No feature flags. All supported arches are always compiled.

## Build requirements

| Required | For | Default |
|---|---|---|
| CUDA toolkit (nvcc) with SM80/90/100 support | kernel compilation | `$CUDA_PATH` |
| `third_party/cutlass` headers | CUTLASS 3.x include path | `$CUTLASS_ROOT`, `$WORKSPACE/third_party/cutlass` |

### Environment variable overrides

| Variable | Default | Purpose |
|---|---|---|
| `CUTLASS_ROOT` | `$WORKSPACE/third_party/cutlass` | Point at any CUTLASS 3.x checkout |
| `CUDA_PATH` | `/opt/cuda` / `/usr/local/cuda` | CUDA toolkit root |

## Usage

```rust
use std::ffi::c_void;

unsafe {
    cutlass_gemm::gemm_dispatch(
        a_ptr, b_ptr, d_ptr,
        m, n, k, /*batch*/ 1,
        lda, ldb, ldd,
        /*stride_a*/ 0, /*stride_b*/ 0, /*stride_d*/ 0,
        /*transa*/ true, /*transb*/ false,
        /*dtype*/ 0,   // 0=BF16, 1=F16, 2=F32, 3=FP8_E4M3
        stream_ptr,
    )?;
}
```

The dispatcher picks the best tile configuration for the GPU's SM
capability automatically.

## Tests

```bash
cargo test -p cutlass-gemm --release
```

Correctness tests cover BF16, F32/TF32, FP8 E4M3, and batched layouts
against a CPU F64 reference.

## Benchmarks

```bash
cargo run -p cutlass-gemm --example bench_kernel --release
```

Sweeps various shapes against cuBLAS and cuBLASLt to establish when the
CUTLASS path is competitive.

## License

MIT. CUTLASS itself is BSD-3-Clause (NVIDIA).

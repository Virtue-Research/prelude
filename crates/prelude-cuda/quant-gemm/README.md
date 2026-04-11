# quant-gemm

GPU quantized matrix multiply for GGUF weight formats. Vendored from
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s CUDA kernels, with a
thin Rust FFI on top. Statically linked — no libtorch, no Python, no
llama.cpp runtime required.

## Scope

| | |
|---|---|
| Upstream | [llama.cpp CUDA kernels](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda) |
| Formats | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` |
| Kernels | `mul_mat_vec_q` (MMVQ — decode, M=1), tiled MMQ (prefill, M>1), `dequantize_block` (block → BF16), `quantize_q8_1` (BF16 → Q8_1) |
| Arch | SM75+ (Turing MMA), SM80+ (Ampere), SM90+ (Hopper tiles) |
| Runtime deps | cudart_static — nothing else |

## Why a separate crate

Because the GGUF quantization formats are a moving target — llama.cpp
adds / changes kernels frequently, and we want to be able to pull updates
without touching the rest of prelude. Isolating the FFI in its own crate
also lets us run `cargo test -p quant-gemm` to validate kernel
correctness against the CPU reference without dragging in the full
prelude workspace.

## Features

```toml
[dependencies]
quant-gemm = "0.1"
```

Single crate surface, no feature flags. Both MMVQ and tiled MMQ paths
are always compiled because the dispatcher picks between them at runtime
based on M.

## Build requirements

| Required | For | Default |
|---|---|---|
| CUDA toolkit (nvcc) | compiling `csrc/*.cu` wrappers + vendored llama.cpp kernels | `$CUDA_PATH`, `/usr/local/cuda` |
| `third_party/llama.cpp` source | kernel headers (`ggml/src/ggml-cuda/*.cuh`) | `$LLAMA_CPP_ROOT` or `$CARGO_WORKSPACE/third_party/llama.cpp` |

### Environment variable overrides

| Variable | Default | Purpose |
|---|---|---|
| `LLAMA_CPP_ROOT` | `$WORKSPACE/third_party/llama.cpp` | Point at any llama.cpp checkout |
| `CUDA_PATH` | `/opt/cuda` / `/usr/local/cuda` | CUDA toolkit root |

## Usage

```rust
use quant_gemm::GgmlType;
use std::ffi::c_void;

// Decode path (M=1): MMVQ
unsafe {
    quant_gemm::mul_mat_vec_q(
        weight_ptr,    // *const c_void — raw GGUF blocks on GPU
        activation_ptr,
        output_ptr,
        /*m*/ 1, n, k,
        GgmlType::Q4_K,
        stream_ptr,
    );
}

// Prefill path (M>1): tiled MMQ (quantize activations first)
unsafe {
    quant_gemm::quantize_q8_1(
        bf16_activations, q8_1_scratch,
        m, k, GgmlType::Q4_K, stream_ptr,
    );
    quant_gemm::mul_mat_q(
        weight_ptr, q8_1_scratch, output_ptr,
        m, n, k, GgmlType::Q4_K,
        /*compute_cap*/ 0, // auto-detect
        stream_ptr,
    );
}
```

## Tests

```bash
cargo test -p quant-gemm --release
```

The tests exercise every supported GGUF format against a llama.cpp CPU
reference (`dequantize_ref`), then run the full MMVQ + tiled MMQ path
against a CPU F64 matmul ground truth.

## Benchmarks

```bash
cargo run -p quant-gemm --example bench_kernel --release
```

Sweeps `(M, N, K)` and reports MMVQ / tiled MMQ / cuBLAS BF16 latencies
side-by-side.

## License

MIT, matching llama.cpp.

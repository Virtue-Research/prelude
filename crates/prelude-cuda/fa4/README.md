# prelude-flash-attn-v4

Flash Attention v4 (CuTeDSL AOT) integration for Prelude. Supports prefill, paged KV decode, multi-arch (SM90/SM100/SM120+).

## Current State

**Working:** Multi-arch kernel compilation, static linking, FFI call, paged KV (TMA + cp.async), correctness tests, microbench.

**Architecture:** Kernel variants are statically linked into the binary. No runtime `dlopen`, no external `.so` files — single binary distribution. Multiple SM architectures can coexist in one fat binary.

## How It Works

```
Build time (on GPU machine with Python + CuTeDSL):
  compile_kernels.py [-j 8]:
    Python (CuTeDSL) → JIT compile → export_to_c(.o) with unique function_name per variant
    Each .o exports __tvm_ffi_{variant_name} (no symbol conflicts)
    Supports incremental multi-arch: run per arch, manifest merges automatically
    Parallel compilation: -j N uses N subprocess workers

  build.rs:
    ar rcs libfa4_kernels.a kernels/*.o → static archive
    cc crate → compile vendored tvm_ffi C++ → libtvm_ffi_static.a
    Link: libfa4_kernels.a (+whole-archive) + libtvm_ffi_static.a + cuda_dialect_runtime
    Generate: fa4_dispatch.rs (extern "C" declarations + lookup table with arch dispatch)
    Change detection: SHA-256 hash of compile_kernels.py stored in manifest.json

Runtime:
  KernelRegistry::new()                    // auto-detects GPU SM arch via CUDA runtime
  let func = registry.get(&key)?;          // lookup by KernelKey (all compile dimensions)

  fa4_varlen_fwd(registry, func, q, k, v, ...)         // non-paged
  fa4_varlen_paged_fwd(registry, func, q, k, v, ...)   // paged KV
    → pack 17 args as TVMFFIAny array (DLTensorPtr + scalars)
    → func(NULL, args, 17, &result)        // direct call, no dlopen
```

## Build

```bash
# As part of server
cargo build -p prelude-server --release --features flash-attn-v4,flash-attn-v3

# Manual kernel compilation (parallel, recommended for first build)
PYTHONPATH=/path/to/flash-attention \
  python3 scripts/compile_kernels.py -j 8

# Then cargo build only links the pre-compiled kernels
cargo build -p prelude-server --release --features flash-attn-v4,flash-attn-v3
```

If `kernels/` has no `.o` files, build.rs auto-compiles (slower, uses min(cpus, 8) workers):
1. Creates Python venv (`uv venv` or `python3 -m venv`)
2. Installs nvidia-cutlass-dsl, quack-kernels, torch
3. Clones flash-attention repo
4. Runs `compile_kernels.py -j N` for the detected GPU arch

Change detection: build.rs compares SHA-256 of `compile_kernels.py` against `manifest.json`.
If the script changed, all `.o` files are cleared and recompiled automatically.

For multi-arch fat binary:
```bash
# Method 1: env var (build.rs compiles for all listed archs)
PRELUDE_FA4_ARCHS=sm_90,sm_120 cargo build --features flash-attn-v4

# Method 2: manual per-arch compilation (manifests merge automatically)
python scripts/compile_kernels.py --arch 90 -j 8 --output-dir kernels/
python scripts/compile_kernels.py --arch 120 -j 8 --output-dir kernels/
cargo build --features flash-attn-v4
```

## Test

```bash
# Correctness: FA4 output vs naive CPU attention reference
# Tests are #[ignore] by default (require GPU + compiled kernels)
cargo test --manifest-path crates/prelude-flash-attn-v4/Cargo.toml \
    --test correctness -- --ignored --nocapture

# Test cases: single/multi seq, GQA 2/4/8, noncausal, window, hdim 64/256, determinism
# Tolerance: atol=1e-2, rtol=1e-2 (BF16 output vs F32 CPU reference)
```

## Bench

```bash
# Kernel-level microbenchmark (CUDA events for accurate GPU timing)
cargo run -p prelude-flash-attn-v4 --bin bench_kernel --release

# Config: FA4_BENCH_WARMUP=5, FA4_BENCH_REPEATS=20
# Sweeps seq_len 128–8192, reports median/min/max latency + tokens/s + TFLOPS
```

## Kernel Variants

Mirrors upstream `flash_attn/cute/interface.py` compile_key dimensions.

### KernelKey (dispatch dimensions)

| Field | Values | Compile-time | Runtime |
|-------|--------|-------------|---------|
| head_dim | 64, 96, 128, 192, 256 | Baked in | KernelKey select |
| head_dim_v | Usually = head_dim; (192,128) for DeepSeek MLA | Baked in | KernelKey select |
| gqa_ratio | 1, 2, 4, 8, 16, 32 | Baked in | KernelKey select |
| causal | true, false | Baked in | KernelKey select |
| window | true, false | Baked in | KernelKey select |
| pack_gqa | Auto (gqa > 1) | Baked in | Auto |
| softcap | None, 30.0, 50.0 | Baked in | KernelKey select |
| paged | true, false | Baked in | KernelKey select |
| paged_non_tma | true (cp.async), false (TMA) | Baked in | Auto (block_size vs tile_n) |
| dtype | bf16, fp16 | Baked in | KernelKey select |
| has_seqused_q | true, false | Baked in | KernelKey select |

### Variant groups (per arch)

| Group | Count | Description |
|-------|-------|-------------|
| Non-paged base (bf16+fp16) | 240 | 5 hdim × 6 gqa × 2 causal × 2 window × 2 dtype |
| Softcap (bf16) | 80 | 5 hdim × 2 gqa(≤2) × 2 causal × 2 window × 2 softcap |
| Paged TMA (bf16+fp16) | 36 | 3 hdim(64,96,128) × 6 gqa × 2 dtype |
| Paged cp.async (bf16) | 18 | 3 hdim × 6 gqa |
| Paged + seqused_q (bf16) | 18 | 3 hdim × 6 gqa |
| DeepSeek MLA (192,128) bf16 | 12 | 6 gqa × 2 causal (SM100+ only) |
| **Total** | **~404** | Some fail on SM90 (expected) |

### Paged KV paths

| Path | Condition | Performance |
|------|-----------|-------------|
| **TMA** | block_size == tile_n (128 for hdim≤128) | Fast — hardware indirect loads |
| **cp.async** | block_size != tile_n | General — software row-by-row copy |

Runtime auto-selects based on actual block_size vs tile_n for the model's head_dim.

### SM90 known limitations

- **hdim192 paged**: tile_n=112 ≠ page_size=128, cp.async path has upstream shape bug
- **hdim256 paged**: tile_n=80, same issue
- **DeepSeek (192,128)**: SM100+ only

These variants fail at compile time and are skipped. Non-paged variants for all hdims work fine.

## Key FFI Details

- Each variant's `__tvm_ffi_{name}` is `TVMFFISafeCallType`:
  ```c
  int __tvm_ffi_{name}(void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result);
  ```
  First arg is self handle (pass NULL for extern C functions).

- 17 args (upstream 2025-07+, stream at end):
  `mQ, mK, mV, mO, mLSE, softmax_scale, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table, window_left, window_right, learnable_sink, blocksparse, aux, stream`

- Paged: cu_seqlens_k=None, seqused_k=per-seq lengths, page_table=block table, K/V are 4D `[num_pages, page_size, heads, hdim]`

- Tensor args use `kTVMFFIDLTensorPtr` (type_index=7). Stream: `OpaquePtr`. None: type_index=0.

## Future: Remove TVM FFI Dependency

Currently kernel .o files still reference TVM FFI symbols internally, so we need
`libtvm_ffi_static.a` at link time. Future optimization:

1. Strip `__tvm_ffi_*` symbols from .o files (the MLIR ciface entry doesn't need them)
2. Call `_mlir_{prefix}__mlir_ciface_{name}(void**, int32_t)` directly (plain C, no TVM)
3. Eliminates 6759 LOC C++ compilation, ~30s build time, ~500KB binary

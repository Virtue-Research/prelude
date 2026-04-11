# External Integration

Prelude integrates with external kernel libraries for GPU and CPU compute, and exposes an HTTP server for inference serving.

Deployment is a single **standalone** binary (`prelude-server`) — self-contained with its own scheduler, executor, and model runtime. No external orchestration layer is required.

```
Client (HTTP)
    │
prelude-server  (Axum router, auth, SSE)
    │
ScheduledEngine  (prelude-core)
    │
Scheduler + Executor + Model
```

## Third-Party Kernel Libraries

All kernel libraries follow the same integration pattern:

- **Vendored** as git submodules under `third_party/`
- **Compiled to static archives** (`.a`) by `build.rs` — no dynamic linking, no runtime `.so` dependencies
- **Bound via manual `extern "C"`** or TVM FFI — no bindgen
- **Dispatched with fallback**: each call site tries the faster library first, falls back on error or unsupported config

### DeepGEMM

| | |
|---|---|
| Source | `third_party/DeepGEMM` (deepseek-ai) |
| Crate | `crates/prelude-cuda/deepgemm/` |
| Target | SM90+ (Hopper+), BF16 only |
| Output | `libdeepgemm.a` (NVCC, fat binary SM90a + SM100a) |

`build.rs` copies the submodule to `OUT_DIR`, patches shared-memory buffer names to avoid symbol collisions in AOT compilation, then compiles with `-gencode=arch=compute_90a,code=sm_90a`.

FFI: manual `extern "C"` in `deepgemm/src/lib.rs`, wrapped in safe Rust functions returning `Result<(), String>`. Key entry points: `deepgemm_bf16_gemm()`, `deepgemm_fp8_gemm()`, MoE variants (`deepgemm_m_grouped_*`).

**Dispatch** (`prelude-cuda/src/ops/gemm.rs`): tries DeepGEMM first for non-batched BF16 when `m ≥ 16 && n ≥ 16`. Falls through to CUTLASS on any error. Not feature-gated — always compiled in.

### CUTLASS

| | |
|---|---|
| Source | `third_party/cutlass` |
| Crate | `crates/prelude-cuda/cutlass-gemm/` |
| Target | SM80+ (Ampere+), BF16 / FP16 / F32 |
| Output | `libcutlass_gemm.a` |

`build.rs` compiles two translation units: `cutlass_wrapper.cu` (CUTLASS 3.x template kernels) and `naive_gemm.cu` (isolated in a separate TU to avoid template instantiation bloat). SM80 and SM90a codegen targets.

FFI: manual `extern "C"`, `cutlass_gemm_dispatch()` returns error codes (-10 unsupported transpose, -20 unsupported dtype, -30 batched GEMM failed).

**Role**: universal fallback when DeepGEMM is skipped or fails. Not feature-gated.

### FlashInfer

| | |
|---|---|
| Source | `third_party/flashinfer` |
| Crate | `crates/prelude-cuda/flashinfer/` |
| Target | SM80+ (FA2) / SM90+ (FA3) |
| Output | `libflashinfer_kernels.a` |

`build.rs` invokes a Python script (`scripts/compile_kernels.py`) that runs FlashInfer's AOT compiler. Kernel variants are configurable via env vars:

| Env var | Default |
|---|---|
| `PRELUDE_FLASHINFER_ARCHS` | `sm_80,sm_90` |
| `PRELUDE_FLASHINFER_HEAD_DIMS` | `64,96,128,192,256,512` |
| `PRELUDE_FLASHINFER_DTYPES` | `bf16,fp16` |
| `PRELUDE_FLASHINFER_WORKERS` | (parallel compile threads) |

The build generates `fi_dispatch.rs` — a match table mapping `(PrefillKey / DecodeKey / MLADecodeKey / …) → extern fn pointer` via TVM FFI. `KernelRegistry` picks FA2 (SM80) or FA3 (SM90+) at runtime based on detected GPU arch.

FlashInfer's `plan()` runs once per forward pass and is cached across all transformer layers.

### FA4 (Flash Attention v4)

| | |
|---|---|
| Source | TVM FFI + CuTeDSL AOT kernels |
| Crate | `crates/prelude-cuda/fa4/` |
| Target | SM80+ |

Uses the same TVM FFI infrastructure as FlashInfer (shared via `tvm-static-ffi`). Calls build DLPack `DLTensor` argument arrays and dispatch via TVM's function registry.

Key entry points: `fa4_varlen_fwd()` (non-paged prefill), `fa4_varlen_paged_fwd()` (paged decode).

**Dispatch priority** (attention backends): FA4 → FlashInfer → CPU.

### oneDNN (CPU GEMM)

| | |
|---|---|
| Source | Downloaded by CMake (FetchContent) |
| Crate | `crates/prelude-cpu/` |
| Target | CPU — BF16 (AMX / AVX-512) + F32 |
| Output | `libdnnl.a` + `libonednn_ffi.a` |
| Feature flag | `onednn` (on by default) |

`build.rs` runs CMake to build both oneDNN core and a C++ wrapper (`onednn-ffi/`) that bridges oneDNN's threadpool interface to rayon. No git submodule — CMake fetches the source.

The C++ wrapper (`onednn_ffi.cpp`) maintains a primitive cache keyed by `(M, K, N, transpose_B, dtype)` to avoid JIT recompilation, and supports pre-packed weights (`onednn_bf16_pack_weights()` / `onednn_bf16_linear_packed()`) for amortized matmul cost.

**Fallback**: if the `onednn` feature is disabled, the `inventory`-based factory registration is skipped and `NaiveLinear` (pure Rust, no SIMD) is used instead. Runtime checks (`amx_bf16_available()`, `brgemm_available()`) guard AMX/BrGEMM paths.

### Summary

| Library | Linking | Feature flag | Dispatch role | Fallback |
|---|---|---|---|---|
| DeepGEMM | NVCC → `libdeepgemm.a` | None (always) | Primary GPU GEMM (SM90+, BF16) | CUTLASS |
| CUTLASS | NVCC → `libcutlass_gemm.a` | None (always) | GPU GEMM fallback (SM80+) | — |
| FlashInfer | Python+NVCC → `libflashinfer_kernels.a` | None (always) | Primary attention (SM80/90) | CPU attention |
| FA4 | TVM+NVCC, AOT | None (always) | Attention (SM80+, BF16) | FlashInfer |
| oneDNN | CMake → `libdnnl.a` | `onednn` | CPU GEMM (BF16/F32) | NaiveLinear |

# cula

Rust bindings to [cuLA](https://github.com/inclusionAI/cuLA) — Ant Group's
linear-attention CUDA kernel library (KDA fused forward, Lightning
Attention, etc.). Statically linked, **no libtorch / libpython runtime
dependency**.

## Scope

cuLA ships kernels in two forms, and this crate surfaces both:

| Path | What it is | Compilation | Exposed Rust API |
|---|---|---|---|
| **C++ CUTLASS 3.x** (`csrc/kda/sm90/*`, `csrc/kda/sm100/*`) | Hand-written warp-specialized TMA kernels | `nvcc` at crate build time | `kda_fwd_prefill_sm90`, `chunk_kda_fwd_intra_sm100`, `chunk_kda_fwd_recomp_wu_sm100` |
| **CuTeDSL AOT** (`cula/kda/*.py`, `cula/ops/*.py`, `cula/lightning/*.py`) | Python CuTeDSL kernels compiled to `.o` via `cute.compile(...).export_to_c(...)` | Python + `cutlass-dsl` at build time, dispatched via TVM SafeCall | `dsl::lookup_dsl(name, arch)` → `TVMSafeCallFn` |

Both paths end up as static libraries linked into your binary — there is
no runtime `dlopen`, no Python interpreter, no torch process.

### Current kernels

- **SM90** (Hopper)
  - `kda_fwd_prefill_sm90` — fused KDA forward prefill (MHA only).
    A multi-value GQA extension lives on the draft fork PR
    [`rucnyz/cuLA feat/sm90-gqa`](https://github.com/rucnyz/cuLA/tree/feat/sm90-gqa);
    this crate does not depend on it.
  - KDA decode, chunk_delta_h, fwd_o, lightning_prefill (CuTeDSL)
- **SM100 / SM103** (Blackwell)
  - `chunk_kda_fwd_intra_sm100`, `chunk_kda_fwd_recomp_wu_sm100`
  - Same CuTeDSL kernel set as SM90 where the kernel is arch-portable

## Features

```toml
[dependencies]
cula = { version = "0.1", features = ["dsl"] }
```

| Feature | Default | Adds |
|---|---|---|
| `dsl` | **on** | CuTeDSL AOT dispatch path. Pulls in `tvm-static-ffi`, runs Python at build time to produce `.o` files, compiles into `libcula_dsl_kernels.a`. |

Disable `dsl` (`default-features = false`) if you only need the C++
prefill / intra shims and want to skip the Python build step, saving a
few minutes on first build and ~50 KB of static kernel data.

## Build requirements

| Required | For | Default |
|---|---|---|
| CUDA toolkit (nvcc) | compiling `csrc/kda/{sm90,sm100}/*.cu` + `src/cula_wrapper.cu` | `/usr/local/cuda`, `$CUDA_PATH` |
| `third_party/cuLA` source | kernel sources | `$CARGO_WORKSPACE/third_party/cuLA`, `$CULA_ROOT` |
| `third_party/cutlass` headers | CUTLASS 3.x include path | `$CARGO_WORKSPACE/third_party/cutlass`, `$CUTLASS_ROOT` |
| Python 3.12 + `nvidia-cutlass-dsl` + `flash-linear-attention` | **only when `dsl` feature is on** | auto-installed into `$OUT_DIR/cula-venv` via `uv` |

### Environment variable overrides

| Variable | Default | Purpose |
|---|---|---|
| `CULA_ROOT` | `$WORKSPACE/third_party/cuLA` | Point at any cuLA checkout (e.g. a local fork being hacked on) |
| `CUTLASS_ROOT` | `$WORKSPACE/third_party/cutlass` | Point at any CUTLASS 3.x source tree |
| `CUDA_PATH` | `/opt/cuda` or `/usr/local/cuda` | CUDA toolkit root |
| `PRELUDE_CULA_WORKERS` | `1` | Parallel DSL compile workers (bumping past 1 is risky because of CUDA-after-fork issues inside the DSL runner) |

## Usage

### C++ shim path (always available)

```rust
use cula::kda_fwd_prefill_sm90;
use std::ffi::c_void;

unsafe {
    kda_fwd_prefill_sm90(
        stream_ptr,
        output_ptr, state_ptr,
        q_ptr, k_ptr, v_ptr,
        Some(input_state_ptr),
        Some(alpha_ptr),
        Some(beta_ptr),
        cu_seqlens_ptr, workspace_ptr,
        /*num_seqs*/ 1, /*num_heads*/ 32, /*head_size*/ 128,
        /*total_seqlen*/ 1024,
        /*scale*/ 1.0 / (128.0f32).sqrt(),
        /*safe_gate*/ true,
        /*sm_count*/ 132,
    )?;
}
```

### CuTeDSL path (requires `dsl` feature)

```rust
use cula::dsl::{lookup_dsl, TVMFFIAny};
use tvm_static_ffi::call_tvm_ffi;

let kernel = lookup_dsl(
    "kda_decode",
    "cula_kda_decode_small_dense_h32_hv32_v128_l2norm_sm90",
    /*arch*/ 90,
).expect("kernel not compiled for this arch");

let args = [/* build TVMFFIAny args for your DLTensors */];
unsafe { call_tvm_ffi(kernel, &args)? };
```

## Tests

```bash
cargo test -p cula --release
```

The test suite:

1. `correctness.rs` — CPU F64 reference vs GPU output for KDA prefill.
   Runs on any Hopper GPU.
2. Whatever ad-hoc smoke tests live under `tests/` for individual DSL
   kernels.

## License

Apache-2.0. cuLA upstream is Apache-2.0 (Ant Group Co., Ltd.).

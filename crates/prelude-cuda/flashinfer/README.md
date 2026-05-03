# flashinfer

Rust bindings to [FlashInfer](https://github.com/flashinfer-ai/flashinfer) —
the high-performance attention + sampling CUDA kernel library used by vLLM,
SGLang, and friends. Statically linked — no libtorch / libpython runtime.

## Scope

FlashInfer ships a mix of hand-written C++ CUTLASS kernels and JIT-compiled
templates. This crate AOT-compiles the kernel variants prelude needs into
`.o` files at crate build time and dispatches to them at runtime via the
TVM SafeCall convention. The upstream plan/run API (separate planning
step + cached workspace buffers) is preserved.

| | |
|---|---|
| Source | `third_party/flashinfer` tracks the Virtue fork, which carries Prelude AOT patches on top of upstream [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) |
| Kernel paths | Prefill (FA2 / FA3 backends), paged decode, MLA, sampling utilities, CUTLASS fused MoE |
| Compile model | Python AOT at build time → `.o` → static archive |
| Dispatch | `KernelRegistry` + `{Prefill,Decode}Key` → statically-linked `__tvm_ffi_<variant>` |
| Arch | SM80+ (FA2 default), SM90+ (FA3 fast path), SM100/SM103 for Blackwell fused MoE |
| Runtime deps | cudart_static, libbacktrace, stdc++ — no libtorch, no Python |

## Features

```toml
[dependencies]
flashinfer = "0.1"
```

All kernels (prefill, decode, sampling) are always compiled — there is no
feature matrix yet. Adding one is straightforward if a consumer wants a
leaner build.

## Build requirements

| Required | For | Default |
|---|---|---|
| CUDA toolkit (nvcc) | kernel compilation | `$CUDA_PATH`, `/usr/local/cuda` |
| `third_party/flashinfer` source | kernel templates | `$FLASHINFER_SRC` or `$CARGO_WORKSPACE/third_party/flashinfer` |
| Python 3.12 + nvidia-cutlass-dsl + torch | AOT compile driver | auto-installed via `uv` into `$OUT_DIR/flashinfer-venv` |

### Environment variable overrides

| Variable | Default | Purpose |
|---|---|---|
| `FLASHINFER_SRC` | `$WORKSPACE/third_party/flashinfer` | Point at any FlashInfer checkout |
| `CUDA_PATH` | `/opt/cuda` / `/usr/local/cuda` | CUDA toolkit root |
| `PRELUDE_FLASHINFER_ARCHS` | detected | Comma-separated SM targets for multi-arch builds |

## Usage

```rust
use flashinfer::{KernelRegistry, PrefillKey, KernelDtype, MaskMode};
use flashinfer::types::*;

// Planning (cheap, call once per model shape)
let reg = KernelRegistry::new()?;
let key = PrefillKey {
    dtype: KernelDtype::BF16,
    head_dim: 128,
    mask_mode: MaskMode::Causal,
    // ...
};
let plan = reg.plan_prefill(&key, /* workspace */)?;

// Run (hot path, per request)
unsafe { plan.run(q, k, v, o, cu_seqlens_q, cu_seqlens_k, /* ... */)? };
```

See [`src/lib.rs`](src/lib.rs) for the complete API surface and
[`tests/correctness.rs`](tests/correctness.rs) for end-to-end examples.

## Tests

```bash
cargo test -p flashinfer --release
```

## License

Apache-2.0 and MIT dual-license, matching upstream FlashInfer.

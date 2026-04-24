# Adding an Attention Backend

This guide walks through adding a new attention backend (e.g., FlashInfer, custom CUDA kernel) to Prelude.

## Overview

Adding a backend requires **3 changes**:

1. Create `attn/<name>.rs` in `crates/prelude-cuda/src/attn/` -- wrapper functions for your attention kernels
2. Implement the `AttentionOps` trait (defined in `prelude-core/src/ops/traits/attention.rs`)
3. Register your backend in `select_attention_backend()` in `crates/prelude-cuda/src/cuda_ops.rs`

No model code, no engine code, no scheduler code needs to change. Backend selection is centralized in `cuda_ops.rs`.

## Architecture

Attention backends live in the device crate, implementing the `AttentionOps` trait defined in prelude-core:

```
crates/prelude-core/src/ops/traits/
  attention.rs      -- trait AttentionOps, VarlenParams, PagedParams, MaskType

crates/prelude-cuda/src/attn/
  mod.rs            -- module exports
  flash_v4.rs       -- FA4 CuTeDSL wrappers
  flash_v3.rs       -- FA3 Hopper wrappers (varlen + paged)
  flash_v2.rs       -- FA2 Ampere+ wrappers
  flashinfer.rs     -- FlashInfer wrappers
  paged.rs          -- paged cache ops (reshape_and_cache, decode_attention)

crates/prelude-cuda/src/
  cuda_ops.rs       -- select_attention_backend() picks best backend at runtime
```

Model architectures call methods on `ops.attn` (an `AttentionOps` trait object). They never import a specific backend.

## The AttentionOps Trait

Your backend implements the `AttentionOps` trait, which has two required methods. The `MaskType` enum handles causal/bidirectional/windowed dispatch -- you don't need separate functions for each mask variant:

```rust
// Defined in prelude-core/src/ops/traits/attention.rs

pub enum MaskType {
    Causal,
    Bidirectional,
    SlidingWindow(usize),
}

pub struct VarlenParams {
    pub mask: MaskType,
    // cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, ...
}

pub struct PagedParams {
    pub mask: MaskType,
    // paged KV cache parameters, slot_mapping, block_table, ...
}

/// Two methods, not four -- mask variant is encoded in MaskType.
pub trait AttentionOps: Send + Sync {
    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor>;
    fn paged_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &PagedParams) -> Result<Tensor>;
}
```

## Step 1: Create the Backend Wrapper

Create `crates/prelude-cuda/src/attn/mybackend.rs`:

```rust
use prelude_core::tensor::{Result, Tensor};
use prelude_core::ops::traits::attention::{AttentionOps, VarlenParams, PagedParams, MaskType};

pub struct MyBackendOps {
    // hold any handles or state your backend needs
}

impl AttentionOps for MyBackendOps {
    fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
        match params.mask {
            MaskType::Causal => {
                // Call your causal attention kernel
                todo!()
            }
            MaskType::Bidirectional => {
                // Call your bidirectional attention kernel
                todo!()
            }
            MaskType::SlidingWindow(window_size) => {
                // Call your windowed attention kernel
                todo!()
            }
        }
    }

    fn paged_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &PagedParams) -> Result<Tensor> {
        todo!()
    }
}
```

## Step 2: Register in Backend Selection

In `crates/prelude-cuda/src/cuda_ops.rs`, add your backend to `select_attention_backend()`. Priority order is checked top-to-bottom:

```rust
fn select_attention_backend() -> Arc<dyn AttentionOps + 'static> {
    // FA4 (SM90+ only)
    #[cfg(feature = "flash-attn-v4")]
    if sm_version >= 90 { return Arc::new(FlashAttnV4Ops::new()); }

    // Your backend -- add here
    #[cfg(feature = "my-backend")]
    { return Arc::new(MyBackendOps::new()); }

    // FlashInfer fallback
    #[cfg(feature = "flashinfer")]
    { return Arc::new(FlashInferOps::new()); }

    // ... FA3, FA2 fallbacks ...
}
```

## Step 3: Add Feature Flag

In `crates/prelude-cuda/Cargo.toml` (NOT prelude-core -- feature flags for device backends live in the device crate):

```toml
[features]
my-backend = ["dep:my-backend-crate"]
```

## Testing

### Compare against existing backend

Run the accuracy test suite against an existing backend (e.g., FA2) and your new backend:

```bash
# Reference: FA2
cargo build -p prelude-server --release --features full
python tests/accuracy/run_accuracy_test.py --variant gpu-fa2 \
  --server prelude --binary target/release/prelude-server \
  --model Qwen/Qwen3-0.6B

# Your backend
cargo build -p prelude-server --release --features my-backend
python tests/accuracy/run_accuracy_test.py --variant gpu-std \
  --server prelude --binary target/release/prelude-server \
  --model Qwen/Qwen3-0.6B
```

Both should produce identical text output and logprob cosine similarity >= 0.99.

### Benchmark

```bash
python benchmark/local/benchmark.py --config benchmark/local/presets/complete_prefill.toml \
  --model Qwen/Qwen3-4B --url http://localhost:8000
```

## Existing Backends Reference

All CUDA backends are in `crates/prelude-cuda/src/attn/` and implement `AttentionOps`:

| Backend | File | Key Characteristics |
|---------|------|--------------------|
| FA4 | `flash_v4.rs` | CuTeDSL AOT, SM80+, prefill only, no paged KV, statically linked |
| FlashInfer | `flashinfer.rs` | SM80/SM90, varlen + paged, prefix cache capable |
| FA3 | `flash_v3.rs` | Hopper SM90, full varlen_paged support, prefix cache capable, GQA packing |
| FA2 | `flash_v2.rs` | Ampere+ SM80, GQA native, v1 cache layout, no varlen_paged (no prefix cache) |

CPU attention is in `crates/prelude-core/src/ops/cpu/attention/` (matmul SDPA + BF16 tiled attention with AVX-512).

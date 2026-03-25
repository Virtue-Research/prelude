# Adding an Attention Backend

This guide walks through adding a new attention backend (e.g., FlashInfer, custom CUDA kernel) to Prelude.

## Overview

Adding a backend requires **3 changes**:

1. Create `attn/<name>.rs` -- wrapper functions for your attention kernels
2. Add dispatch branches in `attn/mod.rs`
3. Add a feature flag in `Cargo.toml`

No model code, no engine code, no scheduler code needs to change. Attention dispatch is the **only** place with backend-specific `#[cfg]` gates.

## Architecture

All attention dispatch lives in one directory:

```
crates/prelude-core/src/models/common/attn/
  mod.rs       -- dispatch functions (the ONLY file with #[cfg] gates)
  flash_v4.rs  -- FA4 CuTeDSL wrappers
  flash_v3.rs  -- FA3 Hopper wrappers (varlen + paged)
  flash_v2.rs  -- FA2 Ampere+ wrappers
  paged.rs     -- paged cache ops (reshape_and_cache, decode_attention)
  cpu.rs       -- CPU matmul SDPA + BF16 tiled attention
```

Model architectures call dispatch functions from `mod.rs`. They never import a specific backend.

## Dispatch Functions

Your backend needs to implement some or all of these entry points:

| Function | Purpose | Required |
|----------|---------|----------|
| `varlen_attention()` | Causal variable-length attention + optional paged KV | Yes |
| `varlen_attention_bidirectional()` | Non-causal attention (for embeddings, classification) | Recommended |
| `varlen_attention_windowed()` | Sliding window attention (Gemma3) | Optional |
| `reshape_and_cache()` | Write K/V tensors to paged cache blocks | Yes (if paged) |
| `varlen_attention_paged()` | Read-only paged attention (for fused KV write paths) | Optional |

### Function Signatures

The key function is `varlen_attention`:

```rust
pub fn varlen_attention(
    q: &Tensor,                           // [total_tokens, num_heads, head_dim]
    k: &Tensor,                           // [total_tokens, num_kv_heads, head_dim]
    v: &Tensor,                           // [total_tokens, num_kv_heads, head_dim]
    cu_seqlens_q: &Tensor,                // [batch+1] cumulative sequence lengths
    cu_seqlens_k: &Tensor,                // [batch+1]
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    paged_kv: Option<&mut PagedKvBatchContext>,  // None = pure varlen, Some = write + read paged
) -> candle_core::Result<Tensor>
```

## Step 1: Create the Backend Wrapper

Create `attn/mybackend.rs`:

```rust
use candle_core::{Result, Tensor};

pub fn varlen_causal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    // Call your attention kernel here
    todo!()
}

pub fn varlen_bidirectional(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    todo!()
}
```

## Step 2: Add Dispatch Branches

In `attn/mod.rs`, add your backend to the dispatch chain. Priority order is checked top-to-bottom:

```rust
#[cfg(feature = "my-backend")]
mod mybackend;

pub fn varlen_attention(/* ... */) -> Result<Tensor> {
    // ... existing paged path ...

    // Non-paged path -- add your backend:
    #[cfg(feature = "flash-attn-v4")]
    { return flash_v4::varlen_causal(q, k, v, ...); }

    #[cfg(feature = "my-backend")]                    // <-- add here
    { return mybackend::varlen_causal(q, k, v, ...); }

    #[cfg(feature = "flash-attn-v3")]
    { return flash_v3::varlen_causal(q, k, v, ...); }

    // ... FA2, CPU fallback ...
}
```

## Step 3: Add Feature Flag

In `crates/prelude-core/Cargo.toml`:

```toml
[features]
my-backend = ["cuda", "dep:my-backend-crate"]
```

## Testing

### Compare against existing backend

Run the accuracy test suite against an existing backend (e.g., FA2) and your new backend:

```bash
# Reference: FA2
cargo build -p prelude-server --release --features flash-attn
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
python benchmark/benchmark.py --config benchmark/presets/complete_prefill.toml \
  --model Qwen/Qwen3-4B --url http://localhost:8000
```

## Existing Backends Reference

| Backend | File | Key Characteristics |
|---------|------|--------------------|
| FA4 | `flash_v4.rs` | CuTeDSL AOT, SM80+, prefill only, no paged KV, statically linked |
| FA3 | `flash_v3.rs` | Hopper SM90, full varlen_paged support, prefix cache capable, GQA packing |
| FA2 | `flash_v2.rs` | Ampere+ SM80, GQA native, v1 cache layout, no varlen_paged (no prefix cache) |
| CPU | `cpu.rs` | Matmul SDPA + BF16 tiled attention (AVX-512), runtime feature detection |

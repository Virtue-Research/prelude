//! FlashInfer AOT — statically linked attention kernels.
//!
//! Provides batch prefill (ragged + paged) and batch decode (paged) attention
//! via FlashInfer's plan-then-run API, compiled ahead-of-time and statically
//! linked. No runtime JIT or Python dependency.
//!
//! ## Architecture coverage
//! - FA2 backend: SM80+ (Ampere, Ada Lovelace)
//! - FA3 backend: SM90+ (Hopper)
//!
//! ## KV cache layout
//! Uses NHD layout: `[num_blocks, block_size, num_kv_heads, head_dim]` — same
//! as "flash layout" used by FA3/FA4.

pub mod loader;
pub mod types;

pub use loader::{
    Backend, DecodeKey, DecodeVariant, KernelDtype, KernelRegistry, MaskMode,
    PrefillKey, PrefillVariant, TVMSafeCallFn,
};

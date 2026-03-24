//! Forward execution paths — all task types (generate, classify, embed)
//! and paged attention operations (prefill, decode).

#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
mod classify;
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
mod embed;
mod generate;
mod paged_decode;
mod paged_prefill;
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
mod prefill_output;
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
mod prefill;
#[cfg(not(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer")))]
mod stubs;

// ── Classify re-exports ──
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
pub(crate) use self::classify::{RawClassifyOutput, classify_postprocess};
#[cfg(not(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer")))]
pub(crate) use self::stubs::{RawClassifyOutput, classify_postprocess};

// ── Embed re-exports ──
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
pub(crate) use self::embed::{RawEmbedOutput, embed_postprocess};
#[cfg(not(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer")))]
pub(crate) use self::stubs::{RawEmbedOutput, embed_postprocess};

// ── Generate re-exports (flash-attn-v3 only) ──
#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
pub(crate) use self::generate::{RawGenerateOutput, generate_postprocess};

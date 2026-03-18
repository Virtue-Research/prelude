//! Forward execution paths — all task types (generate, classify, embed)
//! and paged attention operations (prefill, decode).

#[cfg(feature = "flash-attn-v3")]
mod classify;
#[cfg(feature = "flash-attn-v3")]
mod embed;
mod generate;
mod paged_decode;
mod paged_prefill;
#[cfg(feature = "flash-attn-v3")]
mod prefill_output;
#[cfg(feature = "flash-attn-v3")]
mod prefill;
#[cfg(not(feature = "flash-attn-v3"))]
mod stubs;

// ── Classify re-exports ──
#[cfg(feature = "flash-attn-v3")]
pub(crate) use self::classify::{RawClassifyOutput, classify_postprocess};
#[cfg(not(feature = "flash-attn-v3"))]
pub(crate) use self::stubs::{RawClassifyOutput, classify_postprocess};

// ── Embed re-exports ──
#[cfg(feature = "flash-attn-v3")]
pub(crate) use self::embed::{RawEmbedOutput, embed_postprocess};
#[cfg(not(feature = "flash-attn-v3"))]
pub(crate) use self::stubs::{RawEmbedOutput, embed_postprocess};

// ── Generate re-exports (flash-attn-v3 only) ──
#[cfg(feature = "flash-attn-v3")]
pub(crate) use self::generate::{RawGenerateOutput, generate_postprocess};

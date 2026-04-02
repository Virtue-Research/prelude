//! Attention backends — GPU implementations of flash attention variants.

#[cfg(feature = "flash-attn-v4")]
pub(crate) mod flash_v4;
#[cfg(feature = "flashinfer")]
pub(crate) mod flashinfer;
#[cfg(feature = "flash-attn-v3")]
pub(crate) mod flash_v3;
#[cfg(feature = "flash-attn")]
pub(crate) mod flash_v2;
pub(crate) mod paged;

//! Attention backends — GPU implementations of flash attention variants.

#[cfg(feature = "flash-attn-v4")]
pub(crate) mod flash_v4;
#[cfg(feature = "flashinfer")]
pub(crate) mod flashinfer;

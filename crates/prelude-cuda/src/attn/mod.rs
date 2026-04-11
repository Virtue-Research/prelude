//! Attention backends — GPU implementations of flash attention variants.

pub(crate) mod flash_v4;
pub(crate) mod flashinfer;
pub(crate) mod kda_decode;
pub(crate) mod kda_prefill;

/// Determine FA4 tile_n (kBlockN) for paged KV cache TMA compatibility.
pub(crate) fn fa4_tile_n(head_dim: usize, head_dim_v: usize) -> usize {
    match head_dim {
        d if d <= 64 => {
            if head_dim_v == 512 { 64 }
            else if head_dim_v == 256 { 96 }
            else { 128 }
        }
        d if d <= 96 => 144,
        d if d <= 128 => 128,
        d if d <= 192 => {
            if head_dim_v <= 128 { 128 } else { 112 }
        }
        _ => 80,
    }
}

//! Attention backends — GPU implementations of flash attention variants.

#[cfg(feature = "flash-attn-v4")]
pub(crate) mod flash_v4;
#[cfg(feature = "flashinfer")]
pub(crate) mod flashinfer;

/// Determine FA4 tile_n (kBlockN) for paged KV cache TMA compatibility.
/// Mirrors tile_size_fwd_sm90() from flash-attention/hopper/tile_size.h.
/// For paged attention with TMA, page_size must equal tile_n.
#[cfg(feature = "flash-attn-v4")]
pub(crate) fn fa4_tile_n(head_dim: usize, head_dim_v: usize) -> usize {
    // Simplified: paged_kv_non_TMA=false, is_causal=true (most common for paged decode),
    // element_size=2 (bf16). Returns kBlockN.
    match head_dim {
        d if d <= 64 => {
            if head_dim_v == 512 { 64 }
            else if head_dim_v == 256 { 96 }
            else { 128 } // causal → use_blockN_128
        }
        d if d <= 96 => 144,
        d if d <= 128 => 128, // causal → use_blockN_128
        d if d <= 192 => {
            if head_dim_v <= 128 { 128 } else { 112 }
        }
        _ => 80,
    }
}

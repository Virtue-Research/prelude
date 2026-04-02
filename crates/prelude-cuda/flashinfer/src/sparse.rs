//! Sparse attention: block-sparse attention via BSR mask conversion.
//!
//! Follows the upstream FlashInfer sparse attention workflow:
//! 1. Define sparsity pattern as BSR (Block Sparse Row) format
//! 2. Convert BSR layout → FlashInfer flattened layout (CPU, matches `convert_bsr_mask_layout`)
//! 3. Pack bits on GPU via `segment_packbits` kernel (from `quantization.cuh`)
//! 4. Pass packed mask to any prefill/decode kernel with `MaskMode::Custom`
//!
//! ## Upstream reference
//!
//! `flashinfer/sparse.py` — `BlockSparseAttentionWrapper.plan()`:
//! ```python
//! mask = convert_bsr_mask_layout(mask, indptr)      # CPU layout conversion
//! packed_mask, mask_indptr = segment_packbits(       # GPU bit packing
//!     mask.contiguous().view(-1), mask_indptr, bitorder="little")
//! ```

use crate::loader::{KernelRegistry, TVMSafeCallFn};
use crate::types::TVMFFIAny;

/// GPU-accelerated bit packing for sparse attention masks.
///
/// Wraps FlashInfer's `packbits` and `segment_packbits` CUDA kernels
/// from `quantization.cuh`.
pub struct MaskPacker {
    packbits: TVMSafeCallFn,
    segment_packbits: TVMSafeCallFn,
}

impl MaskPacker {
    /// Create from kernel registry. Returns `None` if kernels not compiled.
    pub fn new(registry: &KernelRegistry) -> Option<Self> {
        Some(Self {
            packbits: registry.get_utility("packbits")?,
            segment_packbits: registry.get_utility("segment_packbits")?,
        })
    }

    /// Pack a boolean mask into uint8 on GPU.
    ///
    /// Upstream: `flashinfer.quantization.packbits(x, bitorder)`
    ///
    /// Args: `[x_bool_gpu, bitorder_str, y_packed_gpu]`
    ///
    /// # Safety
    /// Tensors must be on GPU.
    pub unsafe fn packbits(
        &self,
        registry: &KernelRegistry,
        args: &[TVMFFIAny],
    ) -> Result<(), String> {
        unsafe { registry.call(self.packbits, args)? };
        Ok(())
    }

    /// Segment-aware bit packing for batched masks on GPU.
    ///
    /// Upstream: `flashinfer.quantization.segment_packbits(x, input_indptr, bitorder)`
    ///
    /// Args: `[x_bool_gpu, input_indptr_gpu, output_indptr_gpu, bitorder_str, y_packed_gpu]`
    ///
    /// # Safety
    /// Tensors must be on GPU.
    pub unsafe fn segment_packbits(
        &self,
        registry: &KernelRegistry,
        args: &[TVMFFIAny],
    ) -> Result<(), String> {
        unsafe { registry.call(self.segment_packbits, args)? };
        Ok(())
    }
}

// ── CPU helpers for BSR layout conversion ────────────────────────────
// These match upstream's `convert_bsr_mask_layout` (Python CPU code).

/// Block Sparse Row (BSR) matrix representation.
pub struct BsrMask {
    pub num_block_rows: usize,
    pub num_block_cols: usize,
    pub block_size: usize,
    /// Row pointers: length = num_block_rows + 1.
    pub row_ptrs: Vec<i32>,
    /// Column indices of non-zero blocks.
    pub col_indices: Vec<i32>,
}

/// Convert a BSR sparsity pattern to a flattened boolean mask.
///
/// Matches upstream `convert_bsr_mask_layout`: produces a flat boolean array
/// (one byte per element, true = attend) ready for GPU `segment_packbits`.
///
/// # Returns
/// Flattened mask of length `num_qo_tokens * num_kv_tokens`.
pub fn bsr_to_flat_mask(
    bsr: &BsrMask,
    num_qo_tokens: usize,
    num_kv_tokens: usize,
) -> Vec<u8> {
    let total = num_qo_tokens * num_kv_tokens;
    let mut mask = vec![0u8; total];

    for block_row in 0..bsr.num_block_rows {
        let start = bsr.row_ptrs[block_row] as usize;
        let end = bsr.row_ptrs[block_row + 1] as usize;

        for &block_col_idx in &bsr.col_indices[start..end] {
            let block_col = block_col_idx as usize;
            for dr in 0..bsr.block_size {
                let row = block_row * bsr.block_size + dr;
                if row >= num_qo_tokens { break; }
                for dc in 0..bsr.block_size {
                    let col = block_col * bsr.block_size + dc;
                    if col >= num_kv_tokens { break; }
                    mask[row * num_kv_tokens + col] = 1;
                }
            }
        }
    }
    mask
}

/// Compute `mask_indptr` for batched segment_packbits.
///
/// # Arguments
/// * `qo_lens` - query length per sequence
/// * `kv_lens` - KV length per sequence
///
/// # Returns
/// `(input_indptr, output_indptr)` both of length `batch_size + 1`.
/// `input_indptr[i]` = cumulative boolean elements before sequence i.
/// `output_indptr[i]` = cumulative packed bytes before sequence i.
pub fn compute_mask_indptrs(qo_lens: &[usize], kv_lens: &[usize]) -> (Vec<i32>, Vec<i32>) {
    assert_eq!(qo_lens.len(), kv_lens.len());
    let n = qo_lens.len();
    let mut input_indptr = Vec::with_capacity(n + 1);
    let mut output_indptr = Vec::with_capacity(n + 1);
    input_indptr.push(0i32);
    output_indptr.push(0i32);
    for i in 0..n {
        let bits = (qo_lens[i] * kv_lens[i]) as i32;
        let bytes = (bits + 7) / 8;
        input_indptr.push(input_indptr[i] + bits);
        output_indptr.push(output_indptr[i] + bytes);
    }
    (input_indptr, output_indptr)
}

/// Create a causal BSR mask (lower triangular at block granularity).
pub fn causal_bsr_mask(seq_len: usize, block_size: usize) -> BsrMask {
    let num_blocks = (seq_len + block_size - 1) / block_size;
    let mut row_ptrs = Vec::with_capacity(num_blocks + 1);
    let mut col_indices = Vec::new();

    row_ptrs.push(0);
    for br in 0..num_blocks {
        let last_row = std::cmp::min((br + 1) * block_size, seq_len) - 1;
        for bc in 0..num_blocks {
            let first_col = bc * block_size;
            if first_col <= last_row {
                col_indices.push(bc as i32);
            }
        }
        row_ptrs.push(col_indices.len() as i32);
    }

    BsrMask {
        num_block_rows: num_blocks,
        num_block_cols: num_blocks,
        block_size,
        row_ptrs,
        col_indices,
    }
}

/// Create a sliding window BSR mask.
pub fn sliding_window_bsr_mask(
    qo_len: usize,
    kv_len: usize,
    window_size: usize,
    block_size: usize,
) -> BsrMask {
    let num_block_rows = (qo_len + block_size - 1) / block_size;
    let num_block_cols = (kv_len + block_size - 1) / block_size;
    let mut row_ptrs = Vec::with_capacity(num_block_rows + 1);
    let mut col_indices = Vec::new();

    let offset = kv_len.saturating_sub(qo_len);

    row_ptrs.push(0);
    for br in 0..num_block_rows {
        let last_row = std::cmp::min((br + 1) * block_size, qo_len) - 1;
        let first_row = br * block_size;
        let max_col = last_row + offset;
        let min_col = (first_row + offset).saturating_sub(window_size - 1);

        for bc in 0..num_block_cols {
            let block_first_col = bc * block_size;
            let block_last_col = std::cmp::min((bc + 1) * block_size, kv_len) - 1;
            if block_last_col >= min_col && block_first_col <= max_col {
                col_indices.push(bc as i32);
            }
        }
        row_ptrs.push(col_indices.len() as i32);
    }

    BsrMask {
        num_block_rows,
        num_block_cols,
        block_size,
        row_ptrs,
        col_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_bsr_mask_small() {
        let bsr = causal_bsr_mask(4, 2);
        assert_eq!(bsr.num_block_rows, 2);
        assert_eq!(bsr.row_ptrs, vec![0, 1, 3]);
        assert_eq!(bsr.col_indices, vec![0, 0, 1]);
    }

    #[test]
    fn test_bsr_to_flat_mask() {
        let bsr = BsrMask {
            num_block_rows: 2,
            num_block_cols: 2,
            block_size: 2,
            row_ptrs: vec![0, 1, 2],
            col_indices: vec![0, 1],
        };
        let mask = bsr_to_flat_mask(&bsr, 4, 4);
        // Block (0,0): rows 0-1, cols 0-1 = attend
        // Block (1,1): rows 2-3, cols 2-3 = attend
        assert_eq!(mask, vec![
            1,1,0,0,  // row 0
            1,1,0,0,  // row 1
            0,0,1,1,  // row 2
            0,0,1,1,  // row 3
        ]);
    }

    #[test]
    fn test_compute_mask_indptrs() {
        let (inp, out) = compute_mask_indptrs(&[4, 2], &[4, 8]);
        assert_eq!(inp, vec![0, 16, 32]);
        assert_eq!(out, vec![0, 2, 4]);
    }
}

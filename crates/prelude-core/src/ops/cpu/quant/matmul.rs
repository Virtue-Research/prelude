//! Quantized matrix multiplication: y = x @ W^T where W is Q4_0 quantized.
//!
//! Flow: quantize activations (FP32 → Q8_0) → row-parallel dot products → F32 output.
//! No candle dependency — operates on raw slices.

use super::q4_0::vec_dot_q4_0_q8_0;
use super::quantize::quantize_row_q8_0;
use super::types::*;

/// Quantized matrix multiplication: `x @ W^T`.
///
/// - `x`:   `[M * K]` row-major FP32 activations (M rows, K columns)
/// - `w`:   `[N * num_blocks]` Q4_0 weight blocks (N rows, each row = K/32 blocks)
/// - `out`: `[M * N]` row-major FP32 output (caller-allocated)
/// - `m`:   number of input rows (batch × seq_len)
/// - `n`:   number of output rows (out_features)
/// - `k`:   inner dimension (in_features), must be multiple of 32
///
/// Each output element: `out[i, j] = dot(x[i, :], W[j, :])`
pub fn quantized_matmul_f32(
    x: &[f32],
    w: &[BlockQ4_0],
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(k % QK8_0, 0, "k must be multiple of {QK8_0}");
    let nb = k / QK8_0; // blocks per row
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * nb);
    assert_eq!(out.len(), m * n);

    use rayon::prelude::*;

    // Row-parallel over M (input rows)
    out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
        // Quantize this input row once, reuse for all N dot products
        let x_row = &x[i * k..(i + 1) * k];
        let x_q8 = quantize_row_q8_0(x_row);

        for j in 0..n {
            let w_row = &w[j * nb..(j + 1) * nb];
            out_row[j] = vec_dot_q4_0_q8_0(w_row, &x_q8);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::q4_0::vec_dot_q4_0_q8_0_scalar;
    use super::super::quantize::quantize_row_q8_0_scalar;

    /// Scalar reference matmul: same algorithm as quantized_matmul_f32,
    /// but uses scalar (non-SIMD) dot product. This is the ground truth.
    fn ref_quantized_matmul(
        x: &[f32], w: &[BlockQ4_0], m: usize, n: usize, k: usize,
    ) -> Vec<f32> {
        let nb = k / QK8_0;
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let x_row = &x[i * k..(i + 1) * k];
            let x_q8 = quantize_row_q8_0_scalar(x_row);
            for j in 0..n {
                let w_row = &w[j * nb..(j + 1) * nb];
                out[i * n + j] = vec_dot_q4_0_q8_0_scalar(w_row, &x_q8);
            }
        }
        out
    }

    /// Create Q4_0 blocks from F32 values (simple quantization for testing).
    fn quantize_f32_to_q4_0(values: &[f32]) -> Vec<BlockQ4_0> {
        assert_eq!(values.len() % 32, 0);
        let nb = values.len() / 32;
        let mut blocks = Vec::with_capacity(nb);

        for i in 0..nb {
            let chunk = &values[i * 32..(i + 1) * 32];
            let amax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = amax / 7.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };

            let mut qs = [0u8; 16];
            for j in 0..16 {
                let lo = ((chunk[j] * id).round() as i32).clamp(-8, 7) + 8;
                let hi = ((chunk[j + 16] * id).round() as i32).clamp(-8, 7) + 8;
                qs[j] = (lo as u8) | ((hi as u8) << 4);
            }
            blocks.push(BlockQ4_0 { d: f32_to_fp16(d), qs });
        }
        blocks
    }

    /// Normalized dot product error (llama.cpp style): |result - ref| / n
    fn dot_product_error(result: &[f32], reference: &[f32]) -> f32 {
        assert_eq!(result.len(), reference.len());
        let n = result.len() as f32;
        result.iter().zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / n
    }

    #[test]
    fn matmul_matches_scalar_decode() {
        // M=1 (decode), N=64, K=128
        let k = 128;
        let n = 64;
        let m = 1;

        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let w_f32: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.007).cos() * 0.5).collect();
        let w = quantize_f32_to_q4_0(&w_f32);

        let ref_out = ref_quantized_matmul(&x, &w, m, n, k);
        let mut our_out = vec![0.0f32; m * n];
        quantized_matmul_f32(&x, &w, &mut our_out, m, n, k);

        let err = dot_product_error(&our_out, &ref_out);
        assert!(err < 1e-4, "decode: normalized error {err} too high");
    }

    #[test]
    fn matmul_matches_scalar_prefill() {
        // M=8, N=64, K=128
        let k = 128;
        let n = 64;
        let m = 8;

        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin() * 2.0).collect();
        let w_f32: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.009).cos()).collect();
        let w = quantize_f32_to_q4_0(&w_f32);

        let ref_out = ref_quantized_matmul(&x, &w, m, n, k);
        let mut our_out = vec![0.0f32; m * n];
        quantized_matmul_f32(&x, &w, &mut our_out, m, n, k);

        let err = dot_product_error(&our_out, &ref_out);
        assert!(err < 1e-4, "prefill: normalized error {err} too high");
    }

    #[test]
    fn matmul_matches_scalar_large_k() {
        // M=1, N=16, K=4096 — typical hidden_size
        let k = 4096;
        let n = 16;
        let m = 1;

        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.003).sin()).collect();
        let w_f32: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.002).cos() * 0.3).collect();
        let w = quantize_f32_to_q4_0(&w_f32);

        let ref_out = ref_quantized_matmul(&x, &w, m, n, k);
        let mut our_out = vec![0.0f32; m * n];
        quantized_matmul_f32(&x, &w, &mut our_out, m, n, k);

        let err = dot_product_error(&our_out, &ref_out);
        assert!(err < 1e-4, "large K: normalized error {err} too high");
    }

    #[test]
    fn all_zeros() {
        let x = vec![0.0f32; 2 * 32];
        let w = vec![BlockQ4_0 { d: 0, qs: [0; 16] }; 1];
        let mut out = vec![99.0f32; 2];

        quantized_matmul_f32(&x, &w, &mut out, 2, 1, 32);
        assert_eq!(out, [0.0, 0.0]);
    }
}

//! FFI bindings to ggml quantization kernels for baseline comparison.
//!
//! Links against pre-built ggml-cpu library from llama.cpp.
//! In Docker: GGML_LIB points to llama.cpp build directory.

// From ggml/src/ggml-cpu/quants.h:
//   void ggml_vec_dot_q4_0_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
//   void ggml_vec_dot_q4_K_q8_K(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
//
// From ggml/src/ggml-quants.h:
//   void quantize_row_q8_0_ref(const float *x, block_q8_0 *y, int64_t k);
//   void quantize_row_q8_K_ref(const float *x, block_q8_K *y, int64_t k);

unsafe extern "C" {
    fn ggml_vec_dot_q4_0_q8_0(
        n: i32, s: *mut f32, bs: usize,
        vx: *const u8, bx: usize,
        vy: *const u8, by: usize,
        nrc: i32,
    );
    fn ggml_vec_dot_q4_K_q8_K(
        n: i32, s: *mut f32, bs: usize,
        vx: *const u8, bx: usize,
        vy: *const u8, by: usize,
        nrc: i32,
    );
    fn quantize_row_q8_0_ref(x: *const f32, y: *mut u8, k: i64);
    fn quantize_row_q8_K_ref(x: *const f32, y: *mut u8, k: i64);
}

pub fn dot_q4_0_q8_0(x: &[u8], y: &[u8], n: usize) -> f32 {
    let mut result: f32 = 0.0;
    unsafe {
        ggml_vec_dot_q4_0_q8_0(
            n as i32, &mut result, 0,
            x.as_ptr(), 0,
            y.as_ptr(), 0,
            1,
        );
    }
    result
}

pub fn dot_q4_k_q8_k(x: &[u8], y: &[u8], n: usize) -> f32 {
    let mut result: f32 = 0.0;
    unsafe {
        ggml_vec_dot_q4_K_q8_K(
            n as i32, &mut result, 0,
            x.as_ptr(), 0,
            y.as_ptr(), 0,
            1,
        );
    }
    result
}

pub fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    let n_blocks = x.len() / 32;
    let mut out = vec![0u8; n_blocks * 34];
    unsafe {
        quantize_row_q8_0_ref(x.as_ptr(), out.as_mut_ptr(), x.len() as i64);
    }
    out
}

pub fn quantize_q8_k(x: &[f32]) -> Vec<u8> {
    let n_blocks = x.len() / 256;
    let mut out = vec![0u8; n_blocks * 292];
    unsafe {
        quantize_row_q8_K_ref(x.as_ptr(), out.as_mut_ptr(), x.len() as i64);
    }
    out
}

/// Q4_0 matmul using ggml's dot product kernel + rayon (same scheduling as ours).
/// Compares kernel speed, not scheduling strategy.
pub fn matmul_q4_0(x_data: &[f32], w_raw: &[u8], out: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    let block_size = 18; // Q4_0: 2 bytes scale + 16 bytes qs per 32 elements
    let blocks_per_row = k / 32;
    let row_bytes = blocks_per_row * block_size;

    out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
        let x_row = &x_data[i * k..(i + 1) * k];
        let x_q8 = quantize_q8_0(x_row);
        for j in 0..n {
            let w_row = &w_raw[j * row_bytes..(j + 1) * row_bytes];
            out_row[j] = dot_q4_0_q8_0(w_row, &x_q8, k);
        }
    });
}

/// Q4_K matmul using ggml's dot product kernel + rayon.
pub fn matmul_q4_k(x_data: &[f32], w_raw: &[u8], out: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    let block_size = 144; // Q4_K: 144 bytes per 256 elements
    let blocks_per_row = k / 256;
    let row_bytes = blocks_per_row * block_size;

    out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
        let x_row = &x_data[i * k..(i + 1) * k];
        let x_q8k = quantize_q8_k(x_row);
        for j in 0..n {
            let w_row = &w_raw[j * row_bytes..(j + 1) * row_bytes];
            out_row[j] = dot_q4_k_q8_k(w_row, &x_q8k, k);
        }
    });
}

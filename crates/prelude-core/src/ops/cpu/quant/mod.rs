//! Quantized matmul kernels — native SIMD dot products on GGUF data.
//!
//! Instead of dequantizing to F32 and doing standard matmul (6.7× memory bloat),
//! these kernels compute dot products directly on quantized blocks using integer SIMD.
//!
//! # Supported formats
//!
//! | Weight | Activation | bpw   | Kernel |
//! |--------|-----------|-------|--------|
//! | Q4_0   | Q8_0      | 4.5   | `q4_0::vec_dot_q4_0_q8_0` |
//! | Q4_1   | Q8_1      | 5.0   | `q4_1::vec_dot_q4_1_q8_1` |
//! | Q5_0   | Q8_0      | 5.5   | `q5_0::vec_dot_q5_0_q8_0` |
//! | Q5_1   | Q8_1      | 6.0   | `q5_1::vec_dot_q5_1_q8_1` |
//! | Q2_K   | Q8_K      | 2.625 | `q2_k::vec_dot_q2k_q8k` |
//! | Q3_K   | Q8_K      | 3.438 | `q3_k::vec_dot_q3k_q8k` |
//! | Q4_K   | Q8_K      | 4.5   | `q4_k::vec_dot_q4k_q8k` |
//! | Q5_K   | Q8_K      | 5.5   | `q5_k::vec_dot_q5k_q8k` |
//! | Q6_K   | Q8_K      | 6.563 | `q6_k::vec_dot_q6k_q8k` |
//!
//! # Architecture
//!
//! Each format has scalar + AVX2 implementations. The public `vec_dot_*` function
//! auto-dispatches to the fastest available at runtime via `is_x86_feature_detected!`.

pub mod types;
pub mod q4_0;
pub mod q4_1;
pub mod q5_0;
pub mod q5_1;
pub mod q2_k;
pub mod q3_k;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod quantize;
pub mod matmul;

pub use types::{
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1,
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K,
};
pub use q4_0::vec_dot_q4_0_q8_0;
pub use q4_1::vec_dot_q4_1_q8_1;
pub use q5_0::vec_dot_q5_0_q8_0;
pub use q5_1::vec_dot_q5_1_q8_1;
pub use q2_k::vec_dot_q2k_q8k;
pub use q3_k::vec_dot_q3k_q8k;
pub use q4_k::vec_dot_q4k_q8k;
pub use q5_k::vec_dot_q5k_q8k;
pub use q6_k::vec_dot_q6k_q8k;
pub use quantize::{quantize_row_q8_0, quantize_row_q8_1, quantize_row_q8k};
pub use matmul::quantized_matmul_f32;

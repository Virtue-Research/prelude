//! Quantized matmul kernels — native SIMD dot products on GGUF data.
//!
//! Instead of dequantizing to F32 and doing standard matmul (6.7× memory bloat),
//! these kernels compute dot products directly on quantized blocks using integer SIMD.
//!
//! # Supported formats
//!
//! | Weight | Activation | Kernel |
//! |--------|-----------|--------|
//! | Q4_0   | Q8_0      | `q4_0::vec_dot_q4_0_q8_0` |
//! | Q4_K   | Q8_K      | `q4_k::vec_dot_q4k_q8k` |
//!
//! # Architecture
//!
//! Each format has scalar + AVX2 implementations. The public `vec_dot_*` function
//! auto-dispatches to the fastest available at runtime via `is_x86_feature_detected!`.

pub mod types;
pub mod q4_0;
pub mod q4_k;
pub mod quantize;
pub mod matmul;

pub use types::{BlockQ4_0, BlockQ8_0, BlockQ4K, BlockQ8K};
pub use q4_0::vec_dot_q4_0_q8_0;
pub use q4_k::vec_dot_q4k_q8k;
pub use quantize::{quantize_row_q8_0, quantize_row_q8k};
pub use matmul::quantized_matmul_f32;

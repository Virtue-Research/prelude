//! GPU MMVQ: fused matrix-vector multiply with quantized weights.
//!
//! Computes `y[N] = W[N,K] @ x[K]` where `W` is GGUF-quantized and `x` is BF16.
//! Critical for decode (M=1): avoids materializing the full dequantized weight matrix,
//! saving ~3.6x memory bandwidth vs dequantize + GEMV.
//!
//! Internally quantizes the BF16 activation vector to Q8_1 on GPU, then launches
//! the fused dot-product kernel.

#![allow(non_snake_case)]

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Result, Tensor};

use super::{MOD_MMVQ, PTX_MMVQ};

const WARP_SIZE: u32 = 32;
const MMVQ_NWARPS: u32 = 4;
const Q8_1_BLOCK_SIZE: usize = 36; // sizeof(block_q8_1): d(2) + s(2) + qs(32)

/// Perform quantized matrix-vector multiplication on GPU.
///
/// - `quantized_weights`: 1-D `U8` tensor containing raw GGUF blocks for all N rows.
/// - `activations`: BF16 tensor of shape `[K]`.
/// - `n`: number of output rows.
/// - `k`: number of elements per row (must be multiple of `qk`).
/// - `kernel_name`: MMVQ kernel name (e.g., `"mmvq_q4_0"`).
/// - `qk`: elements per quantized block (32 for simple formats, 256 for K-quants).
///
/// Returns an F32 tensor of shape `[N]`.
pub fn mmvq(
    quantized_weights: &Tensor,
    activations: &Tensor,
    n: usize,
    k: usize,
    kernel_name: &str,
    qk: usize,
) -> Result<Tensor> {
    let (w_storage, w_layout) = quantized_weights.storage_and_layout();
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("mmvq: weights must be on CUDA"),
    };

    let (a_storage, a_layout) = activations.storage_and_layout();
    let a_cuda = match &*a_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("mmvq: activations must be on CUDA"),
    };

    if a_cuda.dtype() != DType::BF16 {
        candle_core::bail!("mmvq: activations must be BF16, got {:?}", a_cuda.dtype());
    }

    assert_eq!(k % qk, 0, "K ({k}) must be a multiple of QK ({qk})");
    assert_eq!(k % 32, 0, "K ({k}) must be a multiple of 32 for Q8_1");

    let dev = w_cuda.device().clone();

    // Step 1: Quantize BF16 activations → Q8_1 on GPU
    let num_q8_blocks = k / 32;
    let q8_buffer = unsafe { dev.alloc::<u8>(num_q8_blocks * Q8_1_BLOCK_SIZE) }?;

    {
        let func =
            dev.get_or_load_custom_func("quantize_bf16_q8_1", MOD_MMVQ, PTX_MMVQ)?;
        let grid = (
            (num_q8_blocks as u32 + MMVQ_NWARPS - 1) / MMVQ_NWARPS,
            1,
            1,
        );
        let block = (WARP_SIZE, MMVQ_NWARPS, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        let a_slice = a_cuda
            .as_cuda_slice::<half::bf16>()?
            .slice(a_layout.start_offset()..);
        let k_val = k as u32;

        let mut builder = func.builder();
        builder.arg(&a_slice);
        builder.arg(&q8_buffer);
        builder.arg(&k_val);
        unsafe { builder.launch(cfg) }.w()?;
    }

    // Step 2: Launch MMVQ kernel
    let blocks_per_row = (k / qk) as u32;
    let output = unsafe { dev.alloc::<f32>(n) }?;

    {
        let func = dev.get_or_load_custom_func(kernel_name, MOD_MMVQ, PTX_MMVQ)?;
        let grid = ((n as u32 + MMVQ_NWARPS - 1) / MMVQ_NWARPS, 1, 1);
        let block = (WARP_SIZE, MMVQ_NWARPS, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        let w_slice = w_cuda
            .as_cuda_slice::<u8>()?
            .slice(w_layout.start_offset()..);
        let n_val = n as u32;

        let mut builder = func.builder();
        builder.arg(&w_slice);
        builder.arg(&q8_buffer);
        builder.arg(&output);
        builder.arg(&n_val);
        builder.arg(&blocks_per_row);
        unsafe { builder.launch(cfg) }.w()?;
    }

    drop(w_storage);
    drop(a_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(output, dev);
    let out_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        candle_core::Shape::from(n),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(out_tensor)
}

// ── Per-format convenience wrappers ─────────────────────────────────────

/// MMVQ with Q4_0 weights (4.5 bpw, 32 elements/block).
pub fn mmvq_q4_0(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q4_0", 32)
}

/// MMVQ with Q4_1 weights (5.0 bpw, 32 elements/block).
pub fn mmvq_q4_1(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q4_1", 32)
}

/// MMVQ with Q5_0 weights (5.5 bpw, 32 elements/block).
pub fn mmvq_q5_0(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q5_0", 32)
}

/// MMVQ with Q5_1 weights (6.0 bpw, 32 elements/block).
pub fn mmvq_q5_1(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q5_1", 32)
}

/// MMVQ with Q8_0 weights (8.5 bpw, 32 elements/block).
pub fn mmvq_q8_0(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q8_0", 32)
}

/// MMVQ with Q2_K weights (2.625 bpw, 256 elements/block).
pub fn mmvq_q2_K(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q2_K", 256)
}

/// MMVQ with Q3_K weights (3.4375 bpw, 256 elements/block).
pub fn mmvq_q3_K(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q3_K", 256)
}

/// MMVQ with Q4_K weights (4.5 bpw, 256 elements/block).
pub fn mmvq_q4_K(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q4_K", 256)
}

/// MMVQ with Q5_K weights (5.5 bpw, 256 elements/block).
pub fn mmvq_q5_K(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q5_K", 256)
}

/// MMVQ with Q6_K weights (6.5625 bpw, 256 elements/block).
pub fn mmvq_q6_K(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_q6_K", 256)
}

// ── IQ (Importance-based Quantization) formats ─────────────────────────

/// MMVQ with IQ4_NL weights (4.5 bpw, non-linear 4-bit, 32 elements/block).
pub fn mmvq_iq4_nl(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_iq4_nl", 32)
}

/// MMVQ with IQ4_XS weights (4.25 bpw, non-linear 4-bit + sub-block scales, 256 elements/block).
pub fn mmvq_iq4_xs(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_iq4_xs", 256)
}

/// MMVQ with IQ3_XXS weights (3.0625 bpw, 256 elements/block).
pub fn mmvq_iq3_xxs(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_iq3_xxs", 256)
}

/// MMVQ with IQ3_S weights (3.4375 bpw, 256 elements/block).
pub fn mmvq_iq3_s(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_iq3_s", 256)
}

/// MMVQ with IQ2_XXS weights (2.0625 bpw, 256 elements/block).
pub fn mmvq_iq2_xxs(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_iq2_xxs", 256)
}

/// MMVQ with IQ2_XS weights (2.3125 bpw, 256 elements/block).
pub fn mmvq_iq2_xs(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_iq2_xs", 256)
}

/// MMVQ with IQ2_S weights (2.5625 bpw, 256 elements/block).
pub fn mmvq_iq2_s(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_iq2_s", 256)
}

/// MMVQ with IQ1_S weights (1.5625 bpw, 256 elements/block).
pub fn mmvq_iq1_s(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_iq1_s", 256)
}

/// MMVQ with IQ1_M weights (1.75 bpw, 256 elements/block).
pub fn mmvq_iq1_m(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_iq1_m", 256)
}

// ── FP4 formats ────────────────────────────────────────────────────────

/// MMVQ with MXFP4 weights (OCP MX spec, E8M0 + E2M1, 32 elements/block).
pub fn mmvq_mxfp4(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_mxfp4", 32)
}

/// MMVQ with NVFP4 weights (NVIDIA spec, UE4M3 + E2M1, 64 elements/block).
pub fn mmvq_nvfp4(w: &Tensor, x: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    mmvq(w, x, n, k, "mmvq_nvfp4", 64)
}

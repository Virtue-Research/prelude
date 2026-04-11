//! GPU tiled MMQ: high-performance quantized matrix multiply for prefill.
//!
//! Wraps `quant-gemm` (vendored from llama.cpp) to provide tiled
//! shared-memory MMQ with DP4A + tensor core support.
//!
//! Use this for M > 1 (prefill). For M = 1 (decode), use `mmvq` instead.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::DevicePtr;
use prelude_core::tensor::{bail, DType, Result, Tensor};
use std::ffi::c_void;

/// Perform tiled quantized matrix multiplication on GPU.
///
/// Y[M,N] = X[M,K] @ W[N,K]^T
///
/// - `quantized_weights`: 1-D `U8` tensor containing raw GGUF blocks for all N rows.
/// - `activations`: BF16 tensor of shape `[M, K]` (or flattened `[M*K]`).
/// - `m, n, k`: matrix dimensions.
/// - `ggml_dtype`: quantization format.
pub fn tiled_mmq(
    quantized_weights: &Tensor,
    activations: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    weight_type: quant_gemm::GgmlType,
) -> Result<Tensor> {
    let (w_storage, w_layout) = quantized_weights.storage_and_layout();
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("tiled_mmq: weights requires CUDA"),
    };

    let (a_storage, _a_layout) = activations.storage_and_layout();
    let a_cuda = match &*a_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("tiled_mmq: activations requires CUDA"),
    };

    if a_cuda.dtype() != DType::BF16 {
        bail!("tiled_mmq: activations must be BF16, got {:?}", a_cuda.dtype());
    }

    let dev = a_cuda.device().clone();
    let stream = dev.cuda_stream();

    // Get raw CUDA stream
    let raw_stream = unsafe { stream.cu_stream() } as *const c_void;

    // Allocate Q8_1_MMQ buffer
    // llama.cpp: nrows * GGML_PAD(K, 512) * 36/32 + mmq_x_max * sizeof(block_q8_1_mmq)
    let ne00_padded = ((k + 511) / 512) * 512;
    let mmq_x_max = 128; // SM75+ (Turing MMA); safe upper bound
    let q8_buf_size = m * ne00_padded * 36 / 32 + mmq_x_max * 144;
    let q8_buffer = unsafe { dev.alloc::<u8>(q8_buf_size) }?;

    // Get raw device pointers (use device stream for all)
    let w_slice = w_cuda.as_cuda_slice::<u8>()?.slice(w_layout.start_offset()..);
    let a_slice = a_cuda.as_cuda_slice::<half::bf16>()?;
    let output = unsafe { dev.alloc::<f32>(m * n) }?;

    let w_ptr = w_slice.device_ptr(&stream).0 as *const c_void;
    let a_ptr = a_slice.device_ptr(&stream).0 as *const c_void;
    let q8_ptr = q8_buffer.device_ptr(&stream).0 as *mut c_void;
    let out_ptr = output.device_ptr(&stream).0 as *mut f32;

    // Step 1: Quantize activations BF16 → Q8_1_MMQ
    unsafe {
        quant_gemm::quantize_q8_1(
            a_ptr, q8_ptr,
            m as i64, k as i64,
            weight_type,
            raw_stream,
        );
    }

    // Step 2: Tiled MMQ (compute_cap=0 → auto-detect in C++ side)
    unsafe {
        quant_gemm::mul_mat_q(
            w_ptr, q8_ptr as *const c_void, out_ptr,
            m as i64, n as i64, k as i64,
            weight_type,
            0, // auto-detect compute capability
            raw_stream,
        );
    }

    drop(w_storage);
    drop(a_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(output, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        (m, n),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

//! GPU tiled MMQ: high-performance quantized matrix multiply for prefill.
//!
//! Wraps `prelude-quant-gemm` (vendored from llama.cpp) to provide tiled
//! shared-memory MMQ with DP4A + tensor core support.
//!
//! Use this for M > 1 (prefill). For M = 1 (decode), use `mmvq` instead.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Result, Tensor};
use std::ffi::c_void;

/// Map candle's GgmlDType to prelude-quant-gemm's GgmlType.
fn to_ggml_type(dtype: candle_core::quantized::GgmlDType) -> Option<prelude_quant_gemm::GgmlType> {
    use candle_core::quantized::GgmlDType;
    use prelude_quant_gemm::GgmlType;
    match dtype {
        GgmlDType::Q4_0 => Some(GgmlType::Q4_0),
        GgmlDType::Q4_1 => Some(GgmlType::Q4_1),
        GgmlDType::Q5_0 => Some(GgmlType::Q5_0),
        GgmlDType::Q5_1 => Some(GgmlType::Q5_1),
        GgmlDType::Q8_0 => Some(GgmlType::Q8_0),
        GgmlDType::Q2K  => Some(GgmlType::Q2K),
        GgmlDType::Q3K  => Some(GgmlType::Q3K),
        GgmlDType::Q4K  => Some(GgmlType::Q4K),
        GgmlDType::Q5K  => Some(GgmlType::Q5K),
        GgmlDType::Q6K  => Some(GgmlType::Q6K),
        _ => None,
    }
}

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
    ggml_dtype: candle_core::quantized::GgmlDType,
) -> Result<Tensor> {
    let weight_type = to_ggml_type(ggml_dtype)
        .ok_or_else(|| candle_core::Error::Msg(format!("tiled_mmq: unsupported dtype {ggml_dtype:?}")))?;

    let dev = activations.device().as_cuda_device()?;

    let (w_storage, w_layout) = quantized_weights.storage_and_layout();
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("tiled_mmq: weights must be on CUDA"),
    };

    let (a_storage, _a_layout) = activations.storage_and_layout();
    let a_cuda = match &*a_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("tiled_mmq: activations must be on CUDA"),
    };

    if a_cuda.dtype() != DType::BF16 {
        candle_core::bail!("tiled_mmq: activations must be BF16, got {:?}", a_cuda.dtype());
    }

    // Get raw CUDA stream
    let stream = dev.cuda_stream();
    let raw_stream = unsafe { stream.cu_stream() } as *const c_void;

    // Allocate Q8_1_MMQ buffer
    // block_q8_1_mmq = 144 bytes, holds 128 values
    // llama.cpp pads K to 512, allocate generously
    let ne00_padded = ((k + 511) / 512) * 512;
    let q8_buf_size = m * (ne00_padded / 128) * 144 + m * 144;
    let q8_buffer = unsafe { dev.alloc::<u8>(q8_buf_size)? };

    // Get raw device pointers (use device stream for all)
    let w_slice = w_cuda.as_cuda_slice::<u8>()?.slice(w_layout.start_offset()..);
    let a_slice = a_cuda.as_cuda_slice::<half::bf16>()?;
    let output = unsafe { dev.alloc::<f32>(m * n)? };

    let w_ptr = w_slice.device_ptr(&stream).0 as *const c_void;
    let a_ptr = a_slice.device_ptr(&stream).0 as *const c_void;
    let q8_ptr = q8_buffer.device_ptr(&stream).0 as *mut c_void;
    let out_ptr = output.device_ptr(&stream).0 as *mut f32;

    // Step 1: Quantize activations BF16 → Q8_1_MMQ
    unsafe {
        prelude_quant_gemm::quantize_q8_1(
            a_ptr, q8_ptr,
            m as i64, k as i64,
            weight_type,
            raw_stream,
        );
    }

    // Step 2: Tiled MMQ (compute_cap=0 → auto-detect in C++ side)
    unsafe {
        prelude_quant_gemm::mul_mat_q(
            w_ptr, q8_ptr as *const c_void, out_ptr,
            m as i64, n as i64, k as i64,
            weight_type,
            0, // auto-detect compute capability
            raw_stream,
        );
    }

    drop(w_storage);
    drop(a_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(output, dev.clone());
    let out_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        candle_core::Shape::from((m, n)),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(out_tensor)
}

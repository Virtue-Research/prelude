//! GPU dequantization — thin wrapper over prelude-quant-gemm FFI.
//!
//! Converts quantized weight blocks to BF16 on the GPU.
//! All kernel implementations live in prelude-quant-gemm.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::DevicePtr;
use prelude_core::tensor::{bail, DType, Result, Tensor};
use std::ffi::c_void;

/// Dequantize a GGUF quantized tensor to BF16 on the GPU.
///
/// `quantized_data` is a 1-D `u8` tensor containing the raw GGUF blocks.
/// `num_elements` is the total number of values (not bytes).
/// `weight_type` selects the quantization format.
/// Returns a BF16 tensor of shape `[num_elements]`.
pub fn dequantize_to_bf16(
    quantized_data: &Tensor,
    num_elements: usize,
    weight_type: prelude_quant_gemm::GgmlType,
) -> Result<Tensor> {
    let (storage, layout) = quantized_data.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("dequantize_to_bf16: requires CUDA"),
    };

    if cuda.dtype() != DType::U8 {
        bail!("dequantize_to_bf16: input must be U8 (raw GGUF bytes)");
    }

    let dev = cuda.device().clone();
    let stream = dev.cuda_stream();
    let input_slice = cuda.as_cuda_slice::<u8>()?.slice(layout.start_offset()..);
    let output = unsafe { dev.alloc::<half::bf16>(num_elements) }?;

    let raw_stream = unsafe { stream.cu_stream() } as *const c_void;

    let in_ptr = input_slice.device_ptr(&stream).0 as *const c_void;
    let out_ptr = output.device_ptr(&stream).0 as *mut c_void;

    unsafe {
        prelude_quant_gemm::gpu_dequantize(
            in_ptr, out_ptr,
            num_elements as i64,
            weight_type,
            raw_stream,
        );
    }

    drop(storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(output, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        num_elements,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

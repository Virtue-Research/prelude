//! GPU dequantization — thin wrapper over prelude-quant-gemm FFI.
//!
//! Converts quantized weight blocks to BF16 on the GPU.
//! All kernel implementations live in prelude-quant-gemm.

use crate::device::{self as cb, CuResultExt, DevicePtr};
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
    let (storage, layout) = cb::storage_and_layout(&quantized_data);
    let device = cb::as_cuda(&storage, "dequantize_to_bf16")?;

    if device.dtype() != DType::U8 {
        bail!("dequantize_to_bf16: input must be U8 (raw GGUF bytes)");
    }

    let stream = device.stream.clone();
    let input_slice = device.as_slice::<u8>()?.slice(layout.start_offset()..);
    let output = unsafe { stream.alloc::<half::bf16>(num_elements) }.ce()?;

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

    Ok(cb::tensor_from_cuda(output, stream, num_elements))
}

//! GPU dequantization kernels for GGUF quantized formats.
//!
//! Converts quantized weight blocks to BF16 on the GPU for use with standard GEMM.
//! Supports: Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.

use candle_core::backend::BackendStorage;
use super::{MOD_DEQUANTIZE, PTX_DEQUANTIZE};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Result, Tensor};

/// Dequantize a GGUF quantized tensor to BF16 on the GPU.
///
/// `quantized_data` is a 1-D `u8` tensor containing the raw GGUF blocks.
/// `num_elements` is the total number of values (not bytes).
/// Returns a BF16 tensor of shape `[num_elements]`.
pub fn dequantize_to_bf16(
    quantized_data: &Tensor,
    num_elements: usize,
    kernel_name: &str,
) -> Result<Tensor> {
    let (storage, layout) = quantized_data.storage_and_layout();
    let cuda_storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("dequantize_to_bf16: requires CUDA tensor"),
    };

    if cuda_storage.dtype() != DType::U8 {
        candle_core::bail!("dequantize_to_bf16: input must be U8 (raw GGUF bytes)");
    }

    let dev = cuda_storage.device().clone();
    let input_slice = cuda_storage.as_cuda_slice::<u8>()?.slice(layout.start_offset()..);
    let out = unsafe { dev.alloc::<half::bf16>(num_elements) }?;

    let threads = 256u32;
    let blocks = ((num_elements as u32) + threads - 1) / threads;

    let func = dev.get_or_load_custom_func(kernel_name, MOD_DEQUANTIZE, PTX_DEQUANTIZE)?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&input_slice);
    builder.arg(&out);
    let n_val = num_elements as u64;
    builder.arg(&n_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    let out_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        candle_core::Shape::from(num_elements),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(out_tensor)
}

/// Dequantize Q4_0 blocks to BF16 on GPU.
pub fn dequantize_q4_0_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q4_0_bf16")
}

/// Dequantize Q4_1 blocks to BF16 on GPU.
pub fn dequantize_q4_1_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q4_1_bf16")
}

/// Dequantize Q5_0 blocks to BF16 on GPU.
pub fn dequantize_q5_0_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q5_0_bf16")
}

/// Dequantize Q5_1 blocks to BF16 on GPU.
pub fn dequantize_q5_1_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q5_1_bf16")
}

/// Dequantize Q8_0 blocks to BF16 on GPU.
pub fn dequantize_q8_0_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q8_0_bf16")
}

/// Dequantize Q2_K blocks to BF16 on GPU.
pub fn dequantize_q2k_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q2_K_bf16")
}

/// Dequantize Q3_K blocks to BF16 on GPU.
pub fn dequantize_q3k_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q3_K_bf16")
}

/// Dequantize Q4_K blocks to BF16 on GPU.
pub fn dequantize_q4k_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q4_K_bf16")
}

/// Dequantize Q5_K blocks to BF16 on GPU.
pub fn dequantize_q5k_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q5_K_bf16")
}

/// Dequantize Q6_K blocks to BF16 on GPU.
pub fn dequantize_q6k_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_q6_K_bf16")
}

/// Dequantize IQ4_NL blocks to BF16 on GPU.
pub fn dequantize_iq4_nl_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_iq4_nl_bf16")
}

/// Dequantize IQ4_XS blocks to BF16 on GPU.
pub fn dequantize_iq4_xs_bf16(quantized_data: &Tensor, num_elements: usize) -> Result<Tensor> {
    dequantize_to_bf16(quantized_data, num_elements, "dequantize_iq4_xs_bf16")
}

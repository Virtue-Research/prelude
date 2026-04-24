use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Result, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{MOD_ADD, MOD_SILU_MUL, PTX_ADD, PTX_SILU_MUL};

/// Fused SiLU(gate) * up. Both inputs must be contiguous BF16 CUDA tensors.
pub fn fused_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (up_storage, up_layout) = up.storage_and_layout();

    let gate_cuda = match &*gate_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_silu_mul: requires CUDA"),
    };
    let up_cuda = match &*up_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_silu_mul: requires CUDA"),
    };

    let n = gate_layout.shape().elem_count();
    if n != up_layout.shape().elem_count() {
        candle_core::bail!("fused_silu_mul: shape mismatch");
    }
    if gate_cuda.dtype() != DType::BF16 || up_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fused_silu_mul: requires BF16");
    }

    let dev = gate_cuda.device().clone();

    let gate_slice = gate_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(gate_layout.start_offset()..);
    let up_slice = up_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(up_layout.start_offset()..);

    let out = unsafe { dev.alloc::<half::bf16>(n) }?;

    let threads = 256u32;
    let elems_per_thread = 8u32;
    let blocks = (n as u32 + threads * elems_per_thread - 1) / (threads * elems_per_thread);

    let func = dev.get_or_load_custom_func("fused_silu_mul_bf16", MOD_SILU_MUL, PTX_SILU_MUL)?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&gate_slice);
    builder.arg(&up_slice);
    builder.arg(&out);
    let n_val = n as u32;
    builder.arg(&n_val);
    unsafe { builder.launch(cfg) }.w()?;

    let out_shape = gate_layout.shape().clone();

    drop(gate_storage);
    drop(up_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        out_shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Vectorized BF16 addition. Both inputs must be contiguous BF16 CUDA tensors.
pub fn vectorized_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (a_storage, a_layout) = a.storage_and_layout();
    let (b_storage, b_layout) = b.storage_and_layout();

    let a_cuda = match &*a_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("vectorized_add: requires CUDA"),
    };
    let b_cuda = match &*b_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("vectorized_add: requires CUDA"),
    };

    let n = a_layout.shape().elem_count();
    if n != b_layout.shape().elem_count() {
        candle_core::bail!("vectorized_add: shape mismatch");
    }
    if a_cuda.dtype() != DType::BF16 || b_cuda.dtype() != DType::BF16 {
        candle_core::bail!("vectorized_add: requires BF16");
    }

    let dev = a_cuda.device().clone();

    let a_slice = a_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(a_layout.start_offset()..);
    let b_slice = b_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(b_layout.start_offset()..);

    let out = unsafe { dev.alloc::<half::bf16>(n) }?;

    let threads = 256u32;
    let elems_per_thread = 8u32;
    let blocks = (n as u32 + threads * elems_per_thread - 1) / (threads * elems_per_thread);

    let func = dev.get_or_load_custom_func("vectorized_add_bf16", MOD_ADD, PTX_ADD)?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&a_slice);
    builder.arg(&b_slice);
    builder.arg(&out);
    let n_val = n as u32;
    builder.arg(&n_val);
    unsafe { builder.launch(cfg) }.w()?;

    let out_shape = a_layout.shape().clone();

    drop(a_storage);
    drop(b_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        out_shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

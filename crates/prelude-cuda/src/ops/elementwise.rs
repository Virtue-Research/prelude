use crate::device::{self as cb, CuResultExt, LaunchConfig, PushKernelArg, CudaStorageExt};
use crate::{MOD_ADD, MOD_SILU_MUL, PTX_ADD, PTX_SILU_MUL};
use prelude_core::tensor::{bail, DType, Result, Tensor};

/// Fused SiLU(gate) * up. Both inputs must be contiguous BF16 CUDA tensors.
pub fn fused_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let (gate_storage, gate_layout) = cb::storage_and_layout(&gate);
    let (up_storage, up_layout) = cb::storage_and_layout(&up);

    let gate_cuda = cb::as_cuda(&gate_storage, "fused_silu_mul")?;
    let up_cuda = cb::as_cuda(&up_storage, "fused_silu_mul: up")?;

    let n = gate_layout.shape().elem_count();
    if n != up_layout.shape().elem_count() {
        bail!("fused_silu_mul: shape mismatch");
    }
    if gate_cuda.dtype() != DType::BF16 || up_cuda.dtype() != DType::BF16 {
        bail!("fused_silu_mul: requires BF16");
    }

    let stream = gate_cuda.stream.clone();

    let gate_slice = gate_cuda
        .as_slice::<half::bf16>()?
        .slice(gate_layout.start_offset()..);
    let up_slice = up_cuda
        .as_slice::<half::bf16>()?
        .slice(up_layout.start_offset()..);

    let out = unsafe { stream.alloc::<half::bf16>(n) }.ce()?;

    let threads = 256u32;
    let elems_per_thread = 8u32;
    let blocks = (n as u32 + threads * elems_per_thread - 1) / (threads * elems_per_thread);

    let func = crate::device::get_or_load_func(gate_cuda.device(), "fused_silu_mul_bf16", MOD_SILU_MUL, PTX_SILU_MUL)?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&gate_slice);
    builder.arg(&up_slice);
    builder.arg(&out);
    let n_val = n as u32;
    builder.arg(&n_val);
    unsafe { builder.launch(cfg) }.ce()?;

    let out_shape = gate_layout.shape().clone();

    Ok(cb::tensor_from_cuda(out, stream, out_shape))
}

/// Vectorized BF16 addition. Both inputs must be contiguous BF16 CUDA tensors.
pub fn vectorized_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (a_storage, a_layout) = cb::storage_and_layout(&a);
    let (b_storage, b_layout) = cb::storage_and_layout(&b);

    let a_cuda = cb::as_cuda(&a_storage, "vectorized_add")?;
    let b_cuda = cb::as_cuda(&b_storage, "vectorized_add: b")?;

    let n = a_layout.shape().elem_count();
    if n != b_layout.shape().elem_count() {
        bail!("vectorized_add: shape mismatch");
    }
    if a_cuda.dtype() != DType::BF16 || b_cuda.dtype() != DType::BF16 {
        bail!("vectorized_add: requires BF16");
    }

    let stream = a_cuda.stream.clone();

    let a_slice = a_cuda
        .as_slice::<half::bf16>()?
        .slice(a_layout.start_offset()..);
    let b_slice = b_cuda
        .as_slice::<half::bf16>()?
        .slice(b_layout.start_offset()..);

    let out = unsafe { stream.alloc::<half::bf16>(n) }.ce()?;

    let threads = 256u32;
    let elems_per_thread = 8u32;
    let blocks = (n as u32 + threads * elems_per_thread - 1) / (threads * elems_per_thread);

    let func = crate::device::get_or_load_func(a_cuda.device(), "vectorized_add_bf16", MOD_ADD, PTX_ADD)?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&a_slice);
    builder.arg(&b_slice);
    builder.arg(&out);
    let n_val = n as u32;
    builder.arg(&n_val);
    unsafe { builder.launch(cfg) }.ce()?;

    let out_shape = a_layout.shape().clone();

    Ok(cb::tensor_from_cuda(out, stream, out_shape))
}

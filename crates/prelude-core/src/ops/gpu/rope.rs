use candle_core::backend::BackendStorage;
use super::{MOD_QKNORM_ROPE, PTX_QKNORM_ROPE};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Result, Tensor};

/// Fused per-head QK-Norm + RoPE for varlen attention.
/// Combines RMSNorm normalization and Rotary Position Embeddings in one kernel.
/// Input: [total_tokens, num_heads, head_dim], output same shape.
/// Eliminates index_select + separate norm + separate rope kernel launches.
pub fn fused_qknorm_rope_varlen(
    input: &Tensor,     // [total_tokens, num_heads, head_dim]
    weight: &Tensor,    // [head_dim]
    cos_table: &Tensor, // [max_seq_len, head_dim/2]
    sin_table: &Tensor, // [max_seq_len, head_dim/2]
    pos_ids: &Tensor,   // [total_tokens] U32
    eps: f64,
) -> Result<Tensor> {
    let (x_storage, x_layout) = input.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (cos_storage, cos_layout) = cos_table.storage_and_layout();
    let (sin_storage, sin_layout) = sin_table.storage_and_layout();
    let (pos_storage, pos_layout) = pos_ids.storage_and_layout();

    let x_cuda = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope: requires CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope: weight requires CUDA"),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope: cos requires CUDA"),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope: sin requires CUDA"),
    };
    let pos_cuda = match &*pos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope: pos_ids requires CUDA"),
    };

    if x_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fused_qknorm_rope: requires BF16");
    }

    let shape = x_layout.shape();
    let dims = shape.dims();
    let total_tokens = dims[0];
    let num_heads = dims[1];
    let head_dim = dims[2];
    let n_rows = total_tokens * num_heads;
    let n = shape.elem_count();

    let dev = x_cuda.device().clone();

    let x_slice = x_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(x_layout.start_offset()..);
    let w_slice = w_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(w_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(sin_layout.start_offset()..);
    let pos_slice = pos_cuda
        .as_cuda_slice::<u32>()?
        .slice(pos_layout.start_offset()..);

    let out = unsafe { dev.alloc::<half::bf16>(n) }?;

    let block = 256u32; // 8 warps per block
    let rows_per_block = block / 32;
    let grid = (n_rows as u32 + rows_per_block - 1) / rows_per_block;

    let func =
        dev.get_or_load_custom_func("fused_qknorm_rope_bf16", MOD_QKNORM_ROPE, PTX_QKNORM_ROPE)?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&x_slice);
    builder.arg(&w_slice);
    builder.arg(&cos_slice);
    builder.arg(&sin_slice);
    builder.arg(&pos_slice);
    builder.arg(&out);
    let n_rows_val = n_rows as u32;
    let num_heads_val = num_heads as u32;
    let d_val = head_dim as u32;
    let eps_val = eps as f32;
    builder.arg(&n_rows_val);
    builder.arg(&num_heads_val);
    builder.arg(&d_val);
    builder.arg(&eps_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_storage);
    drop(w_storage);
    drop(cos_storage);
    drop(sin_storage);
    drop(pos_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    let out_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        shape.clone(),
        candle_core::op::BackpropOp::none(),
        false,
    );

    Ok(out_tensor)
}

/// Fused per-head QK-Norm + RoPE for THD [B,L,H,D] layout.
/// Position is derived from row index: pos = (row / num_heads) % seq_len + offset.
/// Input: [B*L*H, head_dim], output same shape.
pub fn fused_qknorm_rope_thd(
    input: &Tensor,     // [B*L*H, head_dim]
    weight: &Tensor,    // [head_dim]
    cos_table: &Tensor, // [max_seq_len, head_dim/2]
    sin_table: &Tensor, // [max_seq_len, head_dim/2]
    num_heads: usize,
    seq_len: usize,
    offset: usize,
    eps: f64,
) -> Result<Tensor> {
    let (x_storage, x_layout) = input.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (cos_storage, cos_layout) = cos_table.storage_and_layout();
    let (sin_storage, sin_layout) = sin_table.storage_and_layout();

    let x_cuda = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope_thd: requires CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope_thd: weight requires CUDA"),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope_thd: cos requires CUDA"),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope_thd: sin requires CUDA"),
    };

    if x_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fused_qknorm_rope_thd: requires BF16");
    }

    let shape = x_layout.shape();
    let dims = shape.dims();
    let n_rows = dims[0]; // B * L * num_heads
    let head_dim = dims[1];
    let n = shape.elem_count();

    let dev = x_cuda.device().clone();

    let x_slice = x_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(x_layout.start_offset()..);
    let w_slice = w_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(w_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(sin_layout.start_offset()..);

    let out = unsafe { dev.alloc::<half::bf16>(n) }?;

    let block = 256u32;
    let rows_per_block = block / 32;
    let grid = (n_rows as u32 + rows_per_block - 1) / rows_per_block;

    let func = dev.get_or_load_custom_func(
        "fused_qknorm_rope_thd_bf16",
        MOD_QKNORM_ROPE,
        PTX_QKNORM_ROPE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&x_slice);
    builder.arg(&w_slice);
    builder.arg(&cos_slice);
    builder.arg(&sin_slice);
    builder.arg(&out);
    let n_rows_val = n_rows as u32;
    let num_heads_val = num_heads as u32;
    let seq_len_val = seq_len as u32;
    let d_val = head_dim as u32;
    let offset_val = offset as u32;
    let eps_val = eps as f32;
    builder.arg(&n_rows_val);
    builder.arg(&num_heads_val);
    builder.arg(&seq_len_val);
    builder.arg(&d_val);
    builder.arg(&offset_val);
    builder.arg(&eps_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_storage);
    drop(w_storage);
    drop(cos_storage);
    drop(sin_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    let out_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        shape.clone(),
        candle_core::op::BackpropOp::none(),
        false,
    );

    Ok(out_tensor)
}

/// CUDA-graph-safe variant of fused_qknorm_rope_thd.
/// Reads position offset from a device pointer instead of a kernel argument,
/// allowing the kernel to be captured in a CUDA graph and replayed with
/// different offsets by updating the device buffer between replays.
pub fn fused_qknorm_rope_thd_graphsafe(
    input: &Tensor,      // [B*L*H, head_dim]
    weight: &Tensor,     // [head_dim]
    cos_table: &Tensor,  // [max_seq_len, head_dim/2]
    sin_table: &Tensor,  // [max_seq_len, head_dim/2]
    offset_ptr: &Tensor, // [1] U32 on GPU — position offset
    num_heads: usize,
    seq_len: usize,
    eps: f64,
) -> Result<Tensor> {
    let (x_storage, x_layout) = input.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (cos_storage, cos_layout) = cos_table.storage_and_layout();
    let (sin_storage, sin_layout) = sin_table.storage_and_layout();
    let (off_storage, off_layout) = offset_ptr.storage_and_layout();

    let x_cuda = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope_thd_graphsafe: requires CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope_thd_graphsafe: weight requires CUDA"),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope_thd_graphsafe: cos requires CUDA"),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope_thd_graphsafe: sin requires CUDA"),
    };
    let off_cuda = match &*off_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_rope_thd_graphsafe: offset_ptr requires CUDA"),
    };

    if x_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fused_qknorm_rope_thd_graphsafe: requires BF16");
    }

    let shape = x_layout.shape();
    let dims = shape.dims();
    let n_rows = dims[0];
    let head_dim = dims[1];
    let n = shape.elem_count();

    let dev = x_cuda.device().clone();

    let x_slice = x_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(x_layout.start_offset()..);
    let w_slice = w_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(w_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(sin_layout.start_offset()..);
    let off_slice = off_cuda
        .as_cuda_slice::<u32>()?
        .slice(off_layout.start_offset()..);

    let out = unsafe { dev.alloc::<half::bf16>(n) }?;

    let block = 256u32;
    let rows_per_block = block / 32;
    let grid = (n_rows as u32 + rows_per_block - 1) / rows_per_block;

    let func = dev.get_or_load_custom_func(
        "fused_qknorm_rope_thd_graphsafe_bf16",
        MOD_QKNORM_ROPE,
        PTX_QKNORM_ROPE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&x_slice);
    builder.arg(&w_slice);
    builder.arg(&cos_slice);
    builder.arg(&sin_slice);
    builder.arg(&out);
    builder.arg(&off_slice);
    let n_rows_val = n_rows as u32;
    let num_heads_val = num_heads as u32;
    let seq_len_val = seq_len as u32;
    let d_val = head_dim as u32;
    let eps_val = eps as f32;
    builder.arg(&n_rows_val);
    builder.arg(&num_heads_val);
    builder.arg(&seq_len_val);
    builder.arg(&d_val);
    builder.arg(&eps_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_storage);
    drop(w_storage);
    drop(cos_storage);
    drop(sin_storage);
    drop(off_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    let out_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        shape.clone(),
        candle_core::op::BackpropOp::none(),
        false,
    );

    Ok(out_tensor)
}

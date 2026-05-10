use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Result, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{MOD_QKNORM_ROPE, PTX_QKNORM_ROPE};

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
    let n = total_tokens * num_heads * head_dim;

    // Token stride in elements: contiguous = num_heads*head_dim, fused QKV narrow = N_fused.
    let token_stride = x_layout.stride()[0];

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
    let token_stride_val = token_stride as u32;
    builder.arg(&n_rows_val);
    builder.arg(&num_heads_val);
    builder.arg(&d_val);
    builder.arg(&eps_val);
    builder.arg(&token_stride_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_storage);
    drop(w_storage);
    drop(cos_storage);
    drop(sin_storage);
    drop(pos_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        shape.clone(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Fused per-head RMSNorm + partial RoPE for Qwen3.5-style attention.
/// Input: `[total_tokens, num_heads, head_dim]` BF16, F32 norm weight.
/// RoPE is applied to the leading `rotary_dim` channels; the rest pass through
/// after normalization.
pub fn fused_qknorm_partial_rope_varlen_f32_weight(
    input: &Tensor,
    weight: &Tensor,
    cos_table: &Tensor,
    sin_table: &Tensor,
    pos_ids: &Tensor,
    rotary_dim: usize,
    eps: f64,
) -> Result<Tensor> {
    let (x_storage, x_layout) = input.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (cos_storage, cos_layout) = cos_table.storage_and_layout();
    let (sin_storage, sin_layout) = sin_table.storage_and_layout();
    let (pos_storage, pos_layout) = pos_ids.storage_and_layout();

    let x_cuda = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_partial_rope: requires CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_partial_rope: weight requires CUDA"),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_partial_rope: cos requires CUDA"),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_partial_rope: sin requires CUDA"),
    };
    let pos_cuda = match &*pos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qknorm_partial_rope: pos_ids requires CUDA"),
    };

    if x_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fused_qknorm_partial_rope: input must be BF16");
    }
    if w_cuda.dtype() != DType::F32 {
        candle_core::bail!("fused_qknorm_partial_rope: weight must be F32");
    }

    let shape = x_layout.shape();
    let dims = shape.dims();
    if dims.len() != 3 {
        candle_core::bail!("fused_qknorm_partial_rope: input must be 3D");
    }
    let total_tokens = dims[0];
    let num_heads = dims[1];
    let head_dim = dims[2];
    if head_dim > 256 || head_dim % 32 != 0 {
        candle_core::bail!("fused_qknorm_partial_rope: unsupported head_dim={head_dim}");
    }
    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        candle_core::bail!("fused_qknorm_partial_rope: invalid rotary_dim={rotary_dim}");
    }
    let elems_per_lane = head_dim / 32;
    if (rotary_dim / 2) % elems_per_lane != 0 {
        candle_core::bail!(
            "fused_qknorm_partial_rope: rotary_dim={rotary_dim} incompatible with head_dim={head_dim}"
        );
    }
    if weight.elem_count() != head_dim {
        candle_core::bail!("fused_qknorm_partial_rope: weight shape mismatch");
    }

    let n_rows = total_tokens * num_heads;
    let n = n_rows * head_dim;
    let token_stride = x_layout.stride()[0];
    let dev = x_cuda.device().clone();

    let x_slice = x_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(x_layout.start_offset()..);
    let w_slice = w_cuda
        .as_cuda_slice::<f32>()?
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

    let block = 256u32;
    let rows_per_block = block / 32;
    let grid = (n_rows as u32 + rows_per_block - 1) / rows_per_block;

    let func = dev.get_or_load_custom_func(
        "fused_qknorm_partial_rope_bf16_f32_weight",
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
    builder.arg(&pos_slice);
    builder.arg(&out);
    let n_rows_val = n_rows as u32;
    let num_heads_val = num_heads as u32;
    let d_val = head_dim as u32;
    let rotary_dim_val = rotary_dim as u32;
    let eps_val = eps as f32;
    let token_stride_val = token_stride as u32;
    builder.arg(&n_rows_val);
    builder.arg(&num_heads_val);
    builder.arg(&d_val);
    builder.arg(&rotary_dim_val);
    builder.arg(&eps_val);
    builder.arg(&token_stride_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_storage);
    drop(w_storage);
    drop(cos_storage);
    drop(sin_storage);
    drop(pos_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        shape.clone(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

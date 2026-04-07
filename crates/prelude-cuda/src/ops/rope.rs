use crate::device::{self as cb, CuResultExt, LaunchConfig, PushKernelArg, CudaStorageExt};
use crate::{MOD_QKNORM_ROPE, PTX_QKNORM_ROPE};
use prelude_core::tensor::{bail, DType, Result, Tensor};

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
    let (x_storage, x_layout) = cb::storage_and_layout(&input);
    let (w_storage, w_layout) = cb::storage_and_layout(&weight);
    let (cos_storage, cos_layout) = cb::storage_and_layout(&cos_table);
    let (sin_storage, sin_layout) = cb::storage_and_layout(&sin_table);
    let (pos_storage, pos_layout) = cb::storage_and_layout(&pos_ids);

    let x_cuda = cb::as_cuda(&x_storage, "fused_qknorm_rope")?;
    let w_cuda = cb::as_cuda(&w_storage, "fused_qknorm_rope: weight")?;
    let cos_cuda = cb::as_cuda(&cos_storage, "fused_qknorm_rope: cos")?;
    let sin_cuda = cb::as_cuda(&sin_storage, "fused_qknorm_rope: sin")?;
    let pos_cuda = cb::as_cuda(&pos_storage, "fused_qknorm_rope: pos_ids")?;

    if x_cuda.dtype() != DType::BF16 {
        bail!("fused_qknorm_rope: requires BF16");
    }

    let shape = x_layout.shape();
    let dims = shape.dims();
    let total_tokens = dims[0];
    let num_heads = dims[1];
    let head_dim = dims[2];
    let n_rows = total_tokens * num_heads;
    let n = shape.elem_count();

    let stream = x_cuda.stream.clone();

    let x_slice = x_cuda
        .as_slice::<half::bf16>()?
        .slice(x_layout.start_offset()..);
    let w_slice = w_cuda
        .as_slice::<half::bf16>()?
        .slice(w_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_slice::<half::bf16>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_slice::<half::bf16>()?
        .slice(sin_layout.start_offset()..);
    let pos_slice = pos_cuda
        .as_slice::<u32>()?
        .slice(pos_layout.start_offset()..);

    let out = unsafe { stream.alloc::<half::bf16>(n) }.ce()?;

    let block = 256u32; // 8 warps per block
    let rows_per_block = block / 32;
    let grid = (n_rows as u32 + rows_per_block - 1) / rows_per_block;

    let func =
        crate::device::get_or_load_func(x_cuda.device(), "fused_qknorm_rope_bf16", MOD_QKNORM_ROPE, PTX_QKNORM_ROPE)?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
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
    unsafe { builder.launch(cfg) }.ce()?;

    Ok(cb::tensor_from_cuda(out, stream, shape.clone()))
}

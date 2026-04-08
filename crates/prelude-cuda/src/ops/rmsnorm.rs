use crate::device::{self as cb, CuResultExt, LaunchConfig, PushKernelArg};
use crate::{MOD_ADD_RMSNORM, MOD_RMSNORM, PTX_ADD_RMSNORM, PTX_RMSNORM};
use prelude_core::tensor::{bail, DType, Result, Tensor};

/// Fast standalone RMSNorm with register caching and vectorized loads.
/// Replaces naive rmsnorm_bf16 which runs at ~13% of bandwidth.
pub fn fast_rmsnorm(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let (x_storage, x_layout) = cb::storage_and_layout(&input);
    let (w_storage, w_layout) = cb::storage_and_layout(&weight);

    let x_cuda = cb::as_cuda(&x_storage, "fast_rmsnorm")?;
    let w_cuda = cb::as_cuda(&w_storage, "fast_rmsnorm: weight")?;

    if x_cuda.dtype() != DType::BF16 {
        bail!("fast_rmsnorm: requires BF16");
    }

    let shape = x_layout.shape();
    let dims = shape.dims();
    let d = *dims.last().unwrap();
    let n_rows = shape.elem_count() / d;

    let stream = x_cuda.stream.clone();
    let n = shape.elem_count();

    let x_slice = x_cuda
        .as_slice::<half::bf16>()?
        .slice(x_layout.start_offset()..);
    let w_slice = w_cuda
        .as_slice::<half::bf16>()?
        .slice(w_layout.start_offset()..);

    let out = unsafe { stream.alloc::<half::bf16>(n) }.ce()?;

    // For small D (<=256): multi-row warp-parallel, 1 warp per row, 256 threads = 8 rows/block
    // For large D (1024): 1 row per block, 128 threads, vectorized float4 loads
    let (block_size, grid_size, shared_mem) = if d <= 256 {
        let block = 256u32; // 8 warps = 8 rows per block
        let rows_per_block = block / 32;
        let grid = (n_rows as u32 + rows_per_block - 1) / rows_per_block;
        (block, grid, 0u32) // no shared memory needed for warp-only reduction
    } else {
        let block = if d == 1024 { 128u32 } else { 256u32 };
        let num_warps = (block + 31) / 32;
        (block, n_rows as u32, num_warps * 4)
    };

    let func = crate::device::get_or_load_func(x_cuda.device(), "fast_rmsnorm_bf16", MOD_RMSNORM, PTX_RMSNORM)?;
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x_slice);
    builder.arg(&w_slice);
    builder.arg(&out);
    let n_rows_val = n_rows as u32;
    let d_val = d as u32;
    let eps_val = eps as f32;
    builder.arg(&n_rows_val);
    builder.arg(&d_val);
    builder.arg(&eps_val);
    unsafe { builder.launch(cfg) }.ce()?;

    Ok(cb::tensor_from_cuda(out, stream, shape.clone()))
}

/// Fused residual add + RMSNorm.
/// Computes: sum = x + residual; normed = rmsnorm(sum, weight, eps).
/// Returns (sum, normed).
pub fn fused_add_rmsnorm(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let (x_storage, x_layout) = cb::storage_and_layout(&x);
    let (r_storage, r_layout) = cb::storage_and_layout(&residual);
    let (w_storage, w_layout) = cb::storage_and_layout(&weight);

    let x_cuda = cb::as_cuda(&x_storage, "fused_add_rmsnorm")?;
    let r_cuda = cb::as_cuda(&r_storage, "fused_add_rmsnorm: residual")?;
    let w_cuda = cb::as_cuda(&w_storage, "fused_add_rmsnorm: weight")?;

    if x_cuda.dtype() != DType::BF16 {
        bail!("fused_add_rmsnorm: requires BF16");
    }

    let shape = x_layout.shape();
    let dims = shape.dims();
    let d = *dims.last().unwrap();
    let n_rows = shape.elem_count() / d;

    let stream = x_cuda.stream.clone();
    let n = shape.elem_count();

    let x_slice = x_cuda
        .as_slice::<half::bf16>()?
        .slice(x_layout.start_offset()..);
    let r_slice = r_cuda
        .as_slice::<half::bf16>()?
        .slice(r_layout.start_offset()..);
    let w_slice = w_cuda
        .as_slice::<half::bf16>()?
        .slice(w_layout.start_offset()..);

    let out_sum = unsafe { stream.alloc::<half::bf16>(n) }.ce()?;
    let out_norm = unsafe { stream.alloc::<half::bf16>(n) }.ce()?;

    // Launch: one block per row, 256 threads per block
    let block_size = 256u32;
    let num_warps = (block_size + 31) / 32;
    let shared_mem = num_warps * 4; // float per warp for reduction

    let func =
        crate::device::get_or_load_func(x_cuda.device(), "fused_add_rmsnorm_bf16", MOD_ADD_RMSNORM, PTX_ADD_RMSNORM)?;
    let cfg = LaunchConfig {
        grid_dim: (n_rows as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let mut builder = stream.launch_builder(&func);
    builder.arg(&x_slice);
    builder.arg(&r_slice);
    builder.arg(&w_slice);
    builder.arg(&out_sum);
    builder.arg(&out_norm);
    let n_rows_val = n_rows as u32;
    let d_val = d as u32;
    let eps_val = eps as f32;
    builder.arg(&n_rows_val);
    builder.arg(&d_val);
    builder.arg(&eps_val);
    unsafe { builder.launch(cfg) }.ce()?;

    let sum_tensor = cb::tensor_from_cuda(out_sum, stream.clone(), shape.clone());
    let norm_tensor = cb::tensor_from_cuda(out_norm, stream, shape.clone());

    Ok((sum_tensor, norm_tensor))
}

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Result, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{MOD_RMSNORM, PTX_RMSNORM, PTX_RMSNORM_GATED};

/// Fast standalone RMSNorm with register caching and vectorized loads.
/// Replaces naive rmsnorm_bf16 which runs at ~13% of bandwidth.
pub fn fast_rmsnorm(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let (x_storage, x_layout) = input.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();

    let x_cuda = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fast_rmsnorm: requires CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fast_rmsnorm: requires CUDA"),
    };

    if x_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fast_rmsnorm: requires BF16");
    }

    let shape = x_layout.shape();
    let dims = shape.dims();
    let d = *dims.last().unwrap();
    let n_rows = shape.elem_count() / d;

    let dev = x_cuda.device().clone();
    let n = shape.elem_count();

    let x_slice = x_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(x_layout.start_offset()..);
    let w_slice = w_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(w_layout.start_offset()..);

    let out = unsafe { dev.alloc::<half::bf16>(n) }?;

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

    let func = dev.get_or_load_custom_func("fast_rmsnorm_bf16", MOD_RMSNORM, PTX_RMSNORM)?;
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let mut builder = func.builder();
    builder.arg(&x_slice);
    builder.arg(&w_slice);
    builder.arg(&out);
    let n_rows_val = n_rows as u32;
    let d_val = d as u32;
    let eps_val = eps as f32;
    builder.arg(&n_rows_val);
    builder.arg(&d_val);
    builder.arg(&eps_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_storage);
    drop(w_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    let out_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        shape.clone(),
        candle_core::op::BackpropOp::none(),
        false,
    );

    Ok(out_tensor)
}

/// Fused RMSNorm + SiLU gate:  output = RMSNorm(x, weight) * SiLU(gate)
/// x, gate: BF16 [N, D].  weight: F32 [D].  output: BF16 [N, D].
pub fn fast_rmsnorm_gated(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let (x_storage, x_layout) = x.storage_and_layout();
    let (g_storage, g_layout) = gate.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();

    let x_cuda = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("rmsnorm_gated: requires CUDA"),
    };
    let g_cuda = match &*g_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("rmsnorm_gated: requires CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("rmsnorm_gated: requires CUDA"),
    };

    if x_cuda.dtype() != DType::BF16 {
        candle_core::bail!("rmsnorm_gated: x must be BF16");
    }
    if w_cuda.dtype() != DType::F32 {
        candle_core::bail!("rmsnorm_gated: weight must be F32");
    }

    let shape = x_layout.shape();
    let dims = shape.dims();
    let d = *dims.last().unwrap();
    let n_rows = shape.elem_count() / d;
    let n = shape.elem_count();

    let dev = x_cuda.device().clone();

    let x_slice = x_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(x_layout.start_offset()..);
    let g_slice = g_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(g_layout.start_offset()..);
    let w_slice = w_cuda
        .as_cuda_slice::<f32>()?
        .slice(w_layout.start_offset()..);

    let out = unsafe { dev.alloc::<half::bf16>(n) }?;

    let (block_size, grid_size, shared_mem) = if d <= 256 {
        let block = 256u32;
        let rows_per_block = block / 32;
        let grid = (n_rows as u32 + rows_per_block - 1) / rows_per_block;
        (block, grid, 0u32)
    } else {
        let block = 256u32;
        let num_warps = (block + 31) / 32;
        (block, n_rows as u32, num_warps * 4)
    };

    let func =
        dev.get_or_load_custom_func("rmsnorm_gated_bf16", "rmsnorm_gated", PTX_RMSNORM_GATED)?;
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: shared_mem,
    };
    let mut builder = func.builder();
    builder.arg(&x_slice);
    builder.arg(&g_slice);
    builder.arg(&w_slice);
    builder.arg(&out);
    let n_rows_val = n_rows as u32;
    let d_val = d as u32;
    let eps_val = eps as f32;
    builder.arg(&n_rows_val);
    builder.arg(&d_val);
    builder.arg(&eps_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(x_storage);
    drop(g_storage);
    drop(w_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        shape.clone(),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Result, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::{MOD_QKNORM_ROPE, PTX_QKNORM_ROPE};

/// D=128 specialized qknorm+rope kernel enabled? (escape hatch:
/// PRELUDE_QKNORM_D128=0 → generic kernel). Parsed centrally in prelude-core.
fn qknorm_d128_enabled() -> bool {
    prelude_core::config::attn_flags::qknorm_d128_enabled()
}

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

/// Merged Q+K fused QK-Norm + RoPE in a single kernel launch.
/// Equivalent to calling `fused_qknorm_rope_varlen` separately on `q` and `k`,
/// but issues ONE kernel covering both tensors' rows — halving the launch count
/// per attention layer and packing the small K grid into the Q grid's tail.
/// `q`/`k`: `[total_tokens, num_heads, head_dim]` (head_dim shared); returns
/// `(q_out, k_out)` each contiguous in the input's shape.
pub fn fused_qknorm_rope_qk_varlen(
    q: &Tensor,
    k: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos_table: &Tensor,
    sin_table: &Tensor,
    pos_ids: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    fused_qknorm_rope_qk_varlen_force(
        q, k, q_weight, k_weight, cos_table, sin_table, pos_ids, eps, None,
    )
}

/// Like [`fused_qknorm_rope_qk_varlen`] but with an explicit kernel-variant
/// override (`Some(true)` = force the d128 specialization, `Some(false)` =
/// force the generic kernel). Used by the bit-exactness tests to compare both
/// kernels in one process (the env gate is OnceLock-cached).
#[allow(clippy::too_many_arguments)]
pub fn fused_qknorm_rope_qk_varlen_force(
    q: &Tensor,
    k: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos_table: &Tensor,
    sin_table: &Tensor,
    pos_ids: &Tensor,
    eps: f64,
    force_d128: Option<bool>,
) -> Result<(Tensor, Tensor)> {
    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (qw_storage, qw_layout) = q_weight.storage_and_layout();
    let (kw_storage, kw_layout) = k_weight.storage_and_layout();
    let (cos_storage, cos_layout) = cos_table.storage_and_layout();
    let (sin_storage, sin_layout) = sin_table.storage_and_layout();
    let (pos_storage, pos_layout) = pos_ids.storage_and_layout();

    let cuda = |s: &candle_core::Storage, what: &str| -> Result<()> {
        match s {
            candle_core::Storage::Cuda(_) => Ok(()),
            _ => candle_core::bail!("fused_qknorm_rope_qk: {what} requires CUDA"),
        }
    };
    cuda(&q_storage, "q")?;
    cuda(&k_storage, "k")?;
    cuda(&qw_storage, "q_weight")?;
    cuda(&kw_storage, "k_weight")?;
    cuda(&cos_storage, "cos")?;
    cuda(&sin_storage, "sin")?;
    cuda(&pos_storage, "pos_ids")?;

    let q_cuda = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    let k_cuda = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    let qw_cuda = match &*qw_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    let kw_cuda = match &*kw_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => unreachable!(),
    };
    let pos_cuda = match &*pos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => unreachable!(),
    };

    if q_cuda.dtype() != DType::BF16 || k_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fused_qknorm_rope_qk: requires BF16");
    }

    let q_dims = q_layout.shape().dims();
    let k_dims = k_layout.shape().dims();
    let total_tokens = q_dims[0];
    let q_num_heads = q_dims[1];
    let head_dim = q_dims[2];
    let k_num_heads = k_dims[1];
    if k_dims[0] != total_tokens || k_dims[2] != head_dim {
        candle_core::bail!("fused_qknorm_rope_qk: q/k must share total_tokens and head_dim");
    }
    let q_rows = total_tokens * q_num_heads;
    let k_rows = total_tokens * k_num_heads;

    let q_token_stride = q_layout.stride()[0];
    let k_token_stride = k_layout.stride()[0];

    let dev = q_cuda.device().clone();

    let q_slice = q_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(q_layout.start_offset()..);
    let k_slice = k_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(k_layout.start_offset()..);
    let qw_slice = qw_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(qw_layout.start_offset()..);
    let kw_slice = kw_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(kw_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(sin_layout.start_offset()..);
    let pos_slice = pos_cuda
        .as_cuda_slice::<u32>()?
        .slice(pos_layout.start_offset()..);

    let q_out = unsafe { dev.alloc::<half::bf16>(q_rows * head_dim) }?;
    let k_out = unsafe { dev.alloc::<half::bf16>(k_rows * head_dim) }?;

    let block = 256u32; // 8 warps per block
    let rows_per_block = block / 32;
    let total_rows = (q_rows + k_rows) as u32;
    let grid = (total_rows + rows_per_block - 1) / rows_per_block;

    // D=128 fast path: fully-unrolled, uint2-vectorized variant. The generic
    // kernel's runtime-bound lane arrays (vals[8]/partner[8]) live in local
    // memory (64B stack frame) with scalar 2B global accesses — measured ~3x
    // off the memory roof. Requires 8B alignment of every lane group: all
    // element offsets/strides must be multiples of 4. Bit-exact same math.
    // PRELUDE_QKNORM_D128=0 falls back to the generic kernel.
    let use_d128 = head_dim == 128
        && force_d128.unwrap_or_else(qknorm_d128_enabled)
        && q_layout.start_offset() % 4 == 0
        && k_layout.start_offset() % 4 == 0
        && qw_layout.start_offset() % 4 == 0
        && kw_layout.start_offset() % 4 == 0
        && cos_layout.start_offset() % 4 == 0
        && sin_layout.start_offset() % 4 == 0
        && q_token_stride % 4 == 0
        && k_token_stride % 4 == 0;

    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let q_rows_val = q_rows as u32;
    let k_rows_val = k_rows as u32;
    let q_num_heads_val = q_num_heads as u32;
    let k_num_heads_val = k_num_heads as u32;
    let eps_val = eps as f32;
    let q_token_stride_val = q_token_stride as u32;
    let k_token_stride_val = k_token_stride as u32;
    if use_d128 {
        let func = dev.get_or_load_custom_func(
            "fused_qknorm_rope_qk_d128_bf16",
            MOD_QKNORM_ROPE,
            PTX_QKNORM_ROPE,
        )?;
        let mut builder = func.builder();
        builder.arg(&q_slice);
        builder.arg(&k_slice);
        builder.arg(&qw_slice);
        builder.arg(&kw_slice);
        builder.arg(&cos_slice);
        builder.arg(&sin_slice);
        builder.arg(&pos_slice);
        builder.arg(&q_out);
        builder.arg(&k_out);
        builder.arg(&q_rows_val);
        builder.arg(&k_rows_val);
        builder.arg(&q_num_heads_val);
        builder.arg(&k_num_heads_val);
        builder.arg(&eps_val);
        builder.arg(&q_token_stride_val);
        builder.arg(&k_token_stride_val);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let func = dev.get_or_load_custom_func(
            "fused_qknorm_rope_qk_bf16",
            MOD_QKNORM_ROPE,
            PTX_QKNORM_ROPE,
        )?;
        let mut builder = func.builder();
        builder.arg(&q_slice);
        builder.arg(&k_slice);
        builder.arg(&qw_slice);
        builder.arg(&kw_slice);
        builder.arg(&cos_slice);
        builder.arg(&sin_slice);
        builder.arg(&pos_slice);
        builder.arg(&q_out);
        builder.arg(&k_out);
        let d_val = head_dim as u32;
        builder.arg(&q_rows_val);
        builder.arg(&k_rows_val);
        builder.arg(&q_num_heads_val);
        builder.arg(&k_num_heads_val);
        builder.arg(&d_val);
        builder.arg(&eps_val);
        builder.arg(&q_token_stride_val);
        builder.arg(&k_token_stride_val);
        unsafe { builder.launch(cfg) }.w()?;
    }

    drop(q_storage);
    drop(k_storage);
    drop(qw_storage);
    drop(kw_storage);
    drop(cos_storage);
    drop(sin_storage);
    drop(pos_storage);

    let q_storage_out = candle_core::CudaStorage::wrap_cuda_slice(q_out, dev.clone());
    let k_storage_out = candle_core::CudaStorage::wrap_cuda_slice(k_out, dev);
    let q_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(q_storage_out),
        q_layout.shape().clone(),
        candle_core::op::BackpropOp::none(),
        false,
    );
    let k_tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(k_storage_out),
        k_layout.shape().clone(),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok((q_tensor, k_tensor))
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

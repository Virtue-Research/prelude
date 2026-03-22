use candle_core::backend::BackendStorage;
use super::{MOD_KNORM_ROPE_KV_WRITE, PTX_KNORM_ROPE_KV_WRITE};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{DType, Result, Tensor};

// ── Fused K-Norm + RoPE + KV paged cache write ──────────────────────

/// Returns true if the fused KV cache write kernel is enabled.
pub fn fused_kv_cache_write_enabled() -> bool {
    crate::config::global_runtime()
        .map(|r| r.fused_kv_cache_write)
        .unwrap_or(false)
}

/// Fused K-Norm + RoPE + KV paged cache write for THD decode layout.
///
/// Combines: K RMSNorm + K RoPE + scatter-write K/V to paged cache in one kernel.
/// Eliminates the separate `reshape_and_cache` kernel launch and intermediate K buffer.
///
/// - `k`: `[B * num_kv_heads, head_dim]` BF16 (pre-flattened K from projection)
/// - `v`: `[B * num_kv_heads, head_dim]` BF16 (pre-flattened V from projection)
/// - `k_norm_weight`: `[head_dim]` BF16
/// - `cos_table`, `sin_table`: `[max_seq_len, head_dim/2]` BF16
/// - `key_cache`: `[num_blocks, num_kv_heads, head_size/x, block_size, x]` BF16
/// - `value_cache`: `[num_blocks, num_kv_heads, head_size, block_size]` BF16
/// - `slot_mapping`: `[B]` I64
#[cfg(feature = "cuda")]
pub fn fused_knorm_rope_kv_cache_write_thd(
    k: &Tensor,             // [B * num_kv_heads, head_dim]
    v: &Tensor,             // [B * num_kv_heads, head_dim]
    k_norm_weight: &Tensor, // [head_dim]
    cos_table: &Tensor,     // [max_seq_len, head_dim/2]
    sin_table: &Tensor,     // [max_seq_len, head_dim/2]
    key_cache: &Tensor,     // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    value_cache: &Tensor,   // [num_blocks, num_kv_heads, head_size, block_size]
    slot_mapping: &Tensor,  // [B] I64
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    x: usize,
    seq_len: usize,
    offset: usize,
    eps: f64,
) -> Result<()> {
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (w_storage, w_layout) = k_norm_weight.storage_and_layout();
    let (cos_storage, cos_layout) = cos_table.storage_and_layout();
    let (sin_storage, sin_layout) = sin_table.storage_and_layout();
    let (kc_storage, kc_layout) = key_cache.storage_and_layout();
    let (vc_storage, vc_layout) = value_cache.storage_and_layout();
    let (sm_storage, sm_layout) = slot_mapping.storage_and_layout();

    let k_cuda = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_thd: k requires CUDA"),
    };
    let v_cuda = match &*v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_thd: v requires CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_thd: weight requires CUDA"),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_thd: cos requires CUDA"),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_thd: sin requires CUDA"),
    };
    let kc_cuda = match &*kc_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_thd: key_cache requires CUDA"),
    };
    let vc_cuda = match &*vc_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_thd: value_cache requires CUDA"),
    };
    let sm_cuda = match &*sm_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_thd: slot_mapping requires CUDA"),
    };

    if k_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fused_knorm_rope_kv_write_thd: requires BF16");
    }

    let dev = k_cuda.device().clone();
    let total_kv_rows = k_layout.shape().dims()[0];

    let k_slice = k_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(k_layout.start_offset()..);
    let v_slice = v_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(v_layout.start_offset()..);
    let w_slice = w_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(w_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(sin_layout.start_offset()..);
    let kc_slice = kc_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(kc_layout.start_offset()..);
    let vc_slice = vc_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(vc_layout.start_offset()..);
    let sm_slice = sm_cuda
        .as_cuda_slice::<i64>()?
        .slice(sm_layout.start_offset()..);

    let block = 256u32; // 8 warps per block
    let rows_per_block = block / 32;
    let grid = (total_kv_rows as u32 + rows_per_block - 1) / rows_per_block;

    let func = dev.get_or_load_custom_func(
        "fused_knorm_rope_kv_cache_write_thd_bf16",
        MOD_KNORM_ROPE_KV_WRITE,
        PTX_KNORM_ROPE_KV_WRITE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&k_slice);
    builder.arg(&v_slice);
    builder.arg(&w_slice);
    builder.arg(&cos_slice);
    builder.arg(&sin_slice);
    builder.arg(&kc_slice);
    builder.arg(&vc_slice);
    builder.arg(&sm_slice);
    let total_kv_rows_val = total_kv_rows as u32;
    let num_kv_heads_val = num_kv_heads as u32;
    let head_dim_val = head_dim as u32;
    let block_size_val = block_size as u32;
    let x_val = x as u32;
    let seq_len_val = seq_len as u32;
    let offset_val = offset as u32;
    let eps_val = eps as f32;
    builder.arg(&total_kv_rows_val);
    builder.arg(&num_kv_heads_val);
    builder.arg(&head_dim_val);
    builder.arg(&block_size_val);
    builder.arg(&x_val);
    builder.arg(&seq_len_val);
    builder.arg(&offset_val);
    builder.arg(&eps_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(k_storage);
    drop(v_storage);
    drop(w_storage);
    drop(cos_storage);
    drop(sin_storage);
    drop(kc_storage);
    drop(vc_storage);
    drop(sm_storage);

    Ok(())
}

/// Fused K-Norm + RoPE + KV paged cache write for varlen prefill (flash layout).
///
/// - `k`: `[total_tokens, num_kv_heads, head_dim]` BF16
/// - `v`: `[total_tokens, num_kv_heads, head_dim]` BF16
/// - `k_norm_weight`: `[head_dim]` BF16
/// - `cos_table`, `sin_table`: `[max_seq_len, head_dim/2]` BF16
/// - `pos_ids`: `[total_tokens]` U32
/// - `key_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]` BF16
/// - `value_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]` BF16
/// - `slot_mapping`: `[total_tokens]` I64
#[cfg(feature = "flash-attn-v3")]
pub fn fused_knorm_rope_kv_cache_write_varlen(
    k: &Tensor,             // [total_tokens, num_kv_heads, head_dim]
    v: &Tensor,             // [total_tokens, num_kv_heads, head_dim]
    k_norm_weight: &Tensor, // [head_dim]
    cos_table: &Tensor,     // [max_seq_len, head_dim/2]
    sin_table: &Tensor,     // [max_seq_len, head_dim/2]
    pos_ids: &Tensor,       // [total_tokens] U32
    key_cache: &Tensor,     // [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: &Tensor,   // [num_blocks, block_size, num_kv_heads, head_dim]
    slot_mapping: &Tensor,  // [total_tokens] I64
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    eps: f64,
) -> Result<()> {
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let (w_storage, w_layout) = k_norm_weight.storage_and_layout();
    let (cos_storage, cos_layout) = cos_table.storage_and_layout();
    let (sin_storage, sin_layout) = sin_table.storage_and_layout();
    let (pos_storage, pos_layout) = pos_ids.storage_and_layout();
    let (kc_storage, kc_layout) = key_cache.storage_and_layout();
    let (vc_storage, vc_layout) = value_cache.storage_and_layout();
    let (sm_storage, sm_layout) = slot_mapping.storage_and_layout();

    let k_cuda = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_varlen: k requires CUDA"),
    };
    let v_cuda = match &*v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_varlen: v requires CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_varlen: weight requires CUDA"),
    };
    let cos_cuda = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_varlen: cos requires CUDA"),
    };
    let sin_cuda = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_varlen: sin requires CUDA"),
    };
    let pos_cuda = match &*pos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_varlen: pos_ids requires CUDA"),
    };
    let kc_cuda = match &*kc_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_varlen: key_cache requires CUDA"),
    };
    let vc_cuda = match &*vc_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_varlen: value_cache requires CUDA"),
    };
    let sm_cuda = match &*sm_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_knorm_rope_kv_write_varlen: slot_mapping requires CUDA"),
    };

    if k_cuda.dtype() != DType::BF16 {
        candle_core::bail!("fused_knorm_rope_kv_write_varlen: requires BF16");
    }

    let dev = k_cuda.device().clone();
    let k_dims = k_layout.shape().dims();
    let total_tokens = k_dims[0];
    let total_kv_rows = total_tokens * num_kv_heads;

    let k_slice = k_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(k_layout.start_offset()..);
    let v_slice = v_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(v_layout.start_offset()..);
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
    let kc_slice = kc_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(kc_layout.start_offset()..);
    let vc_slice = vc_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(vc_layout.start_offset()..);
    let sm_slice = sm_cuda
        .as_cuda_slice::<i64>()?
        .slice(sm_layout.start_offset()..);

    let block = 256u32;
    let rows_per_block = block / 32;
    let grid = (total_kv_rows as u32 + rows_per_block - 1) / rows_per_block;

    let func = dev.get_or_load_custom_func(
        "fused_knorm_rope_kv_cache_write_varlen_bf16",
        MOD_KNORM_ROPE_KV_WRITE,
        PTX_KNORM_ROPE_KV_WRITE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&k_slice);
    builder.arg(&v_slice);
    builder.arg(&w_slice);
    builder.arg(&cos_slice);
    builder.arg(&sin_slice);
    builder.arg(&pos_slice);
    builder.arg(&kc_slice);
    builder.arg(&vc_slice);
    builder.arg(&sm_slice);
    let total_kv_rows_val = total_kv_rows as u32;
    let num_kv_heads_val = num_kv_heads as u32;
    let head_dim_val = head_dim as u32;
    let block_size_val = block_size as u32;
    let eps_val = eps as f32;
    builder.arg(&total_kv_rows_val);
    builder.arg(&num_kv_heads_val);
    builder.arg(&head_dim_val);
    builder.arg(&block_size_val);
    builder.arg(&eps_val);
    unsafe { builder.launch(cfg) }.w()?;

    drop(k_storage);
    drop(v_storage);
    drop(w_storage);
    drop(cos_storage);
    drop(sin_storage);
    drop(pos_storage);
    drop(kc_storage);
    drop(vc_storage);
    drop(sm_storage);

    Ok(())
}

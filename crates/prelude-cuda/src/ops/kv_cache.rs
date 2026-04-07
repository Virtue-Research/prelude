use crate::device::{self as cb, CuResultExt, LaunchConfig, PushKernelArg};
use crate::{MOD_KNORM_ROPE_KV_WRITE, PTX_KNORM_ROPE_KV_WRITE};
use crate::{MOD_SCATTER_KV_CACHE, PTX_SCATTER_KV_CACHE};
use prelude_core::tensor::{bail, DType, Result, Tensor};

// ── Fused K-Norm + RoPE + KV paged cache write ──────────────────────

/// Returns true if the fused KV cache write kernel is enabled.
pub fn fused_kv_cache_write_enabled() -> bool {
    prelude_core::config::global_runtime()
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
    let (k_storage, k_layout) = cb::storage_and_layout(&k);
    let (v_storage, v_layout) = cb::storage_and_layout(&v);
    let (w_storage, w_layout) = cb::storage_and_layout(&k_norm_weight);
    let (cos_storage, cos_layout) = cb::storage_and_layout(&cos_table);
    let (sin_storage, sin_layout) = cb::storage_and_layout(&sin_table);
    let (kc_storage, kc_layout) = cb::storage_and_layout(&key_cache);
    let (vc_storage, vc_layout) = cb::storage_and_layout(&value_cache);
    let (sm_storage, sm_layout) = cb::storage_and_layout(&slot_mapping);

    let k_cuda = cb::as_cuda(&k_storage, "fused_knorm_rope_kv_write_thd: k")?;
    let v_cuda = cb::as_cuda(&v_storage, "fused_knorm_rope_kv_write_thd: v")?;
    let w_cuda = cb::as_cuda(&w_storage, "fused_knorm_rope_kv_write_thd: weight")?;
    let cos_cuda = cb::as_cuda(&cos_storage, "fused_knorm_rope_kv_write_thd: cos")?;
    let sin_cuda = cb::as_cuda(&sin_storage, "fused_knorm_rope_kv_write_thd: sin")?;
    let kc_cuda = cb::as_cuda(&kc_storage, "fused_knorm_rope_kv_write_thd: key_cache")?;
    let vc_cuda = cb::as_cuda(&vc_storage, "fused_knorm_rope_kv_write_thd: value_cache")?;
    let sm_cuda = cb::as_cuda(&sm_storage, "fused_knorm_rope_kv_write_thd: slot_mapping")?;

    if k_cuda.dtype() != DType::BF16 {
        bail!("fused_knorm_rope_kv_write_thd: requires BF16");
    }

    let stream = k_cuda.stream.clone();
    let total_kv_rows = k_layout.shape().dims()[0];

    let k_slice = k_cuda
        .as_slice::<half::bf16>()?
        .slice(k_layout.start_offset()..);
    let v_slice = v_cuda
        .as_slice::<half::bf16>()?
        .slice(v_layout.start_offset()..);
    let w_slice = w_cuda
        .as_slice::<half::bf16>()?
        .slice(w_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_slice::<half::bf16>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_slice::<half::bf16>()?
        .slice(sin_layout.start_offset()..);
    let kc_slice = kc_cuda
        .as_slice::<half::bf16>()?
        .slice(kc_layout.start_offset()..);
    let vc_slice = vc_cuda
        .as_slice::<half::bf16>()?
        .slice(vc_layout.start_offset()..);
    let sm_slice = sm_cuda
        .as_slice::<i64>()?
        .slice(sm_layout.start_offset()..);

    let block = 256u32; // 8 warps per block
    let rows_per_block = block / 32;
    let grid = (total_kv_rows as u32 + rows_per_block - 1) / rows_per_block;

    let func = crate::device::get_or_load_func(
        k_cuda.device(),
        "fused_knorm_rope_kv_cache_write_thd_bf16",
        MOD_KNORM_ROPE_KV_WRITE,
        PTX_KNORM_ROPE_KV_WRITE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
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
    unsafe { builder.launch(cfg) }.ce()?;

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
    let (k_storage, k_layout) = cb::storage_and_layout(&k);
    let (v_storage, v_layout) = cb::storage_and_layout(&v);
    let (w_storage, w_layout) = cb::storage_and_layout(&k_norm_weight);
    let (cos_storage, cos_layout) = cb::storage_and_layout(&cos_table);
    let (sin_storage, sin_layout) = cb::storage_and_layout(&sin_table);
    let (pos_storage, pos_layout) = cb::storage_and_layout(&pos_ids);
    let (kc_storage, kc_layout) = cb::storage_and_layout(&key_cache);
    let (vc_storage, vc_layout) = cb::storage_and_layout(&value_cache);
    let (sm_storage, sm_layout) = cb::storage_and_layout(&slot_mapping);

    let k_cuda = cb::as_cuda(&k_storage, "fused_knorm_rope_kv_write_varlen: k")?;
    let v_cuda = cb::as_cuda(&v_storage, "fused_knorm_rope_kv_write_varlen: v")?;
    let w_cuda = cb::as_cuda(&w_storage, "fused_knorm_rope_kv_write_varlen: weight")?;
    let cos_cuda = cb::as_cuda(&cos_storage, "fused_knorm_rope_kv_write_varlen: cos")?;
    let sin_cuda = cb::as_cuda(&sin_storage, "fused_knorm_rope_kv_write_varlen: sin")?;
    let pos_cuda = cb::as_cuda(&pos_storage, "fused_knorm_rope_kv_write_varlen: pos_ids")?;
    let kc_cuda = cb::as_cuda(&kc_storage, "fused_knorm_rope_kv_write_varlen: key_cache")?;
    let vc_cuda = cb::as_cuda(&vc_storage, "fused_knorm_rope_kv_write_varlen: value_cache")?;
    let sm_cuda = cb::as_cuda(&sm_storage, "fused_knorm_rope_kv_write_varlen: slot_mapping")?;

    if k_cuda.dtype() != DType::BF16 {
        bail!("fused_knorm_rope_kv_write_varlen: requires BF16");
    }

    let stream = k_cuda.stream.clone();
    let k_dims = k_layout.shape().dims();
    let total_tokens = k_dims[0];
    let total_kv_rows = total_tokens * num_kv_heads;

    let k_slice = k_cuda
        .as_slice::<half::bf16>()?
        .slice(k_layout.start_offset()..);
    let v_slice = v_cuda
        .as_slice::<half::bf16>()?
        .slice(v_layout.start_offset()..);
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
    let kc_slice = kc_cuda
        .as_slice::<half::bf16>()?
        .slice(kc_layout.start_offset()..);
    let vc_slice = vc_cuda
        .as_slice::<half::bf16>()?
        .slice(vc_layout.start_offset()..);
    let sm_slice = sm_cuda
        .as_slice::<i64>()?
        .slice(sm_layout.start_offset()..);

    let block = 256u32;
    let rows_per_block = block / 32;
    let grid = (total_kv_rows as u32 + rows_per_block - 1) / rows_per_block;

    let func = crate::device::get_or_load_func(
        k_cuda.device(),
        "fused_knorm_rope_kv_cache_write_varlen_bf16",
        MOD_KNORM_ROPE_KV_WRITE,
        PTX_KNORM_ROPE_KV_WRITE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
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
    unsafe { builder.launch(cfg) }.ce()?;

    Ok(())
}

// ── Scatter KV cache write (flash layout) ────────────────────────────

/// Scatter-write K/V tokens into a flash-layout paged KV cache.
///
/// Drop-in replacement for paged attention `reshape_and_cache_flash`.
/// Uses vectorized 128-bit copies (8 BF16/thread).
///
/// - `key`: `[num_tokens, num_heads, head_size]` BF16
/// - `value`: `[num_tokens, num_heads, head_size]` BF16
/// - `key_cache`: `[num_blocks, block_size, num_heads, head_size]` BF16
/// - `value_cache`: `[num_blocks, block_size, num_heads, head_size]` BF16
/// - `slot_mapping`: `[num_tokens]` I64
pub fn scatter_kv_cache_flash(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let (k_storage, k_layout) = cb::storage_and_layout(&key);
    let (v_storage, v_layout) = cb::storage_and_layout(&value);
    let (kc_storage, kc_layout) = cb::storage_and_layout(&key_cache);
    let (vc_storage, vc_layout) = cb::storage_and_layout(&value_cache);
    let (sm_storage, sm_layout) = cb::storage_and_layout(&slot_mapping);

    let k_cuda = cb::as_cuda(&k_storage, "scatter_kv_cache_flash: key")?;
    let v_cuda = cb::as_cuda(&v_storage, "scatter_kv_cache_flash: value")?;
    let kc_cuda = cb::as_cuda(&kc_storage, "scatter_kv_cache_flash: key_cache")?;
    let vc_cuda = cb::as_cuda(&vc_storage, "scatter_kv_cache_flash: value_cache")?;
    let sm_cuda = cb::as_cuda(&sm_storage, "scatter_kv_cache_flash: slot_mapping")?;

    if k_cuda.dtype() != DType::BF16 {
        bail!("scatter_kv_cache_flash: requires BF16 (got {:?})", k_cuda.dtype());
    }

    let stream = k_cuda.stream.clone();

    let (num_tokens, num_heads, head_size) = k_layout.shape().dims3()?;
    let (_num_blocks, block_size, _num_heads_c, _head_size_c) = kc_layout.shape().dims4()?;

    let k_slice = k_cuda.as_slice::<half::bf16>()?.slice(k_layout.start_offset()..);
    let v_slice = v_cuda.as_slice::<half::bf16>()?.slice(v_layout.start_offset()..);
    let kc_slice = kc_cuda.as_slice::<half::bf16>()?.slice(kc_layout.start_offset()..);
    let vc_slice = vc_cuda.as_slice::<half::bf16>()?.slice(vc_layout.start_offset()..);
    let sm_slice = sm_cuda.as_slice::<i64>()?.slice(sm_layout.start_offset()..);

    let n: usize = num_heads * head_size;
    // Each thread handles 8 BF16 elements (128-bit vectorized).
    // For 32h×128d (n=4096): 512 threads, each copies 8 elements = done in 1 pass.
    let threads_needed = (n + 7) / 8;
    let block_dim = threads_needed.min(512) as u32;

    let func = crate::device::get_or_load_func(
        k_cuda.device(),
        "scatter_kv_cache_flash_bf16",
        MOD_SCATTER_KV_CACHE,
        PTX_SCATTER_KV_CACHE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    let key_stride = k_layout.stride()[0] as u32;
    let value_stride = v_layout.stride()[0] as u32;
    let num_heads_val = num_heads as u32;
    let head_size_val = head_size as u32;
    let block_size_val = block_size as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&k_slice);
    builder.arg(&v_slice);
    builder.arg(&kc_slice);
    builder.arg(&vc_slice);
    builder.arg(&sm_slice);
    builder.arg(&num_heads_val);
    builder.arg(&head_size_val);
    builder.arg(&block_size_val);
    builder.arg(&key_stride);
    builder.arg(&value_stride);
    unsafe { builder.launch(cfg) }.ce()?;

    Ok(())
}

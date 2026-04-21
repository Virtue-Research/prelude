//! Flash Attention v4 backend (CuTeDSL AOT, SM80+).
//!
//! FA4 kernels are statically linked into the binary — no dlopen, no runtime .so files.
//! Calls kernel functions directly through TVM FFI packed calling convention.
//!
//! Supports both non-paged varlen (prefill) and paged KV (prefill + decode).

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Result, Tensor};
use half::bf16;
use prelude_flash_attn_v4::{KernelKey, KernelRegistry};
use std::ffi::c_void;

/// Get the FA4 kernel registry (cached singleton — arch detection is one-time only).
fn get_registry() -> &'static KernelRegistry {
    use std::sync::OnceLock;
    static REGISTRY: OnceLock<KernelRegistry> = OnceLock::new();
    REGISTRY.get_or_init(KernelRegistry::new)
}

// ── Non-paged varlen attention ─────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn varlen_causal(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    call_fa4(q, k, v, cu_seqlens_q, cu_seqlens_k,
             max_seqlen_q, max_seqlen_k, softmax_scale,
             true, None, None, None)
}

#[allow(clippy::too_many_arguments)]
pub fn varlen_bidirectional(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    call_fa4(q, k, v, cu_seqlens_q, cu_seqlens_k,
             max_seqlen_q, max_seqlen_k, softmax_scale,
             false, None, None, None)
}

#[allow(clippy::too_many_arguments)]
pub fn varlen_windowed(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
    window_left: Option<usize>, window_right: Option<usize>,
) -> Result<Tensor> {
    call_fa4(q, k, v, cu_seqlens_q, cu_seqlens_k,
             max_seqlen_q, max_seqlen_k, softmax_scale,
             // causal = window_right == Some(0)
             window_right == Some(0),
             window_left.map(|v| v as i32),
             window_right.map(|v| v as i32),
             None)
}

/// Varlen causal attention with softcap (Gemma models).
#[allow(clippy::too_many_arguments)]
pub fn varlen_causal_softcap(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, max_seqlen_k: usize,
    softmax_scale: f32,
    softcap: f32,
) -> Result<Tensor> {
    call_fa4(q, k, v, cu_seqlens_q, cu_seqlens_k,
             max_seqlen_q, max_seqlen_k, softmax_scale,
             true, None, None, Some(softcap))
}

// ── Paged varlen attention ─────────────────────────────────────────────

/// Paged varlen attention: varlen Q + paged KV cache read.
///
/// Used for both prefill (variable Q lengths) and decode (Q=1 per seq).
/// key_cache/value_cache shape: `[num_blocks, block_size, num_kv_heads, head_dim]`.
/// block_tables shape: `[batch_size, max_blocks_per_seq]`, I32.
/// seqused_k: per-sequence K lengths `[batch_size]`, I32.
#[allow(clippy::too_many_arguments)]
pub fn varlen_paged(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seqlens_q: &Tensor,
    seqused_k: &Tensor,
    max_seqlen_q: usize, _max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    if q.dtype() != DType::BF16 {
        candle_core::bail!("FA4 only supports BF16 (got {:?})", q.dtype());
    }

    let registry = get_registry();

    // Q: [total_q, num_heads_q, head_dim]
    let (total_q, num_heads_q, head_dim) = q.shape().dims3()?;
    // key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    let (_num_blocks, block_size, num_heads_k, _head_dim_k) = key_cache.shape().dims4()?;
    let gqa_ratio = num_heads_q / num_heads_k;

    // TMA requires page_size == tile_n. Determine tile_n via shared helper.
    let head_dim_v = key_cache.dim(3)?;
    let tile_n = super::fa4_tile_n(head_dim, head_dim_v);
    let non_tma = block_size != tile_n;

    let key = KernelKey::new(head_dim as u32, gqa_ratio as u32, true, false)
        .with_paged(true)
        .with_paged_non_tma(non_tma);

    let func = registry.get(&key).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "FA4 paged kernel variant not found: hdim={head_dim} gqa={gqa_ratio} \
             paged=true non_tma={non_tma} block_size={block_size}"
        ))
    })?;

    let dev = match q.device() {
        candle_core::Device::Cuda(d) => d,
        _ => candle_core::bail!("FA4 requires CUDA device"),
    };
    let out = Tensor::zeros(q.shape(), DType::BF16, q.device())?;
    let stream = dev.cuda_stream();

    {
        let raw_stream = unsafe { stream.cu_stream() as *mut c_void };

        macro_rules! cuda_ptr {
            ($t:expr, $ty:ty) => {{
                let (storage, layout) = $t.storage_and_layout();
                let cuda = match &*storage {
                    candle_core::Storage::Cuda(c) => c,
                    _ => candle_core::bail!("tensor not on CUDA"),
                };
                let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
                let (ptr, _guard) = unsafe { slice.device_ptr(&stream) };
                ptr as u64
            }};
        }

        let q_ptr = cuda_ptr!(q, bf16);
        let k_ptr = cuda_ptr!(key_cache, bf16);
        let v_ptr = cuda_ptr!(value_cache, bf16);
        let o_ptr = cuda_ptr!(&out, bf16);
        let cu_q_ptr = cuda_ptr!(cu_seqlens_q, u32);
        let seqused_k_ptr = cuda_ptr!(seqused_k, u32);
        let pt_ptr = cuda_ptr!(block_tables, u32);

        let q_shape: [i64; 3] = [total_q as i64, num_heads_q as i64, head_dim as i64];
        let k_shape: [i64; 4] = [
            key_cache.dim(0)? as i64,
            key_cache.dim(1)? as i64,
            num_heads_k as i64,
            head_dim as i64,
        ];
        let o_shape = q_shape;
        let lse_shape: [i64; 2] = [num_heads_q as i64, total_q as i64];
        let cu_q_shape: [i64; 1] = [cu_seqlens_q.dim(0)? as i64];
        let seqused_k_shape: [i64; 1] = [seqused_k.dim(0)? as i64];
        let pt_shape: [i64; 2] = [
            block_tables.dim(0)? as i64,
            block_tables.dim(1)? as i64,
        ];

        let device_id = match q.device().location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id as i32,
            _ => 0,
        };

        unsafe {
            prelude_flash_attn_v4::fa4_varlen_paged_fwd(
                registry, func,
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                o_ptr as *mut c_void,
                std::ptr::null_mut(), // no LSE
                softmax_scale,
                raw_stream,
                cu_q_ptr as *mut c_void,
                seqused_k_ptr as *mut c_void,
                pt_ptr as *mut c_void,
                &q_shape, &k_shape, &o_shape, &lse_shape,
                &cu_q_shape, &seqused_k_shape, &pt_shape,
                device_id,
                None, None, // no window
            ).map_err(|e| candle_core::Error::Msg(format!("FA4 paged kernel error: {e}")))?;
        }
    }

    Ok(out)
}

// ── Core non-paged dispatch ────────────────────────────────────────────

/// Core FA4 dispatch: extract raw pointers from candle tensors and call the kernel.
#[allow(clippy::too_many_arguments)]
fn call_fa4(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    _max_seqlen_q: usize, _max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
    window_left: Option<i32>, window_right: Option<i32>,
    softcap: Option<f32>,
) -> Result<Tensor> {
    if q.dtype() != DType::BF16 {
        candle_core::bail!("FA4 only supports BF16 (got {:?})", q.dtype());
    }

    let registry = get_registry();

    // Determine kernel variant key
    // Q: [total_q, num_heads_q, head_dim], K: [total_k, num_heads_k, head_dim]
    let (total_q, num_heads_q, head_dim) = q.shape().dims3()?;
    let (_total_k, num_heads_k, _head_dim_k) = k.shape().dims3()?;
    let gqa_ratio = num_heads_q / num_heads_k;
    let has_window = window_left.is_some() || window_right.is_some();

    let key = KernelKey::new(head_dim as u32, gqa_ratio as u32, causal, has_window)
        .with_softcap(softcap);

    let func = registry.get(&key).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "FA4 kernel variant not found: hdim={head_dim} gqa={gqa_ratio} causal={causal} \
             window={has_window} softcap={softcap:?}"
        ))
    })?;

    // Allocate output tensor (same shape as Q)
    let dev = match q.device() {
        candle_core::Device::Cuda(d) => d,
        _ => candle_core::bail!("FA4 requires CUDA device"),
    };
    let out = Tensor::zeros(q.shape(), DType::BF16, q.device())?;

    // Extract raw device pointers.
    // Scope borrows so `out` can be moved into Ok() at the end.
    let stream = dev.cuda_stream();

    {
        let raw_stream = unsafe { stream.cu_stream() as *mut c_void };

        macro_rules! cuda_ptr {
            ($t:expr, $ty:ty) => {{
                let (storage, layout) = $t.storage_and_layout();
                let cuda = match &*storage {
                    candle_core::Storage::Cuda(c) => c,
                    _ => candle_core::bail!("tensor not on CUDA"),
                };
                let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
                let (ptr, _guard) = unsafe { slice.device_ptr(&stream) };
                ptr as u64
            }};
        }

        let q_ptr = cuda_ptr!(q, bf16);
        let k_ptr = cuda_ptr!(k, bf16);
        let v_ptr = cuda_ptr!(v, bf16);
        let o_ptr = cuda_ptr!(&out, bf16);
        let cu_q_ptr = cuda_ptr!(cu_seqlens_q, u32);
        let cu_k_ptr = cuda_ptr!(cu_seqlens_k, u32);

        let (tq, hq, hd) = q.shape().dims3()?;
        let (tk, hk, _) = k.shape().dims3()?;
        let q_shape: [i64; 3] = [tq as i64, hq as i64, hd as i64];
        let k_shape: [i64; 3] = [tk as i64, hk as i64, hd as i64];
        let o_shape = q_shape;
        let lse_shape: [i64; 2] = [num_heads_q as i64, total_q as i64];
        let cu_shape: [i64; 1] = [cu_seqlens_q.dim(0)? as i64];

        let device_id = match q.device().location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id as i32,
            _ => 0,
        };

        unsafe {
            prelude_flash_attn_v4::fa4_varlen_fwd(
                registry, func,
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                o_ptr as *mut c_void,
                std::ptr::null_mut(), // no LSE
                softmax_scale,
                raw_stream,
                cu_q_ptr as *mut c_void,
                cu_k_ptr as *mut c_void,
                &q_shape, &k_shape, &o_shape, &lse_shape, &cu_shape,
                device_id,
                window_left, window_right,
                None, None, // no seqused
            ).map_err(|e| candle_core::Error::Msg(format!("FA4 kernel error: {e}")))?;
        }
    }

    Ok(out)
}

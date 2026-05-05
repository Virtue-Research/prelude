//! Flash Attention v4 backend (CuTeDSL AOT, SM80+).
//!
//! FA4 kernels are statically linked into the binary — no dlopen, no runtime .so files.
//! Calls kernel functions directly through TVM FFI packed calling convention.
//!
//! Supports both non-paged varlen (prefill) and paged KV (prefill + decode).

use crate::device::{self as cb};
use cudarc::driver::DevicePtr;
use flash_attn_v4::{KernelDtype, KernelKey, KernelRegistry};
use half::bf16;
use prelude_core::tensor::{DType, DeviceExt, Result, Tensor};
use std::ffi::c_void;

fn to_kernel_dtype(dt: DType) -> KernelDtype {
    match dt {
        DType::BF16 => KernelDtype::BF16,
        DType::F16 => KernelDtype::FP16,
        _ => KernelDtype::BF16,
    }
}

/// Get the FA4 kernel registry (cached singleton — arch detection is one-time only).
fn get_registry() -> &'static KernelRegistry {
    use std::sync::OnceLock;
    static REGISTRY: OnceLock<KernelRegistry> = OnceLock::new();
    REGISTRY.get_or_init(KernelRegistry::new)
}

// ── Non-paged varlen attention ─────────────────────────────────────────
// All return Option<Result<Tensor>>: None = FA4 has no kernel for this combo.

#[allow(clippy::too_many_arguments)]
pub fn try_varlen_causal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
) -> Option<Result<Tensor>> {
    try_call_fa4(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        true,
        None,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn try_varlen_bidirectional(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
) -> Option<Result<Tensor>> {
    try_call_fa4(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        false,
        None,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn try_varlen_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_left: Option<usize>,
    window_right: Option<usize>,
) -> Option<Result<Tensor>> {
    try_call_fa4(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        window_right == Some(0),
        window_left.map(|v| v as i32),
        window_right.map(|v| v as i32),
        None,
    )
}

// ── Paged varlen attention ─────────────────────────────────────────────

/// Paged varlen attention: varlen Q + paged KV cache read.
///
/// Used for both prefill (variable Q lengths) and decode (Q=1 per seq).
/// key_cache/value_cache shape: `[num_blocks, block_size, num_kv_heads, head_dim]`.
/// block_tables shape: `[batch_size, max_blocks_per_seq]`, U32.
/// seqused_k: per-sequence K lengths `[batch_size]`, U32.
#[allow(clippy::too_many_arguments)]
pub fn try_varlen_paged(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seqlens_q: &Tensor,
    seqused_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    softmax_scale: f32,
) -> Option<Result<Tensor>> {
    if q.dtype() != DType::BF16 {
        return None; // FA4 only supports BF16
    }

    let registry = get_registry();

    let (total_q, num_heads_q, head_dim) = match q.shape().dims3() {
        Ok(d) => d,
        Err(e) => return Some(Err(e)),
    };
    let (_num_blocks, block_size, num_heads_k, _head_dim_k) = match key_cache.shape().dims4() {
        Ok(d) => d,
        Err(e) => return Some(Err(e)),
    };
    let gqa_ratio = num_heads_q / num_heads_k;

    let head_dim_v = match key_cache.dim(3) {
        Ok(d) => d,
        Err(e) => return Some(Err(e)),
    };
    let tile_n = super::fa4_tile_n(head_dim, head_dim_v);
    let non_tma = block_size != tile_n;

    let key = KernelKey::new(head_dim as u32, gqa_ratio as u32, true, false)
        .with_paged(true)
        .with_paged_non_tma(non_tma);

    let func = registry.get(&key)?; // None = no kernel for this combo

    // FA4 is committed — errors from here are real errors, not "not available".
    Some(varlen_paged_inner(
        registry,
        func,
        q,
        key_cache,
        value_cache,
        block_tables,
        cu_seqlens_q,
        seqused_k,
        softmax_scale,
        total_q,
        num_heads_q,
        head_dim,
        num_heads_k,
    ))
}

/// Inner paged FA4 call — always returns Result (no more Option branching).
#[allow(clippy::too_many_arguments)]
fn varlen_paged_inner(
    registry: &KernelRegistry,
    func: flash_attn_v4::TVMSafeCallFn,
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seqlens_q: &Tensor,
    seqused_k: &Tensor,
    softmax_scale: f32,
    total_q: usize,
    num_heads_q: usize,
    head_dim: usize,
    num_heads_k: usize,
) -> Result<Tensor> {
    let stream = cb::tensor_stream(q)?;
    let out = Tensor::zeros(q.shape(), DType::BF16, q.device())?;

    {
        let raw_stream = stream.cu_stream() as *mut c_void;

        macro_rules! cuda_ptr {
            ($t:expr, $ty:ty) => {{
                let (storage, layout) = $t.storage_and_layout();
                let cuda = match &*storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => candle_core::bail!("FA4: requires CUDA"),
                };
                let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
                let (ptr, _guard) = slice.device_ptr(&stream);
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
        let q_strides: Vec<i64> = q.layout().stride().iter().map(|&s| s as i64).collect();
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
        let pt_shape: [i64; 2] = [block_tables.dim(0)? as i64, block_tables.dim(1)? as i64];

        let device_id = q.device().ordinal() as i32;

        unsafe {
            flash_attn_v4::fa4_varlen_paged_fwd(
                registry,
                func,
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
                &q_shape,
                &q_strides,
                &k_shape,
                &k_shape,
                &o_shape,
                &lse_shape,
                &cu_q_shape,
                &seqused_k_shape,
                &pt_shape,
                device_id,
                None,
                None, // no window
                to_kernel_dtype(q.dtype()),
            )
            .map_err(|e| {
                prelude_core::tensor::Error::Msg(format!("FA4 paged kernel error: {e}"))
            })?;
        }
    }

    Ok(out)
}

// ── Core non-paged dispatch ────────────────────────────────────────────

/// Core FA4 dispatch. Returns None if no kernel variant exists for this combo.
#[allow(clippy::too_many_arguments)]
fn try_call_fa4(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
    window_left: Option<i32>,
    window_right: Option<i32>,
    softcap: Option<f32>,
) -> Option<Result<Tensor>> {
    if q.dtype() != DType::BF16 {
        return None;
    }

    let registry = get_registry();
    let (total_q, num_heads_q, head_dim) = match q.shape().dims3() {
        Ok(d) => d,
        Err(e) => return Some(Err(e)),
    };
    let (_total_k, num_heads_k, _head_dim_k) = match k.shape().dims3() {
        Ok(d) => d,
        Err(e) => return Some(Err(e)),
    };
    let gqa_ratio = num_heads_q / num_heads_k;
    let has_window = window_left.is_some() || window_right.is_some();

    // V-split for head_dim > 256: SM90 MMA limits N ≤ 256, so we compile
    // kernels with (head_dim, head_dim_v=256) and call twice with V split in half.
    if head_dim > 256 {
        let half = head_dim / 2;
        let mut key = KernelKey::new(head_dim as u32, gqa_ratio as u32, causal, has_window);
        key.head_dim_v = half as u32;
        key = key.with_softcap(softcap);
        let func = match registry.get(&key) {
            Some(f) => f,
            None => return None,
        };
        // Committed — V-split path.
        return Some(call_fa4_vsplit(
            registry,
            func,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
            window_left,
            window_right,
            total_q,
            num_heads_q,
            head_dim,
            num_heads_k,
            half,
        ));
    }

    let key =
        KernelKey::new(head_dim as u32, gqa_ratio as u32, causal, has_window).with_softcap(softcap);
    let func = registry.get(&key)?;

    // From here on, FA4 is committed — errors are real errors, not "not available".
    Some(call_fa4_inner(
        registry,
        func,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        window_left,
        window_right,
        total_q,
        num_heads_q,
        head_dim,
        num_heads_k,
    ))
}

/// V-split FA4: split V along head_dim, call kernel twice, concat outputs.
/// Q@K^T uses full head_dim as reduction dimension (SM90 MMA iterates over K).
/// P@V uses half head_dim as output dimension (within MMA N≤256 limit).
#[allow(clippy::too_many_arguments)]
fn call_fa4_vsplit(
    registry: &KernelRegistry,
    func: flash_attn_v4::TVMSafeCallFn,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
    window_left: Option<i32>,
    window_right: Option<i32>,
    total_q: usize,
    num_heads_q: usize,
    head_dim: usize,
    num_heads_k: usize,
    half_dim: usize,
) -> Result<Tensor> {
    let _num_kv_heads = v.dim(1)?;
    // V: [total_k, num_kv_heads, head_dim] → split last dim.
    // Narrow preserves stride(-1)=1, and FA4 is stride-aware, so no contiguous needed.
    let v_lo = v.narrow(2, 0, half_dim)?;
    let v_hi = v.narrow(2, half_dim, half_dim)?;

    let o_lo = call_fa4_inner_vsplit(
        registry,
        func,
        q,
        k,
        &v_lo,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        window_left,
        window_right,
        total_q,
        num_heads_q,
        head_dim,
        num_heads_k,
        half_dim,
    )?;
    let o_hi = call_fa4_inner_vsplit(
        registry,
        func,
        q,
        k,
        &v_hi,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        window_left,
        window_right,
        total_q,
        num_heads_q,
        head_dim,
        num_heads_k,
        half_dim,
    )?;

    // Concat along last dim: [total_q, num_heads_q, half] × 2 → [total_q, num_heads_q, head_dim]
    Tensor::cat(&[&o_lo, &o_hi], 2)
}

/// Inner FA4 call for V-split: Q/K have head_dim, V/O have half_dim.
#[allow(clippy::too_many_arguments)]
fn call_fa4_inner_vsplit(
    registry: &KernelRegistry,
    func: flash_attn_v4::TVMSafeCallFn,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
    window_left: Option<i32>,
    window_right: Option<i32>,
    total_q: usize,
    num_heads_q: usize,
    head_dim: usize,
    _num_heads_k: usize,
    head_dim_v: usize,
) -> Result<Tensor> {
    let stream = cb::tensor_stream(q)?;
    // Output has head_dim_v, not head_dim
    let out = Tensor::zeros(&[total_q, num_heads_q, head_dim_v], DType::BF16, q.device())?;

    {
        let raw_stream = stream.cu_stream() as *mut c_void;

        macro_rules! cuda_ptr {
            ($t:expr, $ty:ty) => {{
                let (storage, layout) = $t.storage_and_layout();
                let cuda = match &*storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => candle_core::bail!("FA4: requires CUDA"),
                };
                let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
                let (ptr, _guard) = slice.device_ptr(&stream);
                ptr as u64
            }};
        }

        let q_ptr = cuda_ptr!(q, bf16);
        let k_ptr = cuda_ptr!(k, bf16);
        let v_ptr = cuda_ptr!(v, bf16);
        let o_ptr = cuda_ptr!(&out, bf16);
        let cu_q_ptr = cuda_ptr!(cu_seqlens_q, u32);
        let cu_k_ptr = cuda_ptr!(cu_seqlens_k, u32);

        let (tq, hq, _) = q.shape().dims3()?;
        let (tk, hk, _) = k.shape().dims3()?;
        // Q/K shapes use full head_dim; V/O shapes use head_dim_v
        let q_shape: [i64; 3] = [tq as i64, hq as i64, head_dim as i64];
        let k_shape: [i64; 3] = [tk as i64, hk as i64, head_dim as i64];
        let v_shape: [i64; 3] = [tk as i64, hk as i64, head_dim_v as i64];
        let o_shape: [i64; 3] = [total_q as i64, num_heads_q as i64, head_dim_v as i64];
        let lse_shape: [i64; 2] = [num_heads_q as i64, total_q as i64];
        let cu_shape: [i64; 1] = [cu_seqlens_q.dim(0)? as i64];

        let q_strides: Vec<i64> = q.layout().stride().iter().map(|&s| s as i64).collect();
        let k_strides: Vec<i64> = k.layout().stride().iter().map(|&s| s as i64).collect();
        let v_strides: Vec<i64> = v.layout().stride().iter().map(|&s| s as i64).collect();

        let device_id = q.device().ordinal() as i32;

        unsafe {
            flash_attn_v4::fa4_varlen_fwd(
                registry,
                func,
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                o_ptr as *mut c_void,
                std::ptr::null_mut(), // no LSE
                softmax_scale,
                raw_stream,
                cu_q_ptr as *mut c_void,
                cu_k_ptr as *mut c_void,
                &q_shape,
                &q_strides,
                &k_shape,
                &k_strides,
                &v_shape,
                &v_strides,
                &o_shape,
                &lse_shape,
                &cu_shape,
                device_id,
                window_left,
                window_right,
                None,
                None, // no seqused
                to_kernel_dtype(q.dtype()),
            )
            .map_err(|e| {
                prelude_core::tensor::Error::Msg(format!("FA4 V-split kernel error: {e}"))
            })?;
        }
    }

    Ok(out)
}

/// Inner FA4 call — always returns Result (no more Option branching).
#[allow(clippy::too_many_arguments)]
fn call_fa4_inner(
    registry: &KernelRegistry,
    func: flash_attn_v4::TVMSafeCallFn,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
    window_left: Option<i32>,
    window_right: Option<i32>,
    total_q: usize,
    num_heads_q: usize,
    _head_dim: usize,
    _num_heads_k: usize,
) -> Result<Tensor> {
    let stream = cb::tensor_stream(q)?;
    let out = Tensor::zeros(q.shape(), DType::BF16, q.device())?;

    // Extract raw device pointers.
    // Scope borrows so `out` can be moved into Ok() at the end.

    {
        let raw_stream = stream.cu_stream() as *mut c_void;

        macro_rules! cuda_ptr {
            ($t:expr, $ty:ty) => {{
                let (storage, layout) = $t.storage_and_layout();
                let cuda = match &*storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => candle_core::bail!("FA4: requires CUDA"),
                };
                let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
                let (ptr, _guard) = slice.device_ptr(&stream);
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
        let v_shape = k_shape; // head_dim_v == head_dim for non-vsplit
        let o_shape = q_shape;
        let lse_shape: [i64; 2] = [num_heads_q as i64, total_q as i64];
        let cu_shape: [i64; 1] = [cu_seqlens_q.dim(0)? as i64];

        let q_strides: Vec<i64> = q.layout().stride().iter().map(|&s| s as i64).collect();
        let k_strides: Vec<i64> = k.layout().stride().iter().map(|&s| s as i64).collect();
        let v_strides: Vec<i64> = v.layout().stride().iter().map(|&s| s as i64).collect();

        let device_id = q.device().ordinal() as i32;

        unsafe {
            flash_attn_v4::fa4_varlen_fwd(
                registry,
                func,
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                o_ptr as *mut c_void,
                std::ptr::null_mut(), // no LSE
                softmax_scale,
                raw_stream,
                cu_q_ptr as *mut c_void,
                cu_k_ptr as *mut c_void,
                &q_shape,
                &q_strides,
                &k_shape,
                &k_strides,
                &v_shape,
                &v_strides,
                &o_shape,
                &lse_shape,
                &cu_shape,
                device_id,
                window_left,
                window_right,
                None,
                None, // no seqused
                to_kernel_dtype(q.dtype()),
            )
            .map_err(|e| prelude_core::tensor::Error::Msg(format!("FA4 kernel error: {e}")))?;
        }
    }

    Ok(out)
}

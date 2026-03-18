//! Raw CPU BF16 forward path — bypasses candle Tensor for zero-overhead inference.
//!
//! All operations work on `&[u16]` / `*mut u16` slices (BF16 as u16).
//! Thread-local scratch buffers are pre-allocated and reused across layers.
//! The only Tensor operations happen at the boundary: extract input, wrap output.

use std::cell::UnsafeCell;

use candle_core::{Device, Result, Tensor};

use crate::ops::cpu::buf_tensor::CpuTensor;
use crate::ops::onednn::BrgemmPackedWeight;

// ── Unified thread-local scratch buffers ───────────────────────────────

/// Pre-allocated scratch buffers for raw CPU BF16 forward path.
/// Grow-only: buffers are never shrunk, eliminating per-layer allocations.
pub(crate) struct RawScratch {
    // Attention buffers
    pub qkv: Vec<u16>,
    pub q: Vec<u16>,         // q_scratch (deinterleave output) then reused
    pub k: Vec<u16>,         // k_scratch (deinterleave output)
    pub q_normed: Vec<u16>,  // separate from qkv — avoids cross-thread data race
    pub k_normed: Vec<u16>,  // separate from q — avoids cross-thread data race
    pub v: Vec<u16>,
    pub attn_out: Vec<u16>,
    pub proj_out: Vec<u16>,
    // MLP buffers
    pub gate_up: Vec<u16>,
    pub silu: Vec<u16>,
    pub mlp_out: Vec<u16>,
}

impl RawScratch {
    fn new() -> Self {
        Self {
            qkv: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            q_normed: Vec::new(),
            k_normed: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            proj_out: Vec::new(),
            gate_up: Vec::new(),
            silu: Vec::new(),
            mlp_out: Vec::new(),
        }
    }
}

thread_local! {
    static SCRATCH: UnsafeCell<RawScratch> = UnsafeCell::new(RawScratch::new());
}

/// Access the thread-local scratch buffers.
pub(crate) fn with_scratch<F, R>(f: F) -> R
where
    F: FnOnce(&mut RawScratch) -> R,
{
    SCRATCH.with(|c| f(unsafe { &mut *c.get() }))
}

/// Ensure a buffer has at least `needed` elements (grow-only).
#[inline]
pub(crate) fn ensure_len(buf: &mut Vec<u16>, needed: usize) {
    if buf.len() < needed {
        buf.resize(needed, 0);
    }
}

// ── Tensor ↔ raw boundary helpers ──────────────────────────────────────

/// Extract contiguous `&[u16]` from a CPU BF16 Tensor.
///
/// The caller must use `tensor_as_u16_slice_pub` or do manual storage extraction
/// because the storage guard lifetime is tied to the tensor.
/// This helper is for documentation — actual extraction is done inline at call sites.
pub(crate) fn extract_u16_slice(tensor: &Tensor) -> Result<&[u16]> {
    crate::ops::cpu::tensor_as_u16_slice_pub(tensor)
}

/// Extract position_ids as Vec<i64> (one allocation, reusable across layers).
pub(crate) fn extract_positions(pos_ids: &Tensor) -> Result<Vec<i64>> {
    pos_ids
        .to_dtype(candle_core::DType::I64)?
        .to_vec1::<i64>()
}

/// Extract seq_lens from cu_seqlens tensor.
pub(crate) fn extract_seq_lens(cu_seqlens: &Tensor) -> Result<Vec<usize>> {
    let cu: Vec<u32> = cu_seqlens.to_vec1()?;
    Ok(cu.windows(2).map(|w| (w[1] - w[0]) as usize).collect())
}

/// Wrap raw `&[u16]` data into a new BF16 Tensor (copies data).
pub(crate) fn wrap_output(data: &[u16], shape: &[usize], device: &Device) -> Result<Tensor> {
    let len = data.len();
    let result_vec: Vec<half::bf16> = unsafe {
        let mut v = Vec::with_capacity(len);
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const half::bf16,
            v.as_mut_ptr(),
            len,
        );
        v.set_len(len);
        v
    };
    Tensor::from_vec(result_vec, shape, device)
}

// ── Raw residual add ───────────────────────────────────────────────────

/// In-place BF16 add: dst[i] += src[i], with round-to-nearest-even.
/// Dispatched via GemmPool for parallelism.
pub(crate) unsafe fn raw_residual_add_bf16(dst: *mut u16, src: *const u16, len: usize) {
    #[repr(C)]
    struct AddCtx {
        dst: usize,
        src: usize,
        len: usize,
    }
    unsafe fn add_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let c = &*(ctx_raw as *const AddCtx);
            let per = (c.len + n_threads - 1) / n_threads;
            let start = tid * per;
            let end = (start + per).min(c.len);
            if start >= end {
                return;
            }
            let dst = (c.dst as *mut u16).add(start);
            let src = (c.src as *const u16).add(start);
            for i in 0..(end - start) {
                let a = f32::from_bits((*dst.add(i) as u32) << 16);
                let b = f32::from_bits((*src.add(i) as u32) << 16);
                let sum = a + b;
                let bits = sum.to_bits();
                let lsb = (bits >> 16) & 1;
                *dst.add(i) = (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16;
            }
        }
    }
    let ctx = AddCtx {
        dst: dst as usize,
        src: src as usize,
        len,
    };
    let pool = crate::ops::cpu::gemm_pool::gemm_pool();
    let n_threads = pool.num_threads().min(len / 64).max(1);
    unsafe {
        pool.dispatch(
            add_work,
            &ctx as *const AddCtx as *const u8,
            n_threads,
        );
    }
}

// ── Raw MLP forward ────────────────────────────────────────────────────

/// Raw MLP forward: gate_up GEMM → SiLU×Mul → down GEMM.
/// All on raw `*const u16` / `*mut u16`, no Tensor allocations.
///
/// - `input`: `[total, hidden_size]` BF16 via CpuTensor
/// - `output`: pre-allocated `[total * hidden_size]` u16 buffer
///
/// # Safety
/// - `output` must point to `[total * hidden_size]` pre-allocated elements.
/// - All brgemm weights must be valid.
pub(crate) unsafe fn raw_mlp_forward(
    scratch: &mut RawScratch,
    input: &CpuTensor,
    gate_up_brg: &BrgemmPackedWeight,
    down_brg: &BrgemmPackedWeight,
    output: *mut u16,
) {
    unsafe {
        let total = input.dim(0);
        let hidden_size = input.dim(1);
        let dim = gate_up_brg.n / 2; // intermediate_size
        let gate_up_len = total * gate_up_brg.n;
        let silu_len = total * dim;
        ensure_len(&mut scratch.gate_up, gate_up_len);
        ensure_len(&mut scratch.silu, silu_len);

        // Fused path (M ≤ 128): gate_up GEMM + SiLU in one pass
        if total <= 128 && gate_up_brg.n % 2 == 0 {
            crate::ops::onednn::brgemm_fused_silu_mul_raw(
                input.as_ptr(),
                gate_up_brg,
                scratch.silu.as_mut_ptr(),
                total,
                dim,
            );
        } else {
            // Unfused path: gate_up GEMM → separate SiLU → down GEMM
            crate::ops::onednn::brgemm_gemm_raw(
                input.as_ptr(),
                gate_up_brg,
                scratch.gate_up.as_mut_ptr(),
                total,
                gate_up_brg.n,
            );
            crate::ops::cpu::silu_mul::silu_and_mul_bf16(
                &mut scratch.silu[..silu_len],
                &scratch.gate_up[..gate_up_len],
                total,
                dim,
            );
        }

        // down GEMM: [total, intermediate] → [total, hidden]
        crate::ops::onednn::brgemm_gemm_raw(
            scratch.silu.as_ptr(),
            down_brg,
            output,
            total,
            hidden_size,
        );
    }
}

// ── Raw fused deinterleave + QK-norm + RoPE ────────────────────────────

/// Fused QKV deinterleave + QK RMSNorm + RoPE, dispatched via GemmPool.
///
/// Input: `qkv_buf` = [total, qkv_n] interleaved Q|K|V from GEMM output.
/// Output: q_normed in `qkv_buf[..total*q_size]`, k_normed in `q_buf[..total*kv_size]`,
///         V in `v_buf[..total*kv_size]`.
///
/// Buffer aliasing plan (per thread's token range):
///   1. Deinterleave QKV → Q into q_buf, K into k_buf, V into v_buf
///   2. RMSNorm Q: q_buf → qkv_buf (safe: qkv already consumed by step 1)
///   3. RMSNorm K: k_buf → q_buf (safe: q_buf consumed by step 2)
///   4. RoPE in-place on qkv_buf (q_normed) and q_buf (k_normed)
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn raw_fused_deinterleave_norm_rope(
    qkv_buf: *mut u16,
    q_buf: *mut u16,
    k_buf: *mut u16,
    q_normed_buf: *mut u16,
    k_normed_buf: *mut u16,
    v_buf: *mut u16,
    q_norm_w: &[u16],
    k_norm_w: &[u16],
    cos_sin_cache: &[u16],
    positions: &[i64],
    total: usize,
    q_size: usize,
    kv_size: usize,
    qkv_n: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
) {
    let embed_dim = rotary_dim / 2;

    #[repr(C)]
    struct FusedCtx {
        qkv_ptr: usize,
        q_scratch_ptr: usize,
        k_scratch_ptr: usize,
        v_out_ptr: usize,
        q_normed_ptr: usize,
        k_normed_ptr: usize,
        q_norm_w_ptr: usize,
        k_norm_w_ptr: usize,
        cache_ptr: usize,
        pos_ptr: usize,
        total: usize,
        q_size: usize,
        kv_size: usize,
        qkv_n: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        embed_dim: usize,
        rotary_dim: usize,
        eps: f32,
    }

    unsafe fn fused_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let ctx = &*(ctx_raw as *const FusedCtx);
            let rows_per = (ctx.total + n_threads - 1) / n_threads;
            let t_start = tid * rows_per;
            let t_end = (t_start + rows_per).min(ctx.total);
            if t_start >= t_end {
                return;
            }
            let chunk = t_end - t_start;
            let hd = ctx.head_dim;

            let qkv = ctx.qkv_ptr as *const u16;
            let q_scratch = ctx.q_scratch_ptr as *mut u16;
            let k_scratch = ctx.k_scratch_ptr as *mut u16;
            let v_out = ctx.v_out_ptr as *mut u16;
            let q_normed = ctx.q_normed_ptr as *mut u16;
            let k_normed = ctx.k_normed_ptr as *mut u16;
            let q_norm_w = std::slice::from_raw_parts(ctx.q_norm_w_ptr as *const u16, hd);
            let k_norm_w = std::slice::from_raw_parts(ctx.k_norm_w_ptr as *const u16, hd);
            let positions = std::slice::from_raw_parts(ctx.pos_ptr as *const i64, ctx.total);

            // Step 1: Deinterleave QKV → q_scratch, k_scratch, v_out
            for t in t_start..t_end {
                let src = qkv.add(t * ctx.qkv_n);
                std::ptr::copy_nonoverlapping(src, q_scratch.add(t * ctx.q_size), ctx.q_size);
                std::ptr::copy_nonoverlapping(
                    src.add(ctx.q_size),
                    k_scratch.add(t * ctx.kv_size),
                    ctx.kv_size,
                );
                std::ptr::copy_nonoverlapping(
                    src.add(ctx.q_size + ctx.kv_size),
                    v_out.add(t * ctx.kv_size),
                    ctx.kv_size,
                );
            }

            // Step 2: RMSNorm Q: q_scratch → q_normed
            let q_in =
                std::slice::from_raw_parts(q_scratch.add(t_start * ctx.q_size), chunk * ctx.q_size);
            let q_out = std::slice::from_raw_parts_mut(
                q_normed.add(t_start * ctx.q_size),
                chunk * ctx.q_size,
            );
            crate::ops::cpu::rmsnorm::rmsnorm_impl(
                q_out,
                q_in,
                q_norm_w,
                chunk * ctx.num_heads,
                hd,
                ctx.eps,
            );

            // Step 3: RMSNorm K: k_scratch → k_normed
            let k_in = std::slice::from_raw_parts(
                k_scratch.add(t_start * ctx.kv_size),
                chunk * ctx.kv_size,
            );
            let k_out_s = std::slice::from_raw_parts_mut(
                k_normed.add(t_start * ctx.kv_size),
                chunk * ctx.kv_size,
            );
            crate::ops::cpu::rmsnorm::rmsnorm_impl(
                k_out_s,
                k_in,
                k_norm_w,
                chunk * ctx.num_kv_heads,
                hd,
                ctx.eps,
            );

            // Step 4: RoPE in-place on q_normed and k_normed
            for t in t_start..t_end {
                let pos = positions[t];
                if pos < 0 {
                    continue;
                }
                let cache_off = pos as usize * ctx.rotary_dim;
                let cache = std::slice::from_raw_parts(
                    ctx.cache_ptr as *const u16,
                    (pos as usize + 1) * ctx.rotary_dim,
                );
                let q_tok =
                    std::slice::from_raw_parts_mut(q_normed.add(t * ctx.q_size), ctx.q_size);
                let k_tok =
                    std::slice::from_raw_parts_mut(k_normed.add(t * ctx.kv_size), ctx.kv_size);
                for h in 0..ctx.num_heads {
                    let off = h * hd;
                    crate::ops::cpu::rope::rope_neox_row(
                        &mut q_tok[off..off + hd],
                        cache,
                        cache_off,
                        ctx.embed_dim,
                    );
                }
                for h in 0..ctx.num_kv_heads {
                    let off = h * hd;
                    crate::ops::cpu::rope::rope_neox_row(
                        &mut k_tok[off..off + hd],
                        cache,
                        cache_off,
                        ctx.embed_dim,
                    );
                }
            }
        }
    }

    let fctx = FusedCtx {
        qkv_ptr: qkv_buf as usize,
        q_scratch_ptr: q_buf as usize,
        k_scratch_ptr: k_buf as usize,
        v_out_ptr: v_buf as usize,
        q_normed_ptr: q_normed_buf as usize, // separate buffer (was qkv_buf — cross-thread data race)
        k_normed_ptr: k_normed_buf as usize, // separate buffer (was q_buf — cross-thread data race)
        q_norm_w_ptr: q_norm_w.as_ptr() as usize,
        k_norm_w_ptr: k_norm_w.as_ptr() as usize,
        cache_ptr: cos_sin_cache.as_ptr() as usize,
        pos_ptr: positions.as_ptr() as usize,
        total,
        q_size,
        kv_size,
        qkv_n,
        num_heads,
        num_kv_heads,
        head_dim,
        embed_dim,
        rotary_dim,
        eps,
    };

    let pool = crate::ops::cpu::gemm_pool::gemm_pool();
    let n_threads = pool.num_threads().min(total);
    unsafe {
        pool.dispatch(
            fused_work,
            &fctx as *const FusedCtx as *const u8,
            n_threads,
        );
    }
}

// ── Raw attention forward (full pipeline) ──────────────────────────────

/// Full raw attention: QKV GEMM → fused deinterleave+norm+rope → attention → O_proj GEMM.
///
/// Returns a `CpuTensor` borrowing `scratch.proj_out` with shape `[total, input_dim]`.
///
/// Shape info derived from CpuTensor params:
/// - `total = x.dim(0)`, `input_dim = x.dim(1)`
/// - `head_dim = q_norm_w.dim(0)`, `rotary_dim = cos_sin_cache.dim(1)`
///
/// # Safety
/// - All brgemm weights must be valid.
pub(crate) unsafe fn raw_attention_forward<'a>(
    scratch: &'a mut RawScratch,
    x: &CpuTensor,
    qkv_brg: &BrgemmPackedWeight,
    oproj_brg: &BrgemmPackedWeight,
    q_norm_w: &CpuTensor,
    k_norm_w: &CpuTensor,
    cos_sin_cache: &CpuTensor,
    positions: &[i64],
    seq_lens: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    eps: f32,
    softmax_scale: f32,
) -> CpuTensor<'a> {
    unsafe {
        let total = x.dim(0);
        let input_dim = x.dim(1);
        let head_dim = q_norm_w.dim(0);
        let rotary_dim = cos_sin_cache.dim(1);

        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let qkv_n = q_size + 2 * kv_size;

        // Ensure scratch buffers
        ensure_len(&mut scratch.qkv, total * qkv_n);
        ensure_len(&mut scratch.q, total * q_size);
        ensure_len(&mut scratch.k, total * kv_size);
        ensure_len(&mut scratch.v, total * kv_size);
        ensure_len(&mut scratch.q_normed, total * q_size);
        ensure_len(&mut scratch.k_normed, total * kv_size);
        ensure_len(&mut scratch.attn_out, total * q_size);
        ensure_len(&mut scratch.proj_out, total * input_dim);

        // 1. QKV GEMM
        crate::ops::onednn::brgemm_gemm_raw(
            x.as_ptr(),
            qkv_brg,
            scratch.qkv.as_mut_ptr(),
            total,
            qkv_n,
        );

        // 2-4. Fused deinterleave + QK-norm + RoPE
        // Note: k_normed gets its own buffer to avoid data race with q_scratch (both were q_buf).
        raw_fused_deinterleave_norm_rope(
            scratch.qkv.as_mut_ptr(),
            scratch.q.as_mut_ptr(),
            scratch.k.as_mut_ptr(),
            scratch.q_normed.as_mut_ptr(),
            scratch.k_normed.as_mut_ptr(),
            scratch.v.as_mut_ptr(),
            q_norm_w.as_slice(),
            k_norm_w.as_slice(),
            cos_sin_cache.as_slice(),
            positions,
            total,
            q_size,
            kv_size,
            qkv_n,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            eps,
        );

        // 5. Attention
        // After fused dispatch: q_normed in q_normed[..], k_normed in k_normed[..], v in v[..]
        let q_normed = &scratch.q_normed[..total * q_size];
        let k_normed = &scratch.k_normed[..total * kv_size];
        let v_final = &scratch.v[..total * kv_size];
        crate::ops::cpu::attention::prefill_attention_bf16(
            &mut scratch.attn_out[..total * q_size],
            q_normed,
            k_normed,
            v_final,
            seq_lens,
            num_heads,
            num_kv_heads,
            head_dim,
            softmax_scale,
        );

        // 6. O_proj GEMM
        crate::ops::onednn::brgemm_gemm_raw(
            scratch.attn_out.as_ptr(),
            oproj_brg,
            scratch.proj_out.as_mut_ptr(),
            total,
            input_dim,
        );

        CpuTensor::from_slice(&scratch.proj_out[..total * input_dim], &[total, input_dim])
    }
}

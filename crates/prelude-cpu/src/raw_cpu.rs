//! Raw CPU BF16 forward path — bypasses Tensor abstraction for zero-overhead inference.
//!
//! All operations work on `&[u16]` / `*mut u16` slices (BF16 as u16).
//! Thread-local scratch buffers are pre-allocated and reused across layers.
//! The only Tensor operations happen at the boundary: extract input, wrap output.

use std::cell::RefCell;

use prelude_core::tensor::{Device, Result, Tensor};

use crate::ops::buf_tensor::CpuTensor;
use crate::onednn::BrgemmPackedWeight;


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
    static SCRATCH: RefCell<RawScratch> = RefCell::new(RawScratch::new());
}

/// Access the thread-local scratch buffers.
pub(crate) fn with_scratch<F, R>(f: F) -> R
where
    F: FnOnce(&mut RawScratch) -> R,
{
    SCRATCH.with_borrow_mut(f)
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
    crate::ops::tensor_as_u16_slice_pub(tensor)
}

/// Extract position_ids as Vec<i64> (one allocation, reusable across layers).
pub(crate) fn extract_positions(pos_ids: &Tensor) -> Result<Vec<i64>> {
    pos_ids
        .to_dtype(prelude_core::tensor::DType::I64)?
        .to_vec1::<i64>()
}

/// Extract seq_lens from cu_seqlens tensor.
pub(crate) fn extract_seq_lens(cu_seqlens: &Tensor) -> Result<Vec<usize>> {
    let cu: Vec<u32> = cu_seqlens.to_vec1()?;
    Ok(cu.windows(2).map(|w| (w[1] - w[0]) as usize).collect())
}

/// Wrap raw `&[u16]` data into a new BF16 Tensor (copies data).
pub(crate) fn wrap_output(data: &[u16], shape: &[usize], device: &Device) -> Result<Tensor> {
    let result_vec: Vec<half::bf16> = bytemuck::cast_slice::<u16, half::bf16>(data).to_vec();
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
    let pool = crate::ops::gemm_pool::gemm_pool();
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
        // nvtx_push
        if total <= 128 && gate_up_brg.n % 2 == 0 {
            crate::onednn::brgemm_fused_silu_mul_raw(
                input.as_ptr(),
                gate_up_brg,
                scratch.silu.as_mut_ptr(),
                total,
                dim,
            );
        } else {
            // Unfused path: gate_up GEMM → separate SiLU → down GEMM
            crate::onednn::brgemm_gemm_raw(
                input.as_ptr(),
                gate_up_brg,
                scratch.gate_up.as_mut_ptr(),
                total,
                gate_up_brg.n,
            );
            crate::ops::silu_mul::silu_and_mul_bf16(
                &mut scratch.silu[..silu_len],
                &scratch.gate_up[..gate_up_len],
                total,
                dim,
            );
        }
        // nvtx_pop

        // down GEMM: [total, intermediate] → [total, hidden]
        // nvtx_push
        crate::onednn::brgemm_gemm_raw(
            scratch.silu.as_ptr(),
            down_brg,
            output,
            total,
            hidden_size,
        );
        // nvtx_pop
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
            crate::ops::rmsnorm::rmsnorm_impl(
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
            crate::ops::rmsnorm::rmsnorm_impl(
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
                    crate::ops::rope::rope_neox_row(
                        &mut q_tok[off..off + hd],
                        cache,
                        cache_off,
                        ctx.embed_dim,
                    );
                }
                for h in 0..ctx.num_kv_heads {
                    let off = h * hd;
                    crate::ops::rope::rope_neox_row(
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

    let pool = crate::ops::gemm_pool::gemm_pool();
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
        // nvtx_push
        crate::onednn::brgemm_gemm_raw(
            x.as_ptr(),
            qkv_brg,
            scratch.qkv.as_mut_ptr(),
            total,
            qkv_n,
        );
        // nvtx_pop

        // 2-4. Fused deinterleave + QK-norm + RoPE
        // nvtx_push
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
        // nvtx_pop

        // 5. Attention
        // nvtx_push
        let q_normed = &scratch.q_normed[..total * q_size];
        let k_normed = &scratch.k_normed[..total * kv_size];
        let v_final = &scratch.v[..total * kv_size];
        crate::ops::attention::prefill_attention_bf16(
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
        // nvtx_pop

        // 6. O_proj GEMM
        // nvtx_push
        crate::onednn::brgemm_gemm_raw(
            scratch.attn_out.as_ptr(),
            oproj_brg,
            scratch.proj_out.as_mut_ptr(),
            total,
            input_dim,
        );
        // nvtx_pop

        CpuTensor::from_slice(&scratch.proj_out[..total * input_dim], &[total, input_dim])
    }
}

// ══════════════════════════════════════════════════════════════════════════
// F32 raw forward path — mirrors the BF16 raw path above but for f32.
// Uses OnednnF32PackedWeight for GEMM, rmsnorm_generic::<f32> for norms,
// rope_neox_row_f32 for RoPE, and onednn_f32_matmul + softmax_f32_inplace
// for attention.
// ══════════════════════════════════════════════════════════════════════════

use crate::ops::buf_tensor::CpuTensorF32;
use crate::onednn::OnednnF32PackedWeight;

pub(crate) struct RawScratchF32 {
    pub qkv: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub q_normed: Vec<f32>,
    pub k_normed: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub proj_out: Vec<f32>,
    pub gate_up: Vec<f32>,
    pub silu: Vec<f32>,
    pub mlp_out: Vec<f32>,
}

impl RawScratchF32 {
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
    static SCRATCH_F32: RefCell<RawScratchF32> = RefCell::new(RawScratchF32::new());
}

pub(crate) fn with_scratch_f32<F, R>(f: F) -> R
where
    F: FnOnce(&mut RawScratchF32) -> R,
{
    SCRATCH_F32.with_borrow_mut(f)
}

#[inline]
pub(crate) fn ensure_len_f32(buf: &mut Vec<f32>, needed: usize) {
    if buf.len() < needed {
        buf.resize(needed, 0.0);
    }
}

pub(crate) fn wrap_output_f32(data: &[f32], shape: &[usize], device: &Device) -> Result<Tensor> {
    Tensor::from_vec(data.to_vec(), shape, device)
}

// ── F32 raw residual add ────────────────────────────────────────────────

pub(crate) unsafe fn raw_residual_add_f32(dst: *mut f32, src: *const f32, len: usize) {
    #[repr(C)]
    struct AddCtx { dst: usize, src: usize, len: usize }
    unsafe fn add_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let c = &*(ctx_raw as *const AddCtx);
            let per = (c.len + n_threads - 1) / n_threads;
            let start = tid * per;
            let end = (start + per).min(c.len);
            if start >= end { return; }
            let dst = (c.dst as *mut f32).add(start);
            let src = (c.src as *const f32).add(start);
            let mut i = 0;
            let chunk = end - start;
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                while i + 16 <= chunk {
                    use core::arch::x86_64::*;
                    let a = _mm512_loadu_ps(dst.add(i));
                    let b = _mm512_loadu_ps(src.add(i));
                    _mm512_storeu_ps(dst.add(i), _mm512_add_ps(a, b));
                    i += 16;
                }
            }
            while i < chunk {
                *dst.add(i) += *src.add(i);
                i += 1;
            }
        }
    }
    let ctx = AddCtx { dst: dst as usize, src: src as usize, len };
    let pool = crate::ops::gemm_pool::gemm_pool();
    let n_threads = pool.num_threads().min(len / 64).max(1);
    unsafe { pool.dispatch(add_work, &ctx as *const AddCtx as *const u8, n_threads); }
}

// ── F32 raw MLP forward ─────────────────────────────────────────────────

/// Raw MLP forward: gate_up GEMM → SiLU×Mul → down GEMM.
/// All on raw `*const f32` / `*mut f32`, no Tensor allocations.
pub(crate) unsafe fn raw_mlp_forward_f32(
    scratch: &mut RawScratchF32,
    input: *const f32,
    total: usize,
    _hidden_size: usize,
    gate_up_pw: &OnednnF32PackedWeight,
    down_pw: &OnednnF32PackedWeight,
    output: *mut f32,
) {
    unsafe {
        let n_gate_up = gate_up_pw.n;
        let dim = n_gate_up / 2;
        let gate_up_len = total * n_gate_up;
        let silu_len = total * dim;
        ensure_len_f32(&mut scratch.gate_up, gate_up_len);
        ensure_len_f32(&mut scratch.silu, silu_len);

        // gate_up GEMM: [total, hidden] → [total, 2*dim]
        gate_up_pw.forward_raw(input, scratch.gate_up.as_mut_ptr(), total);

        // SiLU(gate) * up
        for row in 0..total {
            let base = row * n_gate_up;
            let dst = row * dim;
            let mut i = 0;
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                use core::arch::x86_64::*;
                let _ones = _mm512_set1_ps(1.0);
                while i + 16 <= dim {
                    let gate = _mm512_loadu_ps(scratch.gate_up.as_ptr().add(base + i));
                    let up = _mm512_loadu_ps(scratch.gate_up.as_ptr().add(base + dim + i));
                    // SiLU(x) = x / (1 + exp(-x))
                    // Approximate: use negation + fast exp via scalar fallback for now
                    // Actually, _mm512 doesn't have exp, do scalar
                    let mut gate_arr = [0.0f32; 16];
                    let mut up_arr = [0.0f32; 16];
                    _mm512_storeu_ps(gate_arr.as_mut_ptr(), gate);
                    _mm512_storeu_ps(up_arr.as_mut_ptr(), up);
                    for j in 0..16 {
                        let g = gate_arr[j];
                        gate_arr[j] = (g / (1.0 + (-g).exp())) * up_arr[j];
                    }
                    _mm512_storeu_ps(
                        scratch.silu.as_mut_ptr().add(dst + i),
                        _mm512_loadu_ps(gate_arr.as_ptr()),
                    );
                    i += 16;
                }
            }
            while i < dim {
                let g = scratch.gate_up[base + i];
                scratch.silu[dst + i] = (g / (1.0 + (-g).exp())) * scratch.gate_up[base + dim + i];
                i += 1;
            }
        }

        // down GEMM: [total, dim] → [total, hidden]
        down_pw.forward_raw(scratch.silu.as_ptr(), output, total);
    }
}

// ── F32 raw fused deinterleave + QK-norm + RoPE ─────────────────────────

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn raw_fused_deinterleave_norm_rope_f32(
    qkv_buf: *const f32,
    q_normed_buf: *mut f32,
    k_normed_buf: *mut f32,
    v_buf: *mut f32,
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    cos_sin_cache: &[f32],
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
        q_normed_ptr: usize, k_normed_ptr: usize, v_out_ptr: usize,
        q_norm_w_ptr: usize, k_norm_w_ptr: usize,
        cache_ptr: usize, cache_len: usize, pos_ptr: usize,
        total: usize, q_size: usize, kv_size: usize, qkv_n: usize,
        num_heads: usize, num_kv_heads: usize, head_dim: usize,
        embed_dim: usize, rotary_dim: usize, eps: f32,
    }

    unsafe fn fused_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
        unsafe {
            let ctx = &*(ctx_raw as *const FusedCtx);
            let rows_per = (ctx.total + n_threads - 1) / n_threads;
            let t_start = tid * rows_per;
            let t_end = (t_start + rows_per).min(ctx.total);
            if t_start >= t_end { return; }
            let chunk = t_end - t_start;
            let hd = ctx.head_dim;

            let qkv = ctx.qkv_ptr as *const f32;
            let q_normed = ctx.q_normed_ptr as *mut f32;
            let k_normed = ctx.k_normed_ptr as *mut f32;
            let v_out = ctx.v_out_ptr as *mut f32;
            let q_norm_w = std::slice::from_raw_parts(ctx.q_norm_w_ptr as *const f32, hd);
            let k_norm_w = std::slice::from_raw_parts(ctx.k_norm_w_ptr as *const f32, hd);
            let positions = std::slice::from_raw_parts(ctx.pos_ptr as *const i64, ctx.total);

            // Step 1: Deinterleave QKV → q_normed (temp), k_normed (temp), v_out
            for t in t_start..t_end {
                let src = qkv.add(t * ctx.qkv_n);
                std::ptr::copy_nonoverlapping(src, q_normed.add(t * ctx.q_size), ctx.q_size);
                std::ptr::copy_nonoverlapping(src.add(ctx.q_size), k_normed.add(t * ctx.kv_size), ctx.kv_size);
                std::ptr::copy_nonoverlapping(src.add(ctx.q_size + ctx.kv_size), v_out.add(t * ctx.kv_size), ctx.kv_size);
            }

            // Step 2: RMSNorm Q in-place in q_normed
            let q_slice = std::slice::from_raw_parts_mut(
                q_normed.add(t_start * ctx.q_size), chunk * ctx.q_size,
            );
            {
                let q_in = q_slice.to_vec();
                crate::ops::rmsnorm::rmsnorm_generic(
                    q_slice, &q_in, q_norm_w, chunk * ctx.num_heads, hd, ctx.eps,
                );
            }

            // Step 3: RMSNorm K in-place in k_normed
            let k_slice = std::slice::from_raw_parts_mut(
                k_normed.add(t_start * ctx.kv_size), chunk * ctx.kv_size,
            );
            {
                let k_in = k_slice.to_vec();
                crate::ops::rmsnorm::rmsnorm_generic(
                    k_slice, &k_in, k_norm_w, chunk * ctx.num_kv_heads, hd, ctx.eps,
                );
            }

            // Step 4: RoPE in-place on q_normed and k_normed
            let cache = std::slice::from_raw_parts(ctx.cache_ptr as *const f32, ctx.cache_len);
            for t in t_start..t_end {
                let pos = positions[t];
                if pos < 0 { continue; }
                let cache_off = pos as usize * ctx.rotary_dim;
                let q_tok = std::slice::from_raw_parts_mut(q_normed.add(t * ctx.q_size), ctx.q_size);
                let k_tok = std::slice::from_raw_parts_mut(k_normed.add(t * ctx.kv_size), ctx.kv_size);
                for h in 0..ctx.num_heads {
                    let off = h * hd;
                    crate::ops::rope::rope_neox_row_f32(
                        &mut q_tok[off..off + hd], cache, cache_off, ctx.embed_dim,
                    );
                }
                for h in 0..ctx.num_kv_heads {
                    let off = h * hd;
                    crate::ops::rope::rope_neox_row_f32(
                        &mut k_tok[off..off + hd], cache, cache_off, ctx.embed_dim,
                    );
                }
            }
        }
    }

    let fctx = FusedCtx {
        qkv_ptr: qkv_buf as usize,
        q_normed_ptr: q_normed_buf as usize,
        k_normed_ptr: k_normed_buf as usize,
        v_out_ptr: v_buf as usize,
        q_norm_w_ptr: q_norm_w.as_ptr() as usize,
        k_norm_w_ptr: k_norm_w.as_ptr() as usize,
        cache_ptr: cos_sin_cache.as_ptr() as usize,
        cache_len: cos_sin_cache.len(),
        pos_ptr: positions.as_ptr() as usize,
        total, q_size, kv_size, qkv_n, num_heads, num_kv_heads, head_dim,
        embed_dim, rotary_dim, eps,
    };

    let pool = crate::ops::gemm_pool::gemm_pool();
    let n_threads = pool.num_threads().min(total);
    unsafe { pool.dispatch(fused_work, &fctx as *const FusedCtx as *const u8, n_threads); }
}

// ── F32 raw prefill attention ───────────────────────────────────────────

/// Per-head F32 attention: Q @ K^T * scale → softmax → @ V, parallelized via rayon.
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_prefill_attention_f32(
    output: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_lens: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
) {
    use rayon::prelude::*;

    let gqa_ratio = num_heads / num_kv_heads;
    let total: usize = seq_lens.iter().sum();
    let q_row_stride = num_heads * head_dim;
    let kv_row_stride = num_kv_heads * head_dim;

    struct HeadWork {
        seq_offset: usize,
        seq_len: usize,
        head: usize,
    }

    let mut work_items: Vec<HeadWork> = Vec::with_capacity(total * num_heads / seq_lens.len().max(1));
    let mut offset = 0usize;
    for &slen in seq_lens {
        for h in 0..num_heads {
            work_items.push(HeadWork { seq_offset: offset, seq_len: slen, head: h });
        }
        offset += slen;
    }

    let out_ptr = output.as_mut_ptr() as usize;
    let q_ptr = q.as_ptr() as usize;
    let k_ptr = k.as_ptr() as usize;
    let v_ptr = v.as_ptr() as usize;

    work_items.par_iter().for_each(|w| {
        let slen = w.seq_len;
        let h = w.head;
        let kv_h = h / gqa_ratio;
        let base = w.seq_offset;

        let mut scores = vec![0.0f32; slen * slen];

        // Q @ K^T for this head, with causal masking
        for qi in 0..slen {
            let q_off = (base + qi) * q_row_stride + h * head_dim;
            let q_row = unsafe { std::slice::from_raw_parts((q_ptr as *const f32).add(q_off), head_dim) };
            for ki in 0..=qi {
                let k_off = (base + ki) * kv_row_stride + kv_h * head_dim;
                let k_row = unsafe { std::slice::from_raw_parts((k_ptr as *const f32).add(k_off), head_dim) };
                let mut dot = 0.0f32;
                for d in 0..head_dim { dot += q_row[d] * k_row[d]; }
                scores[qi * slen + ki] = dot * softmax_scale;
            }
            for ki in (qi + 1)..slen {
                scores[qi * slen + ki] = f32::NEG_INFINITY;
            }
        }

        // Softmax
        crate::ops::softmax::softmax_f32_inplace(&mut scores, slen, slen);

        // Scores @ V
        for qi in 0..slen {
            let out_off = (base + qi) * q_row_stride + h * head_dim;
            let o_row = unsafe { std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(out_off), head_dim) };
            o_row.fill(0.0);
            for vi in 0..=qi {
                let v_off = (base + vi) * kv_row_stride + kv_h * head_dim;
                let v_row = unsafe { std::slice::from_raw_parts((v_ptr as *const f32).add(v_off), head_dim) };
                let s = scores[qi * slen + vi];
                for d in 0..head_dim { o_row[d] += s * v_row[d]; }
            }
        }
    });
}

// ── F32 raw attention forward (full pipeline) ───────────────────────────

/// Full raw F32 attention: QKV GEMM → fused deinterleave+norm+rope → attention → O_proj GEMM.
/// Returns a CpuTensorF32 borrowing scratch.proj_out.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn raw_attention_forward_f32<'a>(
    scratch: &'a mut RawScratchF32,
    x: &CpuTensorF32,
    qkv_pw: &OnednnF32PackedWeight,
    oproj_pw: &OnednnF32PackedWeight,
    q_norm_w: &[f32],
    k_norm_w: &[f32],
    cos_sin_cache: &[f32],
    cos_sin_rotary_dim: usize,
    positions: &[i64],
    seq_lens: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    softmax_scale: f32,
) -> CpuTensorF32<'a> {
    unsafe {
        let total = x.dim(0);
        let input_dim = x.dim(1);
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let qkv_n = q_size + 2 * kv_size;

        ensure_len_f32(&mut scratch.qkv, total * qkv_n);
        ensure_len_f32(&mut scratch.q_normed, total * q_size);
        ensure_len_f32(&mut scratch.k_normed, total * kv_size);
        ensure_len_f32(&mut scratch.v, total * kv_size);
        ensure_len_f32(&mut scratch.attn_out, total * q_size);
        ensure_len_f32(&mut scratch.proj_out, total * input_dim);

        // 1. QKV GEMM
        qkv_pw.forward_raw(x.as_slice().as_ptr(), scratch.qkv.as_mut_ptr(), total);

        // 2-4. Fused deinterleave + QK-norm + RoPE
        raw_fused_deinterleave_norm_rope_f32(
            scratch.qkv.as_ptr(),
            scratch.q_normed.as_mut_ptr(),
            scratch.k_normed.as_mut_ptr(),
            scratch.v.as_mut_ptr(),
            q_norm_w, k_norm_w,
            cos_sin_cache, positions,
            total, q_size, kv_size, qkv_n,
            num_heads, num_kv_heads, head_dim, cos_sin_rotary_dim, eps,
        );

        // 5. Attention
        raw_prefill_attention_f32(
            &mut scratch.attn_out[..total * q_size],
            &scratch.q_normed[..total * q_size],
            &scratch.k_normed[..total * kv_size],
            &scratch.v[..total * kv_size],
            seq_lens,
            num_heads, num_kv_heads, head_dim, softmax_scale,
        );

        // 6. O_proj GEMM
        oproj_pw.forward_raw(scratch.attn_out.as_ptr(), scratch.proj_out.as_mut_ptr(), total);

        CpuTensorF32::from_slice(&scratch.proj_out[..total * input_dim], &[total, input_dim])
    }
}

//! candle-fa3-0102 backend: vendored vLLM 0.22 FA3 hopper kernel (SM90).
//!
//! Raw-pointer FFI to `run_mha_v3_prelude` in `libprelude_fa3_0102.a`, built by
//! `crates/prelude-cuda/fa3_0102/_build_v3_kernel_prelude.sh` (CUDA 13.2 +
//! CUTLASS 3.8.0). The host-side parameter assembly inside that entry is a
//! field-for-field port of vLLM 0.22's `hopper/flash_api.cpp`
//! (vllm-project/flash-attention@bce2942), so this wrapper only has to hand
//! over device pointers, strides and shapes.
//!
//! Restrictions: bf16, head_dim 128, contiguous last dim. Two paths:
//!   - non-paged varlen: K/V are (total_k, h_k, 128) with cu_seqlens_k
//!   - paged KV: K/V caches are (num_pages, page_size, h_k, 128) with
//!     page_table (b, max_pages) + seqused_k (b,); cu_seqlens_k is NOT passed
//!     (vLLM convention: page table and cu_seqlens_k are mutually exclusive)

use crate::device::{self as cb};
use cudarc::driver::DevicePtr;
use half::bf16;
use prelude_core::tensor::{DType, Result, Tensor};
use std::ffi::c_void;

#[allow(clippy::too_many_arguments)]
unsafe extern "C" {
    fn run_mha_v3_prelude(
        stream: *mut c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        o: *mut c_void,
        softmax_lse: *mut c_void,
        cu_seqlens_q: *const i32,
        cu_seqlens_k: *const i32,
        seqused_k: *const i32,
        page_table: *const i32,
        sched_metadata: *mut i32,
        page_size: u32,
        num_pages: u32,
        page_table_batch_stride: u32,
        k_batch_stride: u64,
        v_batch_stride: u64,
        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,
        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,
        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        max_seqlen_q: u32,
        max_seqlen_k: u32,
        total_q: u32,
        total_k: u32,
        softmax_scale: f32,
        is_causal: i32,
        is_bf16: i32,
        rotary_cos: *const c_void,
        rotary_sin: *const c_void,
        qnorm_weight: *const c_void,
        qnorm_eps: f32,
    );
}

/// Q-prologue fusion inputs: in-kernel per-head RMSNorm + RoPE on raw Q.
/// `cos`/`sin` are (max_pos, head_dim/2) bf16 rotate-half tables; positions
/// are derived in-kernel (seqused_k - seqlen_q + i), so no position_ids.
pub struct QPrologue<'a> {
    pub q_weight: &'a Tensor,
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub eps: f32,
}

/// Non-paged varlen attention: q (total_q, h, 128), k/v (total_k, h_k, 128).
#[allow(clippy::too_many_arguments)]
pub fn varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    call_v3(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        Some(cu_seqlens_k),
        None,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        None,
    )
}

/// Paged varlen attention: q (total_q, h, 128), caches (num_pages, page_size,
/// h_k, 128), block_tables (b, max_pages) u32/i32, seqused_k (b,) u32/i32.
/// With `prologue` set, Q is RAW (pre-norm, pre-rope) and the kernel applies
/// per-head RMSNorm + RoPE in its prologue.
#[allow(clippy::too_many_arguments)]
pub fn varlen_paged(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seqlens_q: &Tensor,
    seqused_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    prologue: Option<QPrologue<'_>>,
) -> Result<Tensor> {
    call_v3(
        q,
        key_cache,
        value_cache,
        Some(block_tables),
        cu_seqlens_q,
        None,
        Some(seqused_k),
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        true,
        prologue,
    )
}

#[allow(clippy::too_many_arguments)]
fn call_v3(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    page_table: Option<&Tensor>,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: Option<&Tensor>,
    seqused_k: Option<&Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
    prologue: Option<QPrologue<'_>>,
) -> Result<Tensor> {
    let (total_q, h, d) = q.shape().dims3()?;
    if q.dtype() != DType::BF16 {
        candle_core::bail!("fa3-0102: bf16 only, got {:?}", q.dtype());
    }
    if d != 128 {
        candle_core::bail!("fa3-0102: head_dim 128 only, got {d}");
    }
    let paged = page_table.is_some();

    // K/V geometry. Strides are in elements; last dim must be contiguous.
    let k_strides = k.layout().stride().to_vec();
    let v_strides = v.layout().stride().to_vec();
    let q_strides = q.layout().stride().to_vec();
    let (h_k, page_size, num_pages, total_k, k_batch_stride, v_batch_stride, k_row, k_head) =
        if paged {
            let (num_pages, page_size, h_k, dk) = k.shape().dims4()?;
            if dk != 128 {
                candle_core::bail!("fa3-0102: cache head_dim 128 only, got {dk}");
            }
            let b = cu_seqlens_q.dim(0)? - 1;
            // vLLM mha_fwd: total_k = batch_size * k.size(1) when paged.
            (
                h_k,
                page_size,
                num_pages,
                b * page_size,
                k_strides[0] as u64,
                v_strides[0] as u64,
                k_strides[1],
                k_strides[2],
            )
        } else {
            let (total_k, h_k, _) = k.shape().dims3()?;
            (h_k, 1, 0, total_k, 0u64, 0u64, k_strides[0], k_strides[1])
        };
    if *q_strides.last().unwrap_or(&0) != 1 || *k_strides.last().unwrap_or(&0) != 1 {
        candle_core::bail!("fa3-0102: q/k/v must have contiguous last dim");
    }

    let b = cu_seqlens_q.dim(0)? - 1;
    let stream = cb::tensor_stream(q)?;
    let out = Tensor::zeros(&[total_q, h, d], DType::BF16, q.device())?;
    let softmax_lse = Tensor::zeros(&[h * total_q], DType::F32, q.device())?;
    // Scheduler metadata: [prepare_seqlen_q (b_r) | num_nheads_in_l2 (b_r) | semaphore]
    let b_rounded = b.div_ceil(4) * 4;
    let sched = Tensor::zeros(&[1 + b_rounded * 2], DType::U32, q.device())?;

    {
        let raw_stream = stream.cu_stream() as *mut c_void;

        macro_rules! cuda_ptr {
            ($t:expr, $ty:ty) => {{
                let (storage, layout) = $t.storage_and_layout();
                let cuda = match &*storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => candle_core::bail!("fa3-0102: requires CUDA tensors"),
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
        let lse_ptr = cuda_ptr!(&softmax_lse, f32);
        let sched_ptr = cuda_ptr!(&sched, u32);
        let cu_q_ptr = cuda_ptr!(cu_seqlens_q, u32);
        let cu_k_ptr = match cu_seqlens_k {
            Some(t) => cuda_ptr!(t, u32),
            None => 0,
        };
        let seqused_k_ptr = match seqused_k {
            Some(t) => cuda_ptr!(t, u32),
            None => 0,
        };
        let (pt_ptr, pt_batch_stride) = match page_table {
            Some(t) => {
                let s = t.layout().stride().to_vec();
                if *s.last().unwrap_or(&0) != 1 {
                    candle_core::bail!("fa3-0102: page_table must have contiguous last dim");
                }
                (cuda_ptr!(t, u32), s[0] as u32)
            }
            None => (0, 0),
        };
        let (cos_ptr, sin_ptr, qw_ptr, qnorm_eps) = match &prologue {
            Some(p) => (
                cuda_ptr!(p.cos, bf16),
                cuda_ptr!(p.sin, bf16),
                cuda_ptr!(p.q_weight, bf16),
                p.eps,
            ),
            None => (0, 0, 0, 0f32),
        };

        unsafe {
            run_mha_v3_prelude(
                raw_stream,
                q_ptr as *mut c_void,
                k_ptr as *mut c_void,
                v_ptr as *mut c_void,
                o_ptr as *mut c_void,
                lse_ptr as *mut c_void,
                cu_q_ptr as *const i32,
                cu_k_ptr as *const i32,
                seqused_k_ptr as *const i32,
                pt_ptr as *const i32,
                sched_ptr as *mut i32,
                page_size as u32,
                num_pages as u32,
                pt_batch_stride,
                k_batch_stride,
                v_batch_stride,
                q_strides[0] as u32,
                k_row as u32,
                v_strides[if paged { 1 } else { 0 }] as u32,
                (h * d) as u32, // o_row_stride: out is freshly-allocated contiguous
                q_strides[1] as u32,
                k_head as u32,
                v_strides[if paged { 2 } else { 1 }] as u32,
                d as u32, // o_head_stride
                b as u32,
                h as u32,
                h_k as u32,
                d as u32,
                max_seqlen_q as u32,
                max_seqlen_k as u32,
                total_q as u32,
                total_k as u32,
                softmax_scale,
                causal as i32,
                1, // bf16
                cos_ptr as *const c_void,
                sin_ptr as *const c_void,
                qw_ptr as *const c_void,
                qnorm_eps,
            );
        }
    }

    Ok(out)
}

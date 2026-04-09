//! FlashInfer attention backend (SM80+ via FA2, SM90+ via FA3).
//!
//! Wraps `prelude_flashinfer` crate with the plan-then-run API.
//! Workspace buffers are allocated once per device and reused.

use crate::device::{self as cb, DevicePtr};
use cudarc::driver::CudaStream;
use prelude_core::tensor::{bail, DType, Device, DeviceExt, Result, Tensor};
use half::bf16;
use prelude_flashinfer::types::*;
use prelude_flashinfer::{DecodeKey, KernelDtype, KernelRegistry, MaskMode, PrefillKey};
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

// ── Constants ────────────────────────────────────────────────────────

const FLOAT_WS_BYTES: usize = 128 * 1024 * 1024; // 128 MB GPU float workspace
const INT_WS_BYTES: usize = 8 * 1024 * 1024; // 8 MB GPU int workspace
const PINNED_WS_BYTES: usize = 8 * 1024 * 1024; // 8 MB CPU pinned workspace
const KV_LAYOUT_NHD: i64 = 0;

// ── CUDA FFI ─────────────────────────────────────────────────────────

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void,
    ) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;

// ── Global state ─────────────────────────────────────────────────────

static REGISTRY: OnceLock<KernelRegistry> = OnceLock::new();
static WORKSPACE: Mutex<Option<Workspace>> = Mutex::new(None);

fn registry() -> &'static KernelRegistry {
    REGISTRY.get_or_init(KernelRegistry::new)
}

fn dtype_to_kernel(dtype: DType) -> KernelDtype {
    match dtype {
        DType::BF16 => KernelDtype::BF16,
        DType::F16 => KernelDtype::FP16,
        other => panic!("FlashInfer: unsupported dtype {other:?}, expected BF16 or F16"),
    }
}

// ── Plan cache ──────────────────────────────────────────────────────
//
// FlashInfer's plan() is expensive: it does GPU→CPU copies (to_vec1) + CPU
// scheduling computation.  Within a single model.forward() call, all N
// attention layers share the same batch structure, so plan() only needs to
// run once.
//
// The thread-local cache stores the plan_info from the first call and
// reuses it for subsequent layers.  begin_forward() / end_forward() bracket
// the forward pass.

/// Paged metadata with both CPU and GPU representations.
/// CPU data for plan() (avoids D2H round trip), GPU tensors for run().
#[derive(Clone)]
struct PagedMeta {
    // GPU tensors for run()
    indptr_gpu: Tensor,
    indices_gpu: Tensor,
    last_page_len_gpu: Tensor,
    // CPU data for plan() — no D2H needed
    indptr_cpu: Vec<i32>,
    kv_lens_cpu: Vec<i32>,
    last_page_len_cpu: Vec<i32>,
}

struct PlanCache {
    /// Ragged prefill plan (varlen_causal / varlen_bidirectional / varlen_windowed)
    ragged_plan: Option<TVMFFIAny>,
    /// Paged decode plan (Q=1)
    decode_plan: Option<TVMFFIAny>,
    /// Paged prefill plan (Q>1 with paged KV)
    paged_prefill_plan: Option<TVMFFIAny>,
    /// Cached paged metadata — avoids repeated GPU→CPU copies across layers
    paged_metadata: Option<PagedMeta>,
}

thread_local! {
    static PLAN_CACHE: RefCell<Option<PlanCache>> = const { RefCell::new(None) };
}

/// Call before model.forward() to enable plan caching across layers.
pub fn begin_forward() {
    PLAN_CACHE.with(|c| {
        *c.borrow_mut() = Some(PlanCache {
            ragged_plan: None,
            decode_plan: None,
            paged_prefill_plan: None,
            paged_metadata: None,
        });
    });
}

/// Call after model.forward() to clear cached plans.
pub fn end_forward() {
    PLAN_CACHE.with(|c| {
        *c.borrow_mut() = None;
    });
}

/// Pre-compute paged decode plan and inject into cache.
/// Called BEFORE CUDA graph capture/replay so plan() is outside the graph.
/// The graph only captures run() calls.
///
/// `block_tables` and `cu_seqlens_k` are GPU Tensors (same as engine provides).
/// `cu_seqlens_q` is the Q cumulative lengths (GPU Tensor).
pub fn precompute_paged_plan(
    q_shape: (usize, usize, usize), // (batch_size, num_qo_heads, head_dim)
    key_cache: &Tensor,
    cu_seqlens_q: &Tensor,
    block_tables: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
) -> Result<()> {
    precompute_paged_plan_impl(q_shape, key_cache, cu_seqlens_q, block_tables, cu_seqlens_k, softmax_scale, None)
}

/// Pre-compute paged plan with pre-allocated metadata buffers for CUDA graph.
///
/// Like `precompute_paged_plan` but writes FlashInfer metadata (indptr, indices,
/// last_page_len) into fixed-address GPU tensors via `cudaMemcpyAsync` instead
/// of creating new allocations. This ensures the CUDA graph's run() kernels
/// always reference the same GPU addresses between capture and replay.
///
/// SGLang reference: `flashinfer_backend.py` lines 568-571 (paged_kv_*_buffer).
pub fn precompute_paged_plan_graphed(
    q_shape: (usize, usize, usize),
    key_cache: &Tensor,
    cu_seqlens_q: &Tensor,
    block_tables: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
    fi_indptr: &Tensor,
    fi_indices: &Tensor,
    fi_last_page_len: &Tensor,
) -> Result<()> {
    precompute_paged_plan_impl(
        q_shape, key_cache, cu_seqlens_q, block_tables, cu_seqlens_k, softmax_scale,
        Some((fi_indptr, fi_indices, fi_last_page_len)),
    )
}

/// Allocate pre-allocated FlashInfer metadata buffers for CUDA graph.
/// Returns (indptr, indices, last_page_len) with fixed GPU addresses
/// that survive across graph capture and replay.
pub fn allocate_fi_graph_meta(
    batch_size: usize,
    max_total_pages: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    Ok((
        Tensor::zeros((batch_size + 1,), DType::I32, device)?,
        Tensor::zeros((max_total_pages,), DType::I32, device)?,
        Tensor::zeros((batch_size,), DType::I32, device)?,
    ))
}

/// Shared implementation for precompute_paged_plan / precompute_paged_plan_graphed.
///
/// When `graph_buffers` is Some, metadata is written into pre-allocated GPU tensors
/// via cudaMemcpyAsync (for CUDA graph address stability).
/// When None, new GPU tensors are allocated (normal eager path).
fn precompute_paged_plan_impl(
    q_shape: (usize, usize, usize),
    key_cache: &Tensor,
    cu_seqlens_q: &Tensor,
    block_tables: &Tensor,
    cu_seqlens_k: &Tensor,
    _softmax_scale: f32,
    graph_buffers: Option<(&Tensor, &Tensor, &Tensor)>,
) -> Result<()> {
    let (batch_size, num_qo_heads, head_dim) = q_shape;
    let (_, block_size, num_kv_heads, _) = key_cache.shape().dims4()?;

    let stream = cb::tensor_stream(key_cache)?;
    let did = device_id(key_cache.device());
    let raw_stream = stream.cu_stream() as *mut c_void;

    let meta = if let Some((fi_ip, fi_ix, fi_lp)) = graph_buffers {
        convert_paged_metadata_into(block_tables, cu_seqlens_k, block_size, fi_ip, fi_ix, fi_lp, &stream, raw_stream)?
    } else {
        convert_paged_metadata(block_tables, cu_seqlens_k, block_size)?
    };

    let reg = registry();
    let backend = reg.default_backend();
    let variant = reg
        .get_prefill(&PrefillKey {
            dtype: dtype_to_kernel(key_cache.dtype()),
            head_dim_qk: head_dim as u32, head_dim_vo: head_dim as u32,
            sliding_window: false, logits_soft_cap: false,
            backend,
        })
        .ok_or_else(|| prelude_core::tensor::Error::Msg(format!("FlashInfer: no {backend:?} prefill variant")))?;

    let ws_guard = get_workspace(did)?;
    let ws = ws_guard.as_ref().unwrap();

    let is_fa3 = matches!(backend, prelude_flashinfer::Backend::FA3);

    // CPU data — no GPU→CPU copies needed for indptr/kv_lens (from PagedMeta)
    let cuq_cpu: Vec<i32> = cu_seqlens_q.to_vec1::<u32>()?.iter().map(|&v| v as i32).collect();
    let total_q: usize = cuq_cpu.last().copied().unwrap_or(0) as usize;

    let fws: [i64; 1] = [FLOAT_WS_BYTES as i64];
    let iws: [i64; 1] = [INT_WS_BYTES as i64];
    let pws: [i64; 1] = [PINNED_WS_BYTES as i64];
    let cuq_s: [i64; 1] = [(batch_size + 1) as i64];
    let ip_s: [i64; 1] = [(batch_size + 1) as i64];
    let kvl_s: [i64; 1] = [batch_size as i64];

    let fws_st = contiguous_strides(&fws);
    let iws_st = contiguous_strides(&iws);
    let pws_st = contiguous_strides(&pws);
    let cuq_st = contiguous_strides(&cuq_s);
    let ip_st = contiguous_strides(&ip_s);
    let kvl_st = contiguous_strides(&kvl_s);

    let dl_fws = make_gpu_dl(ws.float_ws, did, U8_DT, &fws, &fws_st);
    let dl_iws = make_gpu_dl(ws.int_ws, did, U8_DT, &iws, &iws_st);
    let dl_pws = make_cpu_dl(ws.pinned_ws, U8_DT, &pws, &pws_st);
    let dl_cuq_cpu = make_cpu_dl(cuq_cpu.as_ptr() as *mut c_void, I32_DT, &cuq_s, &cuq_st);
    let dl_ip_cpu = make_cpu_dl(meta.indptr_cpu.as_ptr() as *mut c_void, I32_DT, &ip_s, &ip_st);
    let dl_kvl_cpu = make_cpu_dl(meta.kv_lens_cpu.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);

    reg.set_stream(did, raw_stream);

    let mut plan_args = vec![
        TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), TVMFFIAny::dltensor(&dl_pws),
        TVMFFIAny::dltensor(&dl_cuq_cpu), TVMFFIAny::dltensor(&dl_ip_cpu), TVMFFIAny::dltensor(&dl_kvl_cpu),
        TVMFFIAny::int64(total_q as i64), TVMFFIAny::int64(batch_size as i64),
        TVMFFIAny::int64(num_qo_heads as i64), TVMFFIAny::int64(num_kv_heads as i64),
        TVMFFIAny::int64(block_size as i64),
        TVMFFIAny::bool_val(false), // cuda_graph — false for plan itself
        TVMFFIAny::int64(head_dim as i64), TVMFFIAny::int64(head_dim as i64),
        TVMFFIAny::bool_val(true), // causal
        TVMFFIAny::int64(-1),      // window_left
    ];
    append_fa2_plan_tail(&mut plan_args, is_fa3);

    let pi = unsafe {
        reg.call(variant.plan, &plan_args)
            .map_err(|e| prelude_core::tensor::Error::Msg(format!("FlashInfer precompute plan: {e}")))?
    };

    drop(ws_guard);

    // Inject into plan cache so model.forward() only calls run()
    PLAN_CACHE.with(|c| {
        *c.borrow_mut() = Some(PlanCache {
            ragged_plan: None,
            decode_plan: None,
            paged_prefill_plan: Some(pi),
            paged_metadata: Some(meta),
        });
    });

    Ok(())
}

// ── Workspace ────────────────────────────────────────────────────────

struct Workspace {
    float_ws: *mut c_void,
    int_ws: *mut c_void,
    pinned_ws: *mut c_void,
    device_id: i32,
}

unsafe impl Send for Workspace {}

impl Workspace {
    fn new(device_id: i32) -> Result<Self> {
        unsafe {
            cudaSetDevice(device_id);
            let mut float_ws = std::ptr::null_mut();
            let mut int_ws = std::ptr::null_mut();
            let mut pinned_ws = std::ptr::null_mut();
            if cudaMalloc(&mut float_ws, FLOAT_WS_BYTES) != 0 {
                bail!("FlashInfer: cudaMalloc float_ws failed");
            }
            if cudaMalloc(&mut int_ws, INT_WS_BYTES) != 0 {
                cudaFree(float_ws);
                bail!("FlashInfer: cudaMalloc int_ws failed");
            }
            if cudaMallocHost(&mut pinned_ws, PINNED_WS_BYTES) != 0 {
                cudaFree(float_ws);
                cudaFree(int_ws);
                bail!("FlashInfer: cudaMallocHost pinned_ws failed");
            }
            cudaMemset(float_ws, 0, FLOAT_WS_BYTES);
            cudaMemset(int_ws, 0, INT_WS_BYTES);
            // Sync to ensure workspace zeroing completes before any non-default
            // stream (cudarc) uses these buffers. Without this, the first
            // FlashInfer plan/run can read uninitialized workspace data.
            cudaDeviceSynchronize();
            tracing::debug!("FlashInfer workspace: float=128MB int=8MB pinned=8MB");
            Ok(Self { float_ws, int_ws, pinned_ws, device_id })
        }
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        unsafe {
            cudaSetDevice(self.device_id);
            cudaFree(self.float_ws);
            cudaFree(self.int_ws);
            cudaFreeHost(self.pinned_ws);
        }
    }
}

fn get_workspace(device_id: i32) -> Result<std::sync::MutexGuard<'static, Option<Workspace>>> {
    let mut guard = WORKSPACE
        .lock()
        .map_err(|e| prelude_core::tensor::Error::Msg(format!("FlashInfer workspace lock: {e}")))?;
    if guard.is_none() {
        *guard = Some(Workspace::new(device_id)?);
    }
    Ok(guard)
}

// ── DLTensor helpers ─────────────────────────────────────────────────

fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn make_gpu_dl(
    data: *mut c_void, dev_id: i32, dtype: DLDataType, shape: &[i64], strides: &[i64],
) -> DLTensor {
    DLTensor {
        data,
        device: DLDevice { device_type: KDLCUDA, device_id: dev_id },
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
        byte_offset: 0,
    }
}

fn make_cpu_dl(
    data: *mut c_void, dtype: DLDataType, shape: &[i64], strides: &[i64],
) -> DLTensor {
    DLTensor {
        data,
        device: DLDevice { device_type: KDLCPU, device_id: 0 },
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
        byte_offset: 0,
    }
}

const BF16_DT: DLDataType = DLDataType { code: KDLBFLOAT, bits: 16, lanes: 1 };
const I32_DT: DLDataType = DLDataType { code: KDLINT, bits: 32, lanes: 1 };
const U8_DT: DLDataType = DLDataType { code: KDLUINT, bits: 8, lanes: 1 };

macro_rules! cuda_ptr {
    ($t:expr, $ty:ty, $stream:expr) => {{
        let (storage, layout) = $t.storage_and_layout();
        let cuda = match &*storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("FlashInfer: requires CUDA"),
        };
        let slice = cuda.as_cuda_slice::<$ty>()?.slice(layout.start_offset()..);
        let (ptr, _guard) = slice.device_ptr($stream);
        ptr as u64 as *mut c_void
    }};
}

fn device_id(dev: &Device) -> i32 {
    dev.ordinal() as i32
}

// ── Public API ───────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn varlen_causal(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, _max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    ragged_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale, MaskMode::Causal, -1)
}

#[allow(clippy::too_many_arguments)]
pub fn varlen_bidirectional(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, _max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    ragged_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale, MaskMode::NonCausal, -1)
}

#[allow(clippy::too_many_arguments)]
pub fn varlen_windowed(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, _max_seqlen_k: usize,
    softmax_scale: f32,
    window_left: Option<usize>, _window_right: Option<usize>,
) -> Result<Tensor> {
    let wl = window_left.map(|v| v as i64).unwrap_or(-1);
    ragged_prefill(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, softmax_scale, MaskMode::Causal, wl)
}

#[allow(clippy::too_many_arguments)]
pub fn varlen_paged(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor, block_tables: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    max_seqlen_q: usize, _max_seqlen_k: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let (_, block_size, num_kv_heads, head_dim) = key_cache.shape().dims4()?;

    // Cache paged metadata across layers to avoid repeated GPU→CPU copies.
    let cached_meta = PLAN_CACHE.with(|c| {
        c.borrow().as_ref().and_then(|pc| pc.paged_metadata.clone())
    });
    let meta = if let Some(m) = cached_meta {
        m
    } else {
        let m = convert_paged_metadata(block_tables, cu_seqlens_k, block_size)?;
        PLAN_CACHE.with(|c| {
            if let Some(ref mut pc) = *c.borrow_mut() {
                pc.paged_metadata = Some(m.clone());
            }
        });
        m
    };

    if max_seqlen_q == 1 {
        // Decode: dedicated Q=1 kernel — faster than prefill kernel on all archs.
        paged_decode(q, key_cache, value_cache,
                     &meta.indptr_gpu, &meta.indices_gpu, &meta.last_page_len_gpu,
                     block_size, num_kv_heads, head_dim, softmax_scale)
    } else {
        // Prefill: varlen paged prefill kernel for Q>1.
        paged_prefill_fast(q, key_cache, value_cache, cu_seqlens_q, &meta,
                           block_size, num_kv_heads, head_dim, softmax_scale)
    }
}

// ── Ragged prefill (plan + ragged_run) ──────────────────────────────

#[allow(clippy::too_many_arguments)]
fn ragged_prefill(
    q: &Tensor, k: &Tensor, v: &Tensor,
    cu_seqlens_q: &Tensor, cu_seqlens_k: &Tensor,
    _max_seqlen_q: usize, softmax_scale: f32,
    mask_mode: MaskMode, window_left: i64,
) -> Result<Tensor> {
    let reg = registry();
    let (total_q, num_qo_heads, head_dim) = q.shape().dims3()?;
    let (_, num_kv_heads, _) = k.shape().dims3()?;
    let batch_size = cu_seqlens_q.dim(0)? - 1;

    // SM90+ uses FA3 (Hopper TMA kernels), SM80+ uses FA2.
    // FA3 supports head_dim up to 256; fall back to FA2 for larger (e.g. 512).
    let backend = reg.default_backend();
    let use_swa = window_left >= 0;
    let prefill_key = |b| PrefillKey {
        dtype: dtype_to_kernel(q.dtype()),
        head_dim_qk: head_dim as u32, head_dim_vo: head_dim as u32,
        sliding_window: use_swa, logits_soft_cap: false,
        backend: b,
    };
    let (variant, backend) = reg.get_prefill(&prefill_key(backend))
        .map(|v| (v, backend))
        .or_else(|| {
            // FA3 doesn't have this variant — try FA2
            let fb = prelude_flashinfer::Backend::FA2;
            reg.get_prefill(&prefill_key(fb)).map(|v| (v, fb))
        })
        .ok_or_else(|| prelude_core::tensor::Error::Msg(format!("FlashInfer: no prefill variant for head_dim={head_dim}")))?;

    let stream = cb::tensor_stream(q)?;
    let did = device_id(q.device());
    let out = Tensor::zeros(q.shape(), q.dtype(), q.device())?;
    let raw_stream = stream.cu_stream() as *mut c_void;

    let ws_guard = get_workspace(did)?;
    let ws = ws_guard.as_ref().unwrap();

    let q_ptr = cuda_ptr!(q, bf16, &stream);
    let k_ptr = cuda_ptr!(k, bf16, &stream);
    let v_ptr = cuda_ptr!(v, bf16, &stream);
    let o_ptr = cuda_ptr!(&out, bf16, &stream);
    let cu_q_gpu = cuda_ptr!(cu_seqlens_q, u32, &stream);
    let cu_k_gpu = cuda_ptr!(cu_seqlens_k, u32, &stream);

    let fws: [i64; 1] = [FLOAT_WS_BYTES as i64];
    let iws: [i64; 1] = [INT_WS_BYTES as i64];
    let cus: [i64; 1] = [(batch_size + 1) as i64];
    let qs: [i64; 3] = [total_q as i64, num_qo_heads as i64, head_dim as i64];
    let ks: [i64; 3] = [k.dim(0)? as i64, num_kv_heads as i64, head_dim as i64];

    let fws_st = contiguous_strides(&fws);
    let iws_st = contiguous_strides(&iws);
    let cus_st = contiguous_strides(&cus);
    let qs_st = contiguous_strides(&qs);
    let ks_st = contiguous_strides(&ks);

    let dl_fws = make_gpu_dl(ws.float_ws, did, U8_DT, &fws, &fws_st);
    let dl_iws = make_gpu_dl(ws.int_ws, did, U8_DT, &iws, &iws_st);
    // GPU DLTensors for run
    let dl_cuq_gpu = make_gpu_dl(cu_q_gpu, did, I32_DT, &cus, &cus_st);
    let dl_cuk_gpu = make_gpu_dl(cu_k_gpu, did, I32_DT, &cus, &cus_st);
    let dl_q = make_gpu_dl(q_ptr, did, BF16_DT, &qs, &qs_st);
    let dl_k = make_gpu_dl(k_ptr, did, BF16_DT, &ks, &ks_st);
    let dl_v = make_gpu_dl(v_ptr, did, BF16_DT, &ks, &ks_st);
    let dl_o = make_gpu_dl(o_ptr, did, BF16_DT, &qs, &qs_st);

    reg.set_stream(did, raw_stream);

    let is_fa3 = matches!(backend, prelude_flashinfer::Backend::FA3);

    // Check plan cache — skip expensive plan + GPU→CPU copies if cached
    let cached = PLAN_CACHE.with(|c| c.borrow().as_ref().and_then(|pc| pc.ragged_plan));

    let plan_info = if let Some(pi) = cached {
        pi
    } else {
        // Plan needs CPU tensors for indptrs/kv_lens (it reads them on the host).
        let cu_q_cpu: Vec<i32> = cu_seqlens_q.to_vec1::<u32>()?.iter().map(|&v| v as i32).collect();
        let cu_k_cpu: Vec<i32> = cu_seqlens_k.to_vec1::<u32>()?.iter().map(|&v| v as i32).collect();
        let kvl_cpu: Vec<i32> = (0..batch_size).map(|i| cu_k_cpu[i + 1] - cu_k_cpu[i]).collect();

        let pws: [i64; 1] = [PINNED_WS_BYTES as i64];
        let kvls: [i64; 1] = [batch_size as i64];
        let pws_st = contiguous_strides(&pws);
        let kvls_st = contiguous_strides(&kvls);

        let dl_pws = make_cpu_dl(ws.pinned_ws, U8_DT, &pws, &pws_st);
        let dl_cuq_cpu = make_cpu_dl(cu_q_cpu.as_ptr() as *mut c_void, I32_DT, &cus, &cus_st);
        let dl_cuk_cpu = make_cpu_dl(cu_k_cpu.as_ptr() as *mut c_void, I32_DT, &cus, &cus_st);
        let dl_kvl_cpu = make_cpu_dl(kvl_cpu.as_ptr() as *mut c_void, I32_DT, &kvls, &kvls_st);

        // Plan: FA3 has 16 args (no fixed_split_size/disable_split_kv/num_colocated_ctas)
        let mut plan_args = vec![
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_cuq_cpu), TVMFFIAny::dltensor(&dl_cuk_cpu), TVMFFIAny::dltensor(&dl_kvl_cpu),
            TVMFFIAny::int64(total_q as i64), TVMFFIAny::int64(batch_size as i64),
            TVMFFIAny::int64(num_qo_heads as i64), TVMFFIAny::int64(num_kv_heads as i64),
            TVMFFIAny::int64(1), // page_size (ragged)
            TVMFFIAny::bool_val(false), // enable_cuda_graph
            TVMFFIAny::int64(head_dim as i64), TVMFFIAny::int64(head_dim as i64),
            TVMFFIAny::bool_val(mask_mode == MaskMode::Causal),
            TVMFFIAny::int64(window_left),
        ];
        append_fa2_plan_tail(&mut plan_args, is_fa3);

        let pi = unsafe {
            reg.call(variant.plan, &plan_args)
                .map_err(|e| prelude_core::tensor::Error::Msg(format!("FlashInfer prefill plan: {e}")))?
        };

        // Cache the plan for subsequent layers
        PLAN_CACHE.with(|c| {
            if let Some(ref mut pc) = *c.borrow_mut() {
                pc.ragged_plan = Some(pi);
            }
        });
        pi
    };

    // Run: FA3 additional params differ from FA2
    let mut run_args = vec![
        TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
        plan_info,
        TVMFFIAny::dltensor(&dl_q), TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
        TVMFFIAny::dltensor(&dl_cuq_gpu), TVMFFIAny::dltensor(&dl_cuk_gpu),
        TVMFFIAny::dltensor(&dl_o), TVMFFIAny::none(), // lse
        TVMFFIAny::int64(mask_mode as i64), TVMFFIAny::int64(KV_LAYOUT_NHD),
        TVMFFIAny::int64(window_left), TVMFFIAny::bool_val(false), // pdl
    ];
    append_prefill_run_tail(&mut run_args, is_fa3, softmax_scale);

    unsafe {
        reg.call(variant.ragged_run, &run_args)
            .map_err(|e| prelude_core::tensor::Error::Msg(format!("FlashInfer ragged_run: {e}")))?;
    }

    drop(ws_guard);
    Ok(out)
}

// ── Shared helpers for FA2/FA3 argument tails ───────────────────────

/// Append FA2-specific trailing plan arguments (not needed for FA3).
fn append_fa2_plan_tail(plan_args: &mut Vec<TVMFFIAny>, is_fa3: bool) {
    if !is_fa3 {
        plan_args.push(TVMFFIAny::int64(-1));        // fixed_split_size
        plan_args.push(TVMFFIAny::bool_val(false));   // disable_split_kv
        plan_args.push(TVMFFIAny::int64(0));           // num_colocated_ctas
    }
}

/// Append FA2/FA3 run-time trailing arguments for prefill/paged_run kernels.
fn append_prefill_run_tail(run_args: &mut Vec<TVMFFIAny>, is_fa3: bool, softmax_scale: f32) {
    if is_fa3 {
        // FA3/SM90: prefix_len, token_pos, max_item_len, scale_v, soft_cap, sm_scale, scale_v_scalar, token_pos_in_items_len
        run_args.extend_from_slice(&[
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::none(),
            TVMFFIAny::float64(0.0),                        // logits_soft_cap
            TVMFFIAny::float64(softmax_scale as f64),        // sm_scale
            TVMFFIAny::float64(1.0),                         // scale_v_scalar
            TVMFFIAny::int64(0),                             // token_pos_in_items_len
        ]);
    } else {
        // FA2: masks, alibi, prefix, token, max, soft_cap, sm_scale, rope_scale, rope_theta, token_len
        run_args.extend_from_slice(&[
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::none(), TVMFFIAny::none(), TVMFFIAny::none(),
            TVMFFIAny::float64(0.0), TVMFFIAny::float64(softmax_scale as f64),
            TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4), // rope defaults
            TVMFFIAny::int64(0),                               // token_pos_in_items_len
        ]);
    }
}

// ── Paged decode (plan + run) ───────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn paged_decode(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    kv_indptr: &Tensor, kv_indices: &Tensor, kv_last_page_len: &Tensor,
    block_size: usize, num_kv_heads: usize, head_dim: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let reg = registry();
    let (batch_size, num_qo_heads, _) = q.shape().dims3()?;

    let variant = reg
        .get_decode(&DecodeKey {
            dtype: dtype_to_kernel(q.dtype()),
            head_dim_qk: head_dim as u32, head_dim_vo: head_dim as u32,
            sliding_window: false, logits_soft_cap: false,
        })
        .ok_or_else(|| prelude_core::tensor::Error::Msg("FlashInfer: no decode variant".into()))?;

    let stream = cb::tensor_stream(q)?;
    let did = device_id(q.device());
    let out = Tensor::zeros(q.shape(), q.dtype(), q.device())?;
    let raw_stream = stream.cu_stream() as *mut c_void;

    let ws_guard = get_workspace(did)?;
    let ws = ws_guard.as_ref().unwrap();

    // Plan needs CPU indptr; run needs GPU indptr/indices/last_page_len.
    let indptr_cpu: Vec<i32> = kv_indptr.to_vec1()?;

    let q_ptr = cuda_ptr!(q, bf16, &stream);
    let k_ptr = cuda_ptr!(key_cache, bf16, &stream);
    let v_ptr = cuda_ptr!(value_cache, bf16, &stream);
    let o_ptr = cuda_ptr!(&out, bf16, &stream);
    let indptr_gpu = cuda_ptr!(kv_indptr, i32, &stream);
    let indices_gpu = cuda_ptr!(kv_indices, i32, &stream);
    let lp_gpu = cuda_ptr!(kv_last_page_len, i32, &stream);

    let fws: [i64; 1] = [FLOAT_WS_BYTES as i64];
    let iws: [i64; 1] = [INT_WS_BYTES as i64];
    let pws: [i64; 1] = [PINNED_WS_BYTES as i64];
    let ip_s: [i64; 1] = [(batch_size + 1) as i64];
    let ix_s: [i64; 1] = [kv_indices.dim(0)? as i64];
    let lp_s: [i64; 1] = [batch_size as i64];
    let qs: [i64; 3] = [batch_size as i64, num_qo_heads as i64, head_dim as i64];
    let kvs: [i64; 4] = [key_cache.dim(0)? as i64, block_size as i64, num_kv_heads as i64, head_dim as i64];
    let es: [i64; 1] = [0]; // empty

    let fws_st = contiguous_strides(&fws);
    let iws_st = contiguous_strides(&iws);
    let pws_st = contiguous_strides(&pws);
    let ip_st = contiguous_strides(&ip_s);
    let ix_st = contiguous_strides(&ix_s);
    let lp_st = contiguous_strides(&lp_s);
    let qs_st = contiguous_strides(&qs);
    let kvs_st = contiguous_strides(&kvs);
    let es_st: [i64; 1] = [1];

    let dl_fws = make_gpu_dl(ws.float_ws, did, U8_DT, &fws, &fws_st);
    let dl_iws = make_gpu_dl(ws.int_ws, did, U8_DT, &iws, &iws_st);
    let dl_pws = make_cpu_dl(ws.pinned_ws, U8_DT, &pws, &pws_st);
    let dl_ip_cpu = make_cpu_dl(indptr_cpu.as_ptr() as *mut c_void, I32_DT, &ip_s, &ip_st);
    let dl_ip_gpu = make_gpu_dl(indptr_gpu, did, I32_DT, &ip_s, &ip_st);
    let dl_ix = make_gpu_dl(indices_gpu, did, I32_DT, &ix_s, &ix_st);
    let dl_lp = make_gpu_dl(lp_gpu, did, I32_DT, &lp_s, &lp_st);
    let dl_q = make_gpu_dl(q_ptr, did, BF16_DT, &qs, &qs_st);
    let dl_k = make_gpu_dl(k_ptr, did, BF16_DT, &kvs, &kvs_st);
    let dl_v = make_gpu_dl(v_ptr, did, BF16_DT, &kvs, &kvs_st);
    let dl_o = make_gpu_dl(o_ptr, did, BF16_DT, &qs, &qs_st);
    let dl_eq = make_gpu_dl(std::ptr::null_mut(), did, BF16_DT, &es, &es_st);
    let dl_ek = make_gpu_dl(std::ptr::null_mut(), did, BF16_DT, &es, &es_st);

    reg.set_stream(did, raw_stream);

    let plan_info = unsafe {
        reg.call(variant.plan, &[
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_ip_cpu), // CPU for plan!
            TVMFFIAny::int64(batch_size as i64),
            TVMFFIAny::int64(num_qo_heads as i64), TVMFFIAny::int64(num_kv_heads as i64),
            TVMFFIAny::int64(block_size as i64),
            TVMFFIAny::bool_val(false), // cuda_graph
            TVMFFIAny::int64(-1), // window_left
            TVMFFIAny::float64(0.0), // logits_soft_cap
            TVMFFIAny::int64(head_dim as i64), TVMFFIAny::int64(head_dim as i64),
            TVMFFIAny::dltensor(&dl_eq), TVMFFIAny::dltensor(&dl_ek),
        ]).map_err(|e| prelude_core::tensor::Error::Msg(format!("FlashInfer decode plan: {e}")))?
    };

    unsafe {
        reg.call(variant.run, &[
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
            plan_info,
            TVMFFIAny::dltensor(&dl_q),
            TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
            TVMFFIAny::dltensor(&dl_ip_gpu), TVMFFIAny::dltensor(&dl_ix), TVMFFIAny::dltensor(&dl_lp),
            TVMFFIAny::dltensor(&dl_o), TVMFFIAny::none(), // lse
            TVMFFIAny::int64(KV_LAYOUT_NHD), TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false),
            // decode additional:
            TVMFFIAny::none(), // alibi
            TVMFFIAny::float64(0.0), TVMFFIAny::float64(softmax_scale as f64),
            TVMFFIAny::float64(1.0), TVMFFIAny::float64(1e4), // rope defaults
        ]).map_err(|e| prelude_core::tensor::Error::Msg(format!("FlashInfer decode run: {e}")))?;
    }

    drop(ws_guard);
    Ok(out)
}

// ── Paged prefill (plan + paged_run) ────────────────────────────────

/// Paged prefill kernel: Q attends to paged KV cache.
///
/// `meta`: optional pre-computed PagedMeta with CPU data (fast path, avoids
/// GPU→CPU round trip). When `None`, reads metadata from raw GPU tensors.
#[allow(clippy::too_many_arguments)]
fn paged_prefill_impl(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    cu_seqlens_q: &Tensor,
    meta: Option<&PagedMeta>,
    kv_indptr: Option<&Tensor>, kv_indices: Option<&Tensor>, kv_last_page_len: Option<&Tensor>,
    block_size: usize, num_kv_heads: usize, head_dim: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let reg = registry();
    let (total_q, num_qo_heads, _) = q.shape().dims3()?;
    let batch_size = cu_seqlens_q.dim(0)? - 1;

    let backend = reg.default_backend();
    let variant = reg
        .get_prefill(&PrefillKey {
            dtype: dtype_to_kernel(q.dtype()),
            head_dim_qk: head_dim as u32, head_dim_vo: head_dim as u32,
            sliding_window: false, logits_soft_cap: false,
            backend,
        })
        .ok_or_else(|| prelude_core::tensor::Error::Msg(format!("FlashInfer: no {backend:?} prefill variant")))?;

    let stream = cb::tensor_stream(q)?;
    let did = device_id(q.device());
    let out = Tensor::zeros(q.shape(), q.dtype(), q.device())?;
    let raw_stream = stream.cu_stream() as *mut c_void;

    let ws_guard = get_workspace(did)?;
    let ws = ws_guard.as_ref().unwrap();

    let q_ptr = cuda_ptr!(q, bf16, &stream);
    let k_ptr = cuda_ptr!(key_cache, bf16, &stream);
    let v_ptr = cuda_ptr!(value_cache, bf16, &stream);
    let o_ptr = cuda_ptr!(&out, bf16, &stream);
    let cuq_gpu = cuda_ptr!(cu_seqlens_q, u32, &stream);

    // Resolve GPU pointers for paged metadata from either PagedMeta or raw tensors
    let (ip_gpu, ix_gpu, lp_gpu, ix_dim0) = if let Some(m) = meta {
        (
            cuda_ptr!(&m.indptr_gpu, i32, &stream),
            cuda_ptr!(&m.indices_gpu, i32, &stream),
            cuda_ptr!(&m.last_page_len_gpu, i32, &stream),
            m.indices_gpu.dim(0)?,
        )
    } else {
        (
            cuda_ptr!(kv_indptr.unwrap(), i32, &stream),
            cuda_ptr!(kv_indices.unwrap(), i32, &stream),
            cuda_ptr!(kv_last_page_len.unwrap(), i32, &stream),
            kv_indices.unwrap().dim(0)?,
        )
    };

    let fws: [i64; 1] = [FLOAT_WS_BYTES as i64];
    let iws: [i64; 1] = [INT_WS_BYTES as i64];
    let cuq_s: [i64; 1] = [(batch_size + 1) as i64];
    let ip_s: [i64; 1] = [(batch_size + 1) as i64];
    let ix_s: [i64; 1] = [ix_dim0 as i64];
    let lp_s: [i64; 1] = [batch_size as i64];
    let qs: [i64; 3] = [total_q as i64, num_qo_heads as i64, head_dim as i64];
    let kvs: [i64; 4] = [key_cache.dim(0)? as i64, block_size as i64, num_kv_heads as i64, head_dim as i64];

    let fws_st = contiguous_strides(&fws);
    let iws_st = contiguous_strides(&iws);
    let cuq_st = contiguous_strides(&cuq_s);
    let ip_st = contiguous_strides(&ip_s);
    let ix_st = contiguous_strides(&ix_s);
    let lp_st = contiguous_strides(&lp_s);
    let qs_st = contiguous_strides(&qs);
    let kvs_st = contiguous_strides(&kvs);

    let dl_fws = make_gpu_dl(ws.float_ws, did, U8_DT, &fws, &fws_st);
    let dl_iws = make_gpu_dl(ws.int_ws, did, U8_DT, &iws, &iws_st);
    let dl_cuq_gpu = make_gpu_dl(cuq_gpu, did, I32_DT, &cuq_s, &cuq_st);
    let dl_ip_gpu = make_gpu_dl(ip_gpu, did, I32_DT, &ip_s, &ip_st);
    let dl_ix = make_gpu_dl(ix_gpu, did, I32_DT, &ix_s, &ix_st);
    let dl_lp = make_gpu_dl(lp_gpu, did, I32_DT, &lp_s, &lp_st);
    let dl_q = make_gpu_dl(q_ptr, did, BF16_DT, &qs, &qs_st);
    let dl_k = make_gpu_dl(k_ptr, did, BF16_DT, &kvs, &kvs_st);
    let dl_v = make_gpu_dl(v_ptr, did, BF16_DT, &kvs, &kvs_st);
    let dl_o = make_gpu_dl(o_ptr, did, BF16_DT, &qs, &qs_st);

    reg.set_stream(did, raw_stream);

    let is_fa3 = matches!(backend, prelude_flashinfer::Backend::FA3);

    // Check plan cache — skip plan + GPU→CPU copies if cached
    let cached = PLAN_CACHE.with(|c| c.borrow().as_ref().and_then(|pc| pc.paged_prefill_plan));

    let plan_info = if let Some(pi) = cached {
        pi
    } else {
        // Plan needs CPU data. Use PagedMeta if available (avoids D2H), else pull from GPU.
        let cuq_cpu: Vec<i32> = cu_seqlens_q.to_vec1::<u32>()?.iter().map(|&v| v as i32).collect();
        let (indptr_cpu, kvl_cpu) = if let Some(m) = meta {
            (m.indptr_cpu.clone(), m.kv_lens_cpu.clone())
        } else {
            let ip: Vec<i32> = kv_indptr.unwrap().to_vec1()?;
            let lp: Vec<i32> = kv_last_page_len.unwrap().to_vec1()?;
            let kvl: Vec<i32> = (0..batch_size)
                .map(|i| {
                    let np = ip[i + 1] - ip[i];
                    if np == 0 { 0 } else { (np - 1) * block_size as i32 + lp[i] }
                })
                .collect();
            (ip, kvl)
        };

        let pws: [i64; 1] = [PINNED_WS_BYTES as i64];
        let kvl_s: [i64; 1] = [batch_size as i64];
        let pws_st = contiguous_strides(&pws);
        let kvl_st = contiguous_strides(&kvl_s);

        let dl_pws = make_cpu_dl(ws.pinned_ws, U8_DT, &pws, &pws_st);
        let dl_cuq_cpu = make_cpu_dl(cuq_cpu.as_ptr() as *mut c_void, I32_DT, &cuq_s, &cuq_st);
        let dl_ip_cpu = make_cpu_dl(indptr_cpu.as_ptr() as *mut c_void, I32_DT, &ip_s, &ip_st);
        let dl_kvl_cpu = make_cpu_dl(kvl_cpu.as_ptr() as *mut c_void, I32_DT, &kvl_s, &kvl_st);

        let mut plan_args = vec![
            TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws), TVMFFIAny::dltensor(&dl_pws),
            TVMFFIAny::dltensor(&dl_cuq_cpu), TVMFFIAny::dltensor(&dl_ip_cpu), TVMFFIAny::dltensor(&dl_kvl_cpu),
            TVMFFIAny::int64(total_q as i64), TVMFFIAny::int64(batch_size as i64),
            TVMFFIAny::int64(num_qo_heads as i64), TVMFFIAny::int64(num_kv_heads as i64),
            TVMFFIAny::int64(block_size as i64),
            TVMFFIAny::bool_val(false),
            TVMFFIAny::int64(head_dim as i64), TVMFFIAny::int64(head_dim as i64),
            TVMFFIAny::bool_val(true), // causal
            TVMFFIAny::int64(-1),      // window_left
        ];
        append_fa2_plan_tail(&mut plan_args, is_fa3);

        let pi = unsafe {
            reg.call(variant.plan, &plan_args)
                .map_err(|e| prelude_core::tensor::Error::Msg(format!("FlashInfer paged prefill plan: {e}")))?
        };

        PLAN_CACHE.with(|c| {
            if let Some(ref mut pc) = *c.borrow_mut() {
                pc.paged_prefill_plan = Some(pi);
            }
        });
        pi
    };

    let mut run_args = vec![
        TVMFFIAny::dltensor(&dl_fws), TVMFFIAny::dltensor(&dl_iws),
        plan_info,
        TVMFFIAny::dltensor(&dl_q),
        TVMFFIAny::dltensor(&dl_k), TVMFFIAny::dltensor(&dl_v),
        TVMFFIAny::dltensor(&dl_cuq_gpu), TVMFFIAny::dltensor(&dl_ip_gpu),
        TVMFFIAny::dltensor(&dl_ix), TVMFFIAny::dltensor(&dl_lp),
        TVMFFIAny::dltensor(&dl_o), TVMFFIAny::none(),
        TVMFFIAny::int64(MaskMode::Causal as i64), TVMFFIAny::int64(KV_LAYOUT_NHD),
        TVMFFIAny::int64(-1), TVMFFIAny::bool_val(false),
    ];
    append_prefill_run_tail(&mut run_args, is_fa3, softmax_scale);

    unsafe {
        reg.call(variant.paged_run, &run_args)
            .map_err(|e| prelude_core::tensor::Error::Msg(format!("FlashInfer paged_run: {e}")))?;
    }

    drop(ws_guard);
    Ok(out)
}

/// Paged prefill fast path: uses pre-computed PagedMeta with CPU data.
#[allow(clippy::too_many_arguments)]
fn paged_prefill_fast(
    q: &Tensor,
    key_cache: &Tensor, value_cache: &Tensor,
    cu_seqlens_q: &Tensor, meta: &PagedMeta,
    block_size: usize, num_kv_heads: usize, head_dim: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    paged_prefill_impl(
        q, key_cache, value_cache, cu_seqlens_q,
        Some(meta), None, None, None,
        block_size, num_kv_heads, head_dim, softmax_scale,
    )
}

// ── Metadata conversion ─────────────────────────────────────────────

/// CPU-side paged metadata computation (shared by both allocation paths).
struct RawPagedMeta {
    indptr: Vec<i32>,
    indices: Vec<i32>,
    last_page: Vec<i32>,
    kv_lens: Vec<i32>,
}

fn compute_paged_metadata_cpu(
    block_tables: &Tensor, cu_seqlens_k: &Tensor, block_size: usize,
) -> Result<RawPagedMeta> {
    let batch_size = cu_seqlens_k.dim(0)? - 1;
    let cu_k: Vec<i32> = cu_seqlens_k.to_vec1::<u32>()?.iter().map(|&v| v as i32).collect();
    let bt: Vec<Vec<i32>> = block_tables
        .to_vec2::<u32>()?
        .iter()
        .map(|row| row.iter().map(|&v| v as i32).collect())
        .collect();
    let bs = block_size as i32;

    let mut indptr = vec![0i32; batch_size + 1];
    let mut indices = Vec::new();
    let mut last_page = Vec::with_capacity(batch_size);
    let mut kv_lens = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let kv_len = cu_k[i + 1] - cu_k[i];
        kv_lens.push(kv_len);
        let np = (kv_len + bs - 1) / bs;
        indptr[i + 1] = indptr[i] + np;
        for &idx in bt[i].iter().take(np as usize) {
            indices.push(idx);
        }
        let rem = kv_len % bs;
        last_page.push(if rem == 0 { bs } else { rem });
    }

    Ok(RawPagedMeta { indptr, indices, last_page, kv_lens })
}

/// Convert engine metadata (block_tables + cu_seqlens_k) to FlashInfer format.
/// Allocates new GPU tensors for the metadata.
fn convert_paged_metadata(
    block_tables: &Tensor, cu_seqlens_k: &Tensor, block_size: usize,
) -> Result<PagedMeta> {
    let raw = compute_paged_metadata_cpu(block_tables, cu_seqlens_k, block_size)?;
    let dev = block_tables.device();
    Ok(PagedMeta {
        indptr_gpu: Tensor::new(raw.indptr.as_slice(), dev)?,
        indices_gpu: Tensor::new(raw.indices.as_slice(), dev)?,
        last_page_len_gpu: Tensor::new(raw.last_page.as_slice(), dev)?,
        indptr_cpu: raw.indptr,
        kv_lens_cpu: raw.kv_lens,
        last_page_len_cpu: raw.last_page,
    })
}

/// Like `convert_paged_metadata` but writes GPU data into pre-allocated tensors
/// via `cudaMemcpyAsync` instead of creating new allocations.
/// Used for CUDA graph address stability.
fn convert_paged_metadata_into(
    block_tables: &Tensor, cu_seqlens_k: &Tensor, block_size: usize,
    fi_indptr: &Tensor, fi_indices: &Tensor, fi_last_page_len: &Tensor,
    stream: &std::sync::Arc<CudaStream>,
    raw_stream: *mut c_void,
) -> Result<PagedMeta> {
    let raw = compute_paged_metadata_cpu(block_tables, cu_seqlens_k, block_size)?;

    // Write CPU data into pre-allocated GPU tensors via cudaMemcpyAsync on the
    // same stream as the graph, ensuring correct ordering.
    unsafe {
        let ip_ptr = cuda_ptr!(fi_indptr, i32, stream);
        let rc = cudaMemcpyAsync(
            ip_ptr, raw.indptr.as_ptr() as *const c_void,
            raw.indptr.len() * std::mem::size_of::<i32>(), CUDA_MEMCPY_HOST_TO_DEVICE, raw_stream,
        );
        if rc != 0 { bail!("cudaMemcpyAsync indptr failed: {rc}"); }

        if !raw.indices.is_empty() {
            let ix_ptr = cuda_ptr!(fi_indices, i32, stream);
            let rc = cudaMemcpyAsync(
                ix_ptr, raw.indices.as_ptr() as *const c_void,
                raw.indices.len() * std::mem::size_of::<i32>(), CUDA_MEMCPY_HOST_TO_DEVICE, raw_stream,
            );
            if rc != 0 { bail!("cudaMemcpyAsync indices failed: {rc}"); }
        }

        let lp_ptr = cuda_ptr!(fi_last_page_len, i32, stream);
        let rc = cudaMemcpyAsync(
            lp_ptr, raw.last_page.as_ptr() as *const c_void,
            raw.last_page.len() * std::mem::size_of::<i32>(), CUDA_MEMCPY_HOST_TO_DEVICE, raw_stream,
        );
        if rc != 0 { bail!("cudaMemcpyAsync last_page_len failed: {rc}"); }
    }

    Ok(PagedMeta {
        indptr_gpu: fi_indptr.clone(),
        indices_gpu: fi_indices.clone(),
        last_page_len_gpu: fi_last_page_len.clone(),
        indptr_cpu: raw.indptr,
        kv_lens_cpu: raw.kv_lens,
        last_page_len_cpu: raw.last_page,
    })
}

// ── Utility kernels (activation fusions) ─────────────────────────────

/// `silu(gate) * up` on a concatenated `[tokens, 2*dim]` BF16 tensor.
/// Returns `[tokens, dim]`. Uses FlashInfer's `silu_and_mul` kernel which
/// splits the last dimension internally, avoiding narrow+contiguous copy.
pub fn silu_and_mul(gate_up: &Tensor) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::WrapErr;

    let (gu_storage, gu_layout) = gate_up.storage_and_layout();
    let gu_cuda = match &*gu_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("silu_and_mul: requires CUDA"),
    };
    if gu_cuda.dtype() != DType::BF16 {
        bail!("silu_and_mul: requires BF16");
    }

    let dims = gu_layout.dims();
    if dims.is_empty() || dims[dims.len() - 1] % 2 != 0 {
        bail!("silu_and_mul: last dim must be even, got {:?}", dims);
    }
    let half_dim = dims[dims.len() - 1] / 2;
    let tokens: usize = dims[..dims.len() - 1].iter().product();
    let out_elems = tokens * half_dim;

    let dev = gu_cuda.device().clone();
    let stream = dev.cuda_stream();
    let did = device_id(gate_up.device());

    let reg = registry();
    let silu_fn = reg.get_utility("silu_and_mul")
        .ok_or_else(|| candle_core::Error::Msg("FlashInfer: silu_and_mul not found".into()))?;

    // Input DLTensor: [tokens, 2*dim]
    let gu_slice = gu_cuda.as_cuda_slice::<bf16>()?.slice(gu_layout.start_offset()..);
    let gu_ptr = gu_slice.device_ptr(&stream).0 as *mut c_void;
    let in_shape = [tokens as i64, (half_dim * 2) as i64];
    let in_strides = contiguous_strides(&in_shape);
    let dl_in = make_gpu_dl(gu_ptr, did, BF16_DT, &in_shape, &in_strides);

    // Output: [tokens, dim]
    let out = unsafe { dev.alloc::<bf16>(out_elems) }?;
    let out_ptr = out.device_ptr(&stream).0 as *mut c_void;
    let out_shape = [tokens as i64, half_dim as i64];
    let out_strides = contiguous_strides(&out_shape);
    let dl_out = make_gpu_dl(out_ptr, did, BF16_DT, &out_shape, &out_strides);

    let raw_stream = unsafe { stream.cu_stream() } as *mut c_void;
    reg.set_stream(did, raw_stream);

    let args = [
        TVMFFIAny::dltensor(&dl_out),
        TVMFFIAny::dltensor(&dl_in),
        TVMFFIAny::bool_val(false), // enable_pdl
    ];
    unsafe {
        reg.call(silu_fn, &args)
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer silu_and_mul: {e}")))?;
    }

    drop(gu_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        (tokens, half_dim),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// FlashInfer fused add + RMSNorm (in-place).
/// Computes: input += residual; input = rmsnorm(input, weight, eps)
/// Both input and residual are modified in-place.
pub fn fi_fused_add_rmsnorm(input: &Tensor, residual: &Tensor, weight: &Tensor, eps: f64) -> Result<()> {
    use candle_core::backend::BackendStorage;

    let (in_storage, in_layout) = input.storage_and_layout();
    let (res_storage, res_layout) = residual.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();

    let in_cuda = match &*in_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("fi_fused_add_rmsnorm: requires CUDA"),
    };
    let res_cuda = match &*res_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("fi_fused_add_rmsnorm: requires CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("fi_fused_add_rmsnorm: requires CUDA"),
    };

    if in_cuda.dtype() != DType::BF16 {
        bail!("fi_fused_add_rmsnorm: requires BF16");
    }

    let dims = in_layout.dims();
    let hidden = *dims.last().unwrap();
    let tokens: usize = dims[..dims.len() - 1].iter().product();

    let dev = in_cuda.device().clone();
    let stream = dev.cuda_stream();
    let did = device_id(input.device());

    let reg = registry();
    let fused_fn = reg.get_utility("fused_add_rmsnorm")
        .ok_or_else(|| candle_core::Error::Msg("FlashInfer: fused_add_rmsnorm not found".into()))?;

    let in_slice = in_cuda.as_cuda_slice::<bf16>()?.slice(in_layout.start_offset()..);
    let in_ptr = in_slice.device_ptr(&stream).0 as *mut c_void;
    let res_slice = res_cuda.as_cuda_slice::<bf16>()?.slice(res_layout.start_offset()..);
    let res_ptr = res_slice.device_ptr(&stream).0 as *mut c_void;
    let w_slice = w_cuda.as_cuda_slice::<bf16>()?.slice(w_layout.start_offset()..);
    let w_ptr = w_slice.device_ptr(&stream).0 as *mut c_void;

    let in_shape = [tokens as i64, hidden as i64];
    let in_strides = contiguous_strides(&in_shape);
    let w_shape = [hidden as i64];
    let w_strides = [1i64];

    let dl_in = make_gpu_dl(in_ptr, did, BF16_DT, &in_shape, &in_strides);
    let dl_res = make_gpu_dl(res_ptr, did, BF16_DT, &in_shape, &in_strides);
    let dl_w = make_gpu_dl(w_ptr, did, BF16_DT, &w_shape, &w_strides);

    let raw_stream = unsafe { stream.cu_stream() } as *mut c_void;
    reg.set_stream(did, raw_stream);

    let args = [
        TVMFFIAny::dltensor(&dl_in),
        TVMFFIAny::dltensor(&dl_res),
        TVMFFIAny::dltensor(&dl_w),
        TVMFFIAny::float64(eps),
        TVMFFIAny::bool_val(false), // enable_pdl
    ];
    unsafe {
        reg.call(fused_fn, &args)
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer fused_add_rmsnorm: {e}")))?;
    }

    drop(in_storage);
    drop(res_storage);
    drop(w_storage);
    Ok(())
}

// ── Integration test: tensors + real CUDA kernels ────────────
#[cfg(test)]
mod tests {
    use super::*;
    use prelude_core::tensor::{DType, Device, Tensor};

    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// CPU reference: GQA-aware scaled dot-product attention.
    fn cpu_attention_ref(
        q: &[f32], k: &[f32], v: &[f32],
        num_qo: usize, num_kv: usize, head_dim: usize, seq_len: usize, scale: f32,
    ) -> Vec<f32> {
        let gqa = num_qo / num_kv;
        let mut out = vec![0.0f32; num_qo * head_dim];
        for h in 0..num_qo {
            let kh = h / gqa;
            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[h * head_dim + d]
                        * k[t * num_kv * head_dim + kh * head_dim + d];
                }
                scores[t] = dot * scale;
            }
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores { *s = (*s - max_s).exp(); sum += *s; }
            for s in &mut scores { *s /= sum; }
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..seq_len {
                    val += scores[t] * v[t * num_kv * head_dim + kh * head_dim + d];
                }
                out[h * head_dim + d] = val;
            }
        }
        out
    }

    /// Full engine-path test using tensors + real CUDA reshape_and_cache_flash.
    ///
    /// Run: cargo test -p prelude-core --features flashinfer -- flashinfer_paged_decode_engine_path --nocapture
    #[test]
    fn flashinfer_paged_decode_engine_path() {
        crate::register();
        let dev = Device::Cuda(0);

        let num_qo: usize = 16;
        let num_kv: usize = 8;
        let head_dim: usize = 128;
        let block_size: usize = 128; // engine default with flashinfer
        let total_blocks: usize = 8;
        let sm_scale = 1.0 / (head_dim as f32).sqrt();

        // Test case: batch=2, different KV lengths, non-contiguous blocks
        // seq0: 50 tokens in block 3 (partial)
        // seq1: 200 tokens in blocks 5,1 (spans 2 blocks)
        let kv_lens: Vec<usize> = vec![50, 200];
        let block_tables: Vec<Vec<u32>> = vec![vec![3], vec![5, 1]];
        let batch_size = kv_lens.len();
        let kv_stride = num_kv * head_dim;

        // ── Allocate paged KV cache: [total_blocks, block_size, num_kv, head_dim] ──
        let mut key_cache = Tensor::zeros(
            (total_blocks, block_size, num_kv, head_dim), DType::BF16, &dev,
        ).unwrap();
        let mut value_cache = Tensor::zeros(
            (total_blocks, block_size, num_kv, head_dim), DType::BF16, &dev,
        ).unwrap();

        // ── Generate K/V data and write to cache using real CUDA scatter kernel ──
        let mut all_k_f32: Vec<Vec<f32>> = Vec::new();
        let mut all_v_f32: Vec<Vec<f32>> = Vec::new();

        for (seq_i, &kv_len) in kv_lens.iter().enumerate() {
            let bt = &block_tables[seq_i];

            // Generate deterministic K/V data: [kv_len, num_kv, head_dim]
            // V values are all-positive so the attention output is clearly non-zero.
            let mut k_f32 = vec![0.0f32; kv_len * kv_stride];
            let mut v_f32 = vec![0.0f32; kv_len * kv_stride];
            for i in 0..k_f32.len() {
                k_f32[i] = 0.005 * (((seq_i * 1000 + i) % 11) as f32);
                v_f32[i] = 0.01 * (((seq_i * 1000 + i) % 5) as f32) + 0.01;
            }

            // Build slot_mapping for this sequence (same as BlockManager::slot)
            let slots: Vec<i64> = (0..kv_len)
                .map(|pos| {
                    let block_idx = pos / block_size;
                    let offset = pos % block_size;
                    (bt[block_idx] as i64) * (block_size as i64) + (offset as i64)
                })
                .collect();

            // Create tensors
            let k_t = Tensor::new(&k_f32[..], &dev).unwrap()
                .to_dtype(DType::BF16).unwrap()
                .reshape((kv_len, num_kv, head_dim)).unwrap();
            let v_t = Tensor::new(&v_f32[..], &dev).unwrap()
                .to_dtype(DType::BF16).unwrap()
                .reshape((kv_len, num_kv, head_dim)).unwrap();
            let slot_t = Tensor::new(&slots[..], &dev).unwrap();

            // Call the REAL CUDA scatter kernel
            crate::ops::kv_cache::scatter_kv_cache_flash(
                &k_t, &v_t, &key_cache, &value_cache, &slot_t,
            ).unwrap();

            all_k_f32.push(k_f32);
            all_v_f32.push(v_f32);
        }

        // ── Build engine-format metadata ──
        let mut cu_seqlens_k_vals = vec![0u32];
        let mut cumsum = 0u32;
        for &kv_len in &kv_lens {
            cumsum += kv_len as u32;
            cu_seqlens_k_vals.push(cumsum);
        }

        // Block tables: padded to max blocks per seq
        let max_blocks_per_seq = block_tables.iter().map(|bt| bt.len()).max().unwrap();
        let mut flat_bt = Vec::new();
        for bt in &block_tables {
            flat_bt.extend_from_slice(bt);
            flat_bt.resize(flat_bt.len() + max_blocks_per_seq - bt.len(), 0);
        }

        let cu_seqlens_k_t = Tensor::new(&cu_seqlens_k_vals[..], &dev).unwrap()
            .to_dtype(DType::U32).unwrap();
        let block_tables_t = Tensor::new(&flat_bt[..], &dev).unwrap()
            .to_dtype(DType::U32).unwrap()
            .reshape((batch_size, max_blocks_per_seq)).unwrap();

        // ── Generate Q: [batch_size, num_qo, head_dim] ──
        let qo_stride = num_qo * head_dim;
        let mut q_f32 = vec![0.0f32; batch_size * qo_stride];
        for i in 0..q_f32.len() {
            q_f32[i] = 0.01 * (i as f32 % 7.0);
        }
        let q_t = Tensor::new(&q_f32[..], &dev).unwrap()
            .to_dtype(DType::BF16).unwrap()
            .reshape((batch_size, num_qo, head_dim)).unwrap();

        // ── CPU reference (read data back from GPU cache for accurate comparison) ──
        let key_cache_cpu: Vec<u16> = key_cache.flatten_all().unwrap()
            .to_vec1::<half::bf16>().unwrap()
            .iter().map(|v| v.to_bits()).collect();
        let value_cache_cpu: Vec<u16> = value_cache.flatten_all().unwrap()
            .to_vec1::<half::bf16>().unwrap()
            .iter().map(|v| v.to_bits()).collect();

        let q_f32_rt: Vec<f32> = q_t.to_dtype(DType::F32).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut ref_out = vec![0.0f32; batch_size * qo_stride];
        for (seq_i, &kv_len) in kv_lens.iter().enumerate() {
            let bt = &block_tables[seq_i];
            // Read K/V from paged cache
            let mut k_read = vec![0.0f32; kv_len * kv_stride];
            let mut v_read = vec![0.0f32; kv_len * kv_stride];
            for t in 0..kv_len {
                let block_idx = bt[t / block_size] as usize;
                let block_off = t % block_size;
                let cache_base = (block_idx * block_size + block_off) * kv_stride;
                for i in 0..kv_stride {
                    k_read[t * kv_stride + i] = bf16_to_f32(key_cache_cpu[cache_base + i]);
                    v_read[t * kv_stride + i] = bf16_to_f32(value_cache_cpu[cache_base + i]);
                }
            }
            let q_start = seq_i * qo_stride;
            let seq_ref = cpu_attention_ref(
                &q_f32_rt[q_start..q_start + qo_stride],
                &k_read, &v_read,
                num_qo, num_kv, head_dim, kv_len, sm_scale,
            );
            ref_out[q_start..q_start + qo_stride].copy_from_slice(&seq_ref);
        }

        // ── Call FlashInfer varlen_paged (the actual engine dispatch target) ──
        let cu_seqlens_q_vals: Vec<u32> = (0..=batch_size as u32).collect();
        let cu_seqlens_q_t = Tensor::new(&cu_seqlens_q_vals[..], &dev).unwrap()
            .to_dtype(DType::U32).unwrap();

        let out_t = varlen_paged(
            &q_t,
            &key_cache, &value_cache, &block_tables_t,
            &cu_seqlens_q_t, &cu_seqlens_k_t,
            1, // max_seqlen_q = 1 (decode)
            *kv_lens.iter().max().unwrap(),
            sm_scale,
        ).unwrap();

        // ── Compare ──
        let gpu_out: Vec<f32> = out_t.to_dtype(DType::F32).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for seq_i in 0..batch_size {
            let base = seq_i * qo_stride;
            let mut max_abs_err = 0.0f32;
            let gpu_sum: f32 = (0..qo_stride).map(|i| gpu_out[base + i].abs()).sum();
            for i in 0..qo_stride {
                let err = (gpu_out[base + i] - ref_out[base + i]).abs();
                if err > max_abs_err { max_abs_err = err; }
            }
            eprintln!("  seq {seq_i}: max_abs_err={max_abs_err:.6}, gpu_sum={gpu_sum:.4}");
            assert!(gpu_sum > 0.001, "seq {seq_i}: GPU output is all zeros");
            assert!(
                max_abs_err < 0.01,
                "seq {seq_i}: max_abs_err={max_abs_err:.6} (threshold 0.01)"
            );
        }
    }

    /// Multi-layer simulation: 28 layers each calling reshape_and_cache + varlen_paged.
    /// Tests workspace reuse across layers (same global workspace buffer).
    ///
    /// Run: cargo test -p prelude-core --lib --features flashinfer -- flashinfer_multi_layer_decode --nocapture
    #[test]
    fn flashinfer_multi_layer_decode() {
        crate::register();
        let dev = Device::Cuda(0);

        let num_layers: usize = 28; // Qwen3-0.6B
        let num_qo: usize = 16;
        let num_kv: usize = 8;
        let head_dim: usize = 128;
        let block_size: usize = 128;
        let total_blocks: usize = 8;
        let sm_scale = 1.0 / (head_dim as f32).sqrt();

        let batch_size: usize = 2;
        let kv_lens: Vec<usize> = vec![50, 200];
        let block_tables: Vec<Vec<u32>> = vec![vec![3], vec![5, 1]];
        let kv_stride = num_kv * head_dim;
        let qo_stride = num_qo * head_dim;

        // Allocate per-layer KV caches
        let mut key_caches = Vec::new();
        let mut value_caches = Vec::new();
        for _ in 0..num_layers {
            key_caches.push(Tensor::zeros(
                (total_blocks, block_size, num_kv, head_dim), DType::BF16, &dev,
            ).unwrap());
            value_caches.push(Tensor::zeros(
                (total_blocks, block_size, num_kv, head_dim), DType::BF16, &dev,
            ).unwrap());
        }

        // Metadata tensors (shared across layers, same as engine)
        let mut cu_seqlens_k_vals = vec![0u32];
        let mut cumsum = 0u32;
        for &kv_len in &kv_lens { cumsum += kv_len as u32; cu_seqlens_k_vals.push(cumsum); }
        let cu_seqlens_k_t = Tensor::new(&cu_seqlens_k_vals[..], &dev).unwrap()
            .to_dtype(DType::U32).unwrap();
        let cu_seqlens_q_vals: Vec<u32> = (0..=batch_size as u32).collect();
        let cu_seqlens_q_t = Tensor::new(&cu_seqlens_q_vals[..], &dev).unwrap()
            .to_dtype(DType::U32).unwrap();

        let max_bps = block_tables.iter().map(|bt| bt.len()).max().unwrap();
        let mut flat_bt = Vec::new();
        for bt in &block_tables {
            flat_bt.extend_from_slice(bt);
            flat_bt.resize(flat_bt.len() + max_bps - bt.len(), 0);
        }
        let block_tables_t = Tensor::new(&flat_bt[..], &dev).unwrap()
            .to_dtype(DType::U32).unwrap()
            .reshape((batch_size, max_bps)).unwrap();

        // Enable plan cache (simulates begin_forward/end_forward)
        begin_forward();

        for layer_idx in 0..num_layers {
            // Generate per-layer K/V data and scatter into cache
            for (seq_i, &kv_len) in kv_lens.iter().enumerate() {
                let bt = &block_tables[seq_i];
                let mut k_f32 = vec![0.0f32; kv_len * kv_stride];
                let mut v_f32 = vec![0.0f32; kv_len * kv_stride];
                for i in 0..k_f32.len() {
                    k_f32[i] = (((layer_idx * 10000 + seq_i * 1000 + i) % 23) as f32 - 11.0) * 0.01;
                    v_f32[i] = (((layer_idx * 10000 + seq_i * 1000 + i) % 19) as f32 - 9.0) * 0.01;
                }
                let slots: Vec<i64> = (0..kv_len)
                    .map(|pos| {
                        let bi = pos / block_size;
                        let off = pos % block_size;
                        (bt[bi] as i64) * (block_size as i64) + (off as i64)
                    })
                    .collect();
                let k_t = Tensor::new(&k_f32[..], &dev).unwrap()
                    .to_dtype(DType::BF16).unwrap()
                    .reshape((kv_len, num_kv, head_dim)).unwrap();
                let v_t = Tensor::new(&v_f32[..], &dev).unwrap()
                    .to_dtype(DType::BF16).unwrap()
                    .reshape((kv_len, num_kv, head_dim)).unwrap();
                let slot_t = Tensor::new(&slots[..], &dev).unwrap();
                crate::ops::kv_cache::scatter_kv_cache_flash(
                    &k_t, &v_t, &key_caches[layer_idx], &value_caches[layer_idx], &slot_t,
                ).unwrap();
            }

            // Generate Q for this layer
            let mut q_f32 = vec![0.0f32; batch_size * qo_stride];
            for i in 0..q_f32.len() {
                q_f32[i] = (((layer_idx * 10000 + i) % 17) as f32 - 8.0) * 0.01;
            }
            let q_t = Tensor::new(&q_f32[..], &dev).unwrap()
                .to_dtype(DType::BF16).unwrap()
                .reshape((batch_size, num_qo, head_dim)).unwrap();

            // Call FlashInfer paged decode
            let out_t = varlen_paged(
                &q_t,
                &key_caches[layer_idx], &value_caches[layer_idx], &block_tables_t,
                &cu_seqlens_q_t, &cu_seqlens_k_t,
                1, *kv_lens.iter().max().unwrap(), sm_scale,
            ).unwrap();

            // CPU reference for this layer
            let kc_cpu: Vec<u16> = key_caches[layer_idx].flatten_all().unwrap()
                .to_vec1::<half::bf16>().unwrap().iter().map(|v| v.to_bits()).collect();
            let vc_cpu: Vec<u16> = value_caches[layer_idx].flatten_all().unwrap()
                .to_vec1::<half::bf16>().unwrap().iter().map(|v| v.to_bits()).collect();
            let q_f32_rt: Vec<f32> = q_t.to_dtype(DType::F32).unwrap()
                .flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let gpu_out: Vec<f32> = out_t.to_dtype(DType::F32).unwrap()
                .flatten_all().unwrap().to_vec1::<f32>().unwrap();

            for seq_i in 0..batch_size {
                let bt = &block_tables[seq_i];
                let kv_len = kv_lens[seq_i];
                let mut k_read = vec![0.0f32; kv_len * kv_stride];
                let mut v_read = vec![0.0f32; kv_len * kv_stride];
                for t in 0..kv_len {
                    let bi = bt[t / block_size] as usize;
                    let bo = t % block_size;
                    let cb = (bi * block_size + bo) * kv_stride;
                    for j in 0..kv_stride {
                        k_read[t * kv_stride + j] = bf16_to_f32(kc_cpu[cb + j]);
                        v_read[t * kv_stride + j] = bf16_to_f32(vc_cpu[cb + j]);
                    }
                }
                let q_start = seq_i * qo_stride;
                let ref_out = cpu_attention_ref(
                    &q_f32_rt[q_start..q_start + qo_stride],
                    &k_read, &v_read,
                    num_qo, num_kv, head_dim, kv_len, sm_scale,
                );

                let base = seq_i * qo_stride;
                let mut max_abs_err = 0.0f32;
                let mut any_nonzero = false;
                for i in 0..qo_stride {
                    let g = gpu_out[base + i];
                    let r = ref_out[i];
                    if g.abs() > 1e-10 { any_nonzero = true; }
                    let err = (g - r).abs();
                    if err > max_abs_err { max_abs_err = err; }
                }
                assert!(any_nonzero, "layer {layer_idx} seq {seq_i}: all zeros");
                assert!(
                    max_abs_err < 0.01,
                    "layer {layer_idx} seq {seq_i}: max_abs_err={max_abs_err:.6}"
                );
            }
        }

        end_forward();
        eprintln!("  All {num_layers} layers x {batch_size} seqs: PASS");
    }
}

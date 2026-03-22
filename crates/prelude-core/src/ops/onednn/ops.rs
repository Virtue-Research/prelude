//! Safe wrappers around oneDNN FFI for BF16 and F32 GEMM operations.
//! Only available when the `onednn` feature is enabled.

use candle_core::{DType, Device, Module, Result, Tensor};
use half::bf16;
use std::sync::Arc;

/// Initialize oneDNN engine and stream. Idempotent — safe to call multiple times.
pub fn init() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe { super::ffi::onednn_init() });
}

/// Set the number of threads (no-op with THREADPOOL runtime, kept for API compat).
pub fn set_num_threads(n: i32) {
    unsafe { super::ffi::onednn_set_num_threads(n) }
}

/// Bind threads to CPU cores (no-op with THREADPOOL runtime, kept for API compat).
pub fn bind_threads(cpu_ids: &[usize]) {
    let ids_i32: Vec<i32> = cpu_ids.iter().map(|&id| id as i32).collect();
    unsafe {
        super::ffi::onednn_bind_threads(ids_i32.as_ptr(), ids_i32.len() as i32);
    }
}

// ── BRGeMM micro-kernel packed weight API ─────────────────────────────

/// Check if brgemm ukernel is available on this CPU (cached).
pub fn brgemm_available() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| unsafe { super::ffi::brgemm_available() != 0 })
}

/// Pre-packed weight in VNNI block format for brgemm micro-kernel.
/// Uses the same approach as SGLang: JIT'd oneDNN brgemm + VNNI packed weights.
pub struct BrgemmPackedWeight {
    ptr: *mut std::ffi::c_void,
    pub k: usize,
    pub n: usize,
}

impl std::fmt::Debug for BrgemmPackedWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrgemmPackedWeight")
            .field("k", &self.k)
            .field("n", &self.n)
            .finish()
    }
}

unsafe impl Send for BrgemmPackedWeight {}
unsafe impl Sync for BrgemmPackedWeight {}

impl BrgemmPackedWeight {
    /// Pack a BF16 weight tensor `[N, K]` into VNNI block format for brgemm.
    /// Returns None if brgemm is not available.
    pub fn pack(weight: &Tensor) -> Result<Option<Self>> {
        if !brgemm_available() {
            return Ok(None);
        }
        let weight = weight.contiguous()?;
        if weight.dtype() != DType::BF16 {
            return Ok(None);
        }
        let (n, k) = weight.dims2()?;

        let (w_storage, w_layout) = weight.storage_and_layout();
        let w_data = match &*w_storage {
            candle_core::Storage::Cpu(s) => s.as_slice::<bf16>()?,
            _ => return Ok(None),
        };
        let w_ptr = w_data[w_layout.start_offset()..].as_ptr() as *const std::ffi::c_void;

        let ptr = unsafe { super::ffi::brgemm_bf16_pack(w_ptr, k as i64, n as i64) };
        drop(w_storage);

        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(Self { ptr, k, n }))
    }
}

impl Drop for BrgemmPackedWeight {
    fn drop(&mut self) {
        unsafe { super::ffi::brgemm_bf16_pack_destroy(self.ptr) }
    }
}

// ── F32 packed weight API (oneDNN blocked format) ─────────────────────

/// Pre-packed F32 weight in oneDNN's optimal blocked format.
/// Reorders weight from user layout to oneDNN's internal layout once at model load;
/// subsequent matmuls skip the on-the-fly reorder, improving cache utilization.
pub struct OnednnF32PackedWeight {
    ptr: *mut std::ffi::c_void,
    pub k: usize,
    pub n: usize,
}

impl std::fmt::Debug for OnednnF32PackedWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnednnF32PackedWeight")
            .field("k", &self.k)
            .field("n", &self.n)
            .finish()
    }
}

unsafe impl Send for OnednnF32PackedWeight {}
unsafe impl Sync for OnednnF32PackedWeight {}

impl OnednnF32PackedWeight {
    /// Pack an F32 weight tensor `[N, K]` into oneDNN's optimal blocked format.
    /// `ref_m` is a representative M dimension for the matmul (affects blocking choice).
    pub fn pack(weight: &Tensor, ref_m: usize) -> Result<Option<Self>> {
        let weight = weight.contiguous()?;
        if weight.dtype() != DType::F32 || !weight.device().is_cpu() {
            return Ok(None);
        }
        let (n, k) = weight.dims2()?;

        let (w_storage, w_layout) = weight.storage_and_layout();
        let w_data = match &*w_storage {
            candle_core::Storage::Cpu(s) => s.as_slice::<f32>()?,
            _ => return Ok(None),
        };
        let w_ptr = w_data[w_layout.start_offset()..].as_ptr() as *const std::ffi::c_void;

        let ptr = unsafe {
            super::ffi::onednn_f32_pack_weights(w_ptr, k as i64, n as i64, ref_m as i64)
        };
        drop(w_storage);

        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(Self { ptr, k, n }))
    }

    /// Raw F32 GEMM: output[M, N] = input[M, K] × packed^T, on pre-allocated buffers.
    ///
    /// # Safety
    /// `input` must point to `[m * self.k]` f32 elements,
    /// `output` must point to `[m * self.n]` f32 elements.
    pub unsafe fn forward_raw(&self, input: *const f32, output: *mut f32, m: usize) {
        unsafe {
            super::ffi::onednn_f32_linear_packed(
                input as *const std::ffi::c_void,
                self.ptr,
                output as *mut std::ffi::c_void,
                m as i64,
            );
        }
    }

    /// Execute F32 linear with packed weights: output[M, N] = input[M, K] × packed^T
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.dims();
        let m: usize = dims[..dims.len() - 1].iter().product();
        let input = input.contiguous()?;

        let (in_storage, in_layout) = input.storage_and_layout();
        let in_data = match &*in_storage {
            candle_core::Storage::Cpu(s) => s.as_slice::<f32>()?,
            _ => candle_core::bail!("OnednnF32PackedWeight::forward: CPU tensor required"),
        };
        let in_ptr = in_data[in_layout.start_offset()..].as_ptr() as *const std::ffi::c_void;

        let mut out_buf = vec![0.0f32; m * self.n];
        unsafe {
            super::ffi::onednn_f32_linear_packed(
                in_ptr, self.ptr, out_buf.as_mut_ptr() as *mut std::ffi::c_void, m as i64,
            );
        }

        drop(in_storage);
        let mut shape = dims.to_vec();
        *shape.last_mut().unwrap() = self.n;
        Tensor::from_vec(out_buf, shape.as_slice(), &Device::Cpu)
    }
}

impl Drop for OnednnF32PackedWeight {
    fn drop(&mut self) {
        unsafe { super::ffi::onednn_packed_weights_destroy(self.ptr) }
    }
}

// ── OnednnLinear: drop-in replacement for candle_nn::Linear ──────────

/// Drop-in Linear layer that dispatches BF16/F32 CPU to oneDNN packed GEMM,
/// falling back to candle for other dtypes/devices.
#[derive(Clone, Debug)]
pub struct OnednnLinear {
    candle_linear: candle_nn::Linear,
    brgemm_packed: Option<Arc<BrgemmPackedWeight>>,
    f32_packed: Option<Arc<OnednnF32PackedWeight>>,
}

impl OnednnLinear {
    /// Wrap a `candle_nn::Linear`. If BF16 or F32 CPU, pre-packs weights for oneDNN.
    /// Also packs brgemm VNNI weights and AMX VNNI weights if available.
    pub fn new(linear: candle_nn::Linear) -> Result<Self> {
        init(); // ensure oneDNN engine/stream exist
        let w = linear.weight();

        // Pack brgemm VNNI weights for BF16 GEMM (preferred over custom AMX)
        let brgemm_packed = if w.device().is_cpu() && w.dtype() == DType::BF16 {
            match BrgemmPackedWeight::pack(w) {
                Ok(Some(bp)) => {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static LOGGED: AtomicBool = AtomicBool::new(false);
                    if !LOGGED.swap(true, Ordering::Relaxed) {
                        let (n, k) = w.dims2().unwrap_or((0, 0));
                        tracing::info!(
                            "brgemm GEMM: packing weights (K={k}, N={n})"
                        );
                    }
                    Some(Arc::new(bp))
                }
                Ok(None) => None,
                Err(e) => {
                    tracing::debug!("brgemm weight packing failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Pack F32 weights into oneDNN's optimal blocked format (done once at model load)
        let f32_packed = if w.device().is_cpu() && w.dtype() == DType::F32 {
            match OnednnF32PackedWeight::pack(w, 1) {
                Ok(Some(pw)) => {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static F32_LOGGED: AtomicBool = AtomicBool::new(false);
                    if !F32_LOGGED.swap(true, Ordering::Relaxed) {
                        let (n, k) = w.dims2().unwrap_or((0, 0));
                        tracing::info!(
                            "oneDNN F32: packing weights (K={k}, N={n})"
                        );
                    }
                    Some(Arc::new(pw))
                }
                Ok(None) => None,
                Err(e) => {
                    tracing::debug!("F32 weight packing failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            candle_linear: linear,
            brgemm_packed,
            f32_packed,
        })
    }

    /// Access the underlying weight tensor.
    pub fn weight(&self) -> &Tensor {
        self.candle_linear.weight()
    }

    /// Access the brgemm packed weight (if available) for fused operations.
    pub fn brgemm_weight(&self) -> Option<&BrgemmPackedWeight> {
        self.brgemm_packed.as_deref()
    }

    /// Access the F32 packed weight (if available) for raw forward path.
    pub fn f32_packed_weight(&self) -> Option<&OnednnF32PackedWeight> {
        self.f32_packed.as_deref()
    }
}

impl Module for OnednnLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BF16 CPU: use brgemm or AMX packed GEMM
        if x.device().is_cpu() && x.dtype() == DType::BF16 {
            let dims = x.dims();
            let (flat, m) = if dims.len() == 3 {
                let (b, s, h) = x.dims3()?;
                (x.reshape((b * s, h))?, b * s)
            } else {
                let m: usize = dims[..dims.len() - 1].iter().product();
                (x.contiguous()?, m)
            };

            let out = if let Some(ref brg) = self.brgemm_packed {
                brgemm_gemm_forward(&flat, brg, m, brg.k, brg.n)?
            } else {
                let w = self.candle_linear.weight();
                let (n, k) = w.dims2()?;
                cpu_gemm_forward(&flat, w, m, k, n)?
            };

            let n = out.dim(out.dims().len() - 1)?;
            if dims.len() == 3 {
                return out.reshape((dims[0], dims[1], n));
            }
            return Ok(out);
        }
        // F32 CPU: use packed weights if available, otherwise unpacked oneDNN matmul
        if x.device().is_cpu() && x.dtype() == DType::F32 {
            if let Some(ref pw) = self.f32_packed {
                return pw.forward(x);
            }

            let dims = x.dims();
            let (flat, m) = if dims.len() == 3 {
                let (b, s, h) = x.dims3()?;
                (x.reshape((b * s, h))?, b * s)
            } else {
                let m: usize = dims[..dims.len() - 1].iter().product();
                (x.contiguous()?, m)
            };

            let w = self.candle_linear.weight();
            let (n, k) = w.dims2()?;
            let out = f32_linear_forward(&flat, w, m, k, n)?;

            if dims.len() == 3 {
                return out.reshape((dims[0], dims[1], n));
            }
            return Ok(out);
        }

        self.candle_linear.forward(x)
    }
}

/// Run oneDNN F32 linear: input [M, K] × weight [N, K]^T → output [M, N]
fn f32_linear_forward(input: &Tensor, weight: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    let input = input.contiguous()?;
    let weight = weight.contiguous()?;

    let (in_storage, in_layout) = input.storage_and_layout();
    let in_data = match &*in_storage {
        candle_core::Storage::Cpu(s) => s.as_slice::<f32>()?,
        _ => candle_core::bail!("f32_linear_forward: CPU tensor required"),
    };
    let (w_storage, w_layout) = weight.storage_and_layout();
    let w_data = match &*w_storage {
        candle_core::Storage::Cpu(s) => s.as_slice::<f32>()?,
        _ => candle_core::bail!("f32_linear_forward: CPU tensor required"),
    };

    let in_ptr = in_data[in_layout.start_offset()..].as_ptr() as *const std::ffi::c_void;
    let w_ptr = w_data[w_layout.start_offset()..].as_ptr() as *const std::ffi::c_void;

    let mut out_buf = vec![0.0f32; m * n];
    unsafe {
        super::ffi::onednn_f32_linear(
            in_ptr, w_ptr, out_buf.as_mut_ptr() as *mut std::ffi::c_void,
            m as i64, k as i64, n as i64,
        );
    }

    drop(in_storage);
    drop(w_storage);
    Tensor::from_vec(out_buf, &[m, n], &Device::Cpu)
}

/// Run brgemm BF16 GEMM with VNNI-packed weights (oneDNN micro-kernel).
/// Input: [M, K], Output: [M, N]
fn brgemm_gemm_forward(input: &Tensor, brg: &BrgemmPackedWeight, m: usize, k: usize, n: usize) -> Result<Tensor> {
    brgemm_gemm_forward_pub(input, brg, m, k, n)
}

/// Context passed to spinning pool workers for brgemm dispatch.
#[repr(C)]
struct BrgemmDispatchCtx {
    input_ptr: usize,
    brg_ptr: usize,
    output_ptr: usize,
    m: i64,
    n_total: usize,
    block_n: usize,
    n_blocks: usize,
}

/// Worker function for spinning pool: each thread computes its N-block range.
unsafe fn brgemm_pool_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
    unsafe {
        let ctx = &*(ctx_raw as *const BrgemmDispatchCtx);
        let blocks_per_thread = (ctx.n_blocks + n_threads - 1) / n_threads;
        let blk_start = tid * blocks_per_thread;
        let blk_end = ((tid + 1) * blocks_per_thread).min(ctx.n_blocks);
        let n_start = blk_start * ctx.block_n;
        let n_end = (blk_end * ctx.block_n).min(ctx.n_total);
        if n_start >= n_end {
            return;
        }

        super::ffi::brgemm_bf16_linear(
            ctx.input_ptr as *const std::ffi::c_void,
            ctx.brg_ptr as *mut std::ffi::c_void,
            ctx.output_ptr as *mut std::ffi::c_void,
            ctx.m,
            ctx.n_total as i64,
            n_start as i64,
            n_end as i64,
        );
    }
}

// ── Thread-local output buffer reuse (P1 optimization) ──────────────
// Avoids per-GEMM heap allocation by reusing the same buffer.
// SGLang gets this for free via jemalloc's thread-local cache returning
// the same virtual address. We make it explicit.
use std::cell::RefCell;
thread_local! {
    static BRGEMM_OUT_BUF: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
}

// ── 2D (M×N) tiling for brgemm dispatch ──────────────────────────────
// At large M, 1D (N-only) tiling forces each thread to read the entire A
// matrix (M×K), which exceeds L2 cache. 2D tiling splits work along both
// M and N dimensions so each thread's A working set is BLOCK_M_2D × K,
// fitting in L2 and reducing total DRAM bandwidth by ~nth_m×.

/// M-block size for 2D tiling. Matches SGLang's block_size_m() = 2 * TILE_M = 32.
/// A tile = 32 × K × 2 bytes. For K=896: 56KB, fits L2 cache.
const BLOCK_M_2D: usize = 32;

/// Compute 2D thread grid: nth_m × nth_n = n_threads.
/// Uses SGLang's square-preference heuristic: nth_m ≈ ceil(sqrt((m_blocks/n_blocks) * nth)).
fn compute_thread_grid(m_blocks: usize, n_blocks: usize, n_threads: usize) -> (usize, usize) {
    if n_threads <= 1 || m_blocks <= 1 {
        return (1, n_threads.max(1));
    }
    let r = m_blocks as f32 / n_blocks as f32;
    let mut nth_m = (r * n_threads as f32).sqrt().ceil() as usize;
    nth_m = nth_m.max(1).min(n_threads).min(m_blocks);
    // Find nearest factor of n_threads ≤ nth_m
    while nth_m > 1 && n_threads % nth_m != 0 {
        nth_m -= 1;
    }
    let nth_n = n_threads / nth_m;
    (nth_m, nth_n)
}

/// Context for 2D brgemm dispatch via spinning pool.
#[repr(C)]
struct BrgemmDispatch2dCtx {
    input_ptr: usize,
    brg_ptr: usize,
    output_ptr: usize,
    k: usize,
    m_total: usize,
    n_total: usize,
    block_m: usize,
    block_n: usize,
    m_blocks: usize,
    n_blocks: usize,
}

/// Worker for 2D dispatch: each thread handles a tile of M-blocks × N-blocks.
/// M-outer loop keeps A tile (BLOCK_M × K) in L2 across N-block sweeps.
unsafe fn brgemm_pool_work_2d(tid: usize, n_threads: usize, ctx_raw: *const u8) {
    unsafe {
        let ctx = &*(ctx_raw as *const BrgemmDispatch2dCtx);
        let (nth_m, nth_n) = compute_thread_grid(ctx.m_blocks, ctx.n_blocks, n_threads);
        let ith_m = tid / nth_n;
        let ith_n = tid % nth_n;

        let mb_per = (ctx.m_blocks + nth_m - 1) / nth_m;
        let mb_start = ith_m * mb_per;
        let mb_end = (mb_start + mb_per).min(ctx.m_blocks);

        let nb_per = (ctx.n_blocks + nth_n - 1) / nth_n;
        let nb_start = ith_n * nb_per;
        let nb_end = (nb_start + nb_per).min(ctx.n_blocks);

        let n_col_start = nb_start * ctx.block_n;
        let n_col_end = (nb_end * ctx.block_n).min(ctx.n_total);
        if n_col_start >= n_col_end {
            return;
        }

        for mb in mb_start..mb_end {
            let m_off = mb * ctx.block_m;
            let m_size = ctx.block_m.min(ctx.m_total - m_off);

            super::ffi::brgemm_bf16_linear(
                (ctx.input_ptr + m_off * ctx.k * 2) as *const std::ffi::c_void,
                ctx.brg_ptr as *mut std::ffi::c_void,
                (ctx.output_ptr + m_off * ctx.n_total * 2) as *mut std::ffi::c_void,
                m_size as i64,
                ctx.n_total as i64,
                n_col_start as i64,
                n_col_end as i64,
            );
        }
    }
}

/// Context for 2D fused brgemm gate_up + SiLU×Mul dispatch.
#[repr(C)]
struct BrgemmFusedSilu2dCtx {
    input_ptr: usize,
    brg_ptr: usize,
    output_ptr: usize,
    k: usize,
    m_total: usize,
    dim: usize, // intermediate_size (output stride)
    block_m: usize,
    block_n: usize,
    m_blocks: usize,
    n_blocks: usize, // blocks in the gate half only
}

/// Worker for 2D fused SiLU dispatch.
unsafe fn brgemm_fused_silu_work_2d(tid: usize, n_threads: usize, ctx_raw: *const u8) {
    unsafe {
        let ctx = &*(ctx_raw as *const BrgemmFusedSilu2dCtx);
        let (nth_m, nth_n) = compute_thread_grid(ctx.m_blocks, ctx.n_blocks, n_threads);
        let ith_m = tid / nth_n;
        let ith_n = tid % nth_n;

        let mb_per = (ctx.m_blocks + nth_m - 1) / nth_m;
        let mb_start = ith_m * mb_per;
        let mb_end = (mb_start + mb_per).min(ctx.m_blocks);

        let nb_per = (ctx.n_blocks + nth_n - 1) / nth_n;
        let nb_start = ith_n * nb_per;
        let nb_end = (nb_start + nb_per).min(ctx.n_blocks);

        let n_col_start = nb_start * ctx.block_n;
        let n_col_end = (nb_end * ctx.block_n).min(ctx.dim);
        if n_col_start >= n_col_end {
            return;
        }

        for mb in mb_start..mb_end {
            let m_off = mb * ctx.block_m;
            let m_size = ctx.block_m.min(ctx.m_total - m_off);

            super::ffi::brgemm_bf16_linear_fused_silu_mul(
                (ctx.input_ptr + m_off * ctx.k * 2) as *const std::ffi::c_void,
                ctx.brg_ptr as *mut std::ffi::c_void,
                (ctx.output_ptr + m_off * ctx.dim * 2) as *mut std::ffi::c_void,
                m_size as i64,
                ctx.dim as i64,
                n_col_start as i64,
                n_col_end as i64,
            );
        }
    }
}

/// Profile flag: set BRGEMM_PROFILE=1 env var to see per-call phase timing.
fn brgemm_profile() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("BRGEMM_PROFILE").ok().map_or(false, |v| v == "1"))
}

/// Public entry point for benchmarking.
pub fn brgemm_gemm_forward_pub(input: &Tensor, brg: &BrgemmPackedWeight, m: usize, k: usize, n: usize) -> Result<Tensor> {
    let profile = brgemm_profile();
    let t0 = if profile { Some(std::time::Instant::now()) } else { None };

    let input = input.contiguous()?;
    let (in_storage, in_layout) = input.storage_and_layout();
    let in_data = match &*in_storage {
        candle_core::Storage::Cpu(s) => s.as_slice::<half::bf16>()?,
        _ => candle_core::bail!("brgemm_gemm_forward: CPU tensor required"),
    };
    let in_ptr = in_data[in_layout.start_offset()..].as_ptr();

    let t1 = t0.map(|_| std::time::Instant::now());

    // Reuse thread-local output buffer (avoids malloc/mmap per call)
    let mut out_buf = BRGEMM_OUT_BUF.with_borrow_mut(|buf| {
        let mut taken = std::mem::take(buf);
        let needed = m * n;
        if taken.capacity() < needed {
            taken.reserve(needed - taken.len());
        }
        unsafe { taken.set_len(needed) };
        taken
    });

    let t2 = t0.map(|_| std::time::Instant::now());

    let block_n = 32_usize; // must match BRGEMM_BLOCK_N in C++
    let n_blocks = (n + block_n - 1) / block_n;

    let pool = crate::ops::cpu::gemm_pool::gemm_pool();

    if m > BLOCK_M_2D {
        // 2D M×N tiling: A tile (BLOCK_M_2D × K) fits L2
        let m_blocks = (m + BLOCK_M_2D - 1) / BLOCK_M_2D;
        let total_blocks = m_blocks * n_blocks;
        let num_threads = pool.num_threads().min(total_blocks).max(1);

        let ctx = BrgemmDispatch2dCtx {
            input_ptr: in_ptr as usize,
            brg_ptr: brg.ptr as usize,
            output_ptr: out_buf.as_mut_ptr() as usize,
            k,
            m_total: m,
            n_total: n,
            block_m: BLOCK_M_2D,
            block_n,
            m_blocks,
            n_blocks,
        };

        unsafe {
            pool.dispatch(
                brgemm_pool_work_2d,
                &ctx as *const BrgemmDispatch2dCtx as *const u8,
                num_threads,
            );
        }
    } else {
        // 1D N-only tiling (small M, A already fits L2)
        let num_threads = pool.num_threads().min(n_blocks).max(1);

        let ctx = BrgemmDispatchCtx {
            input_ptr: in_ptr as usize,
            brg_ptr: brg.ptr as usize,
            output_ptr: out_buf.as_mut_ptr() as usize,
            m: m as i64,
            n_total: n,
            block_n,
            n_blocks,
        };

        unsafe {
            pool.dispatch(
                brgemm_pool_work,
                &ctx as *const BrgemmDispatchCtx as *const u8,
                num_threads,
            );
        }
    }

    let t3 = t0.map(|_| std::time::Instant::now());

    drop(in_storage);

    let bf16_vec: Vec<half::bf16> =
        unsafe { std::mem::transmute::<Vec<u16>, Vec<half::bf16>>(out_buf) };
    let result = Tensor::from_vec(bf16_vec, &[m, n], &Device::Cpu);

    if let (Some(t0v), Some(t1v), Some(t2v), Some(t3v)) = (t0, t1, t2, t3) {
        let t4 = std::time::Instant::now();
        use std::sync::atomic::{AtomicU32, Ordering};
        static CALL_ID: AtomicU32 = AtomicU32::new(0);
        let id = CALL_ID.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "brgemm #{id} M={m} K={k} N={n} extract={}us alloc={}us gemm={}us tensor={}us total={}us",
            t1v.duration_since(t0v).as_micros(),
            t2v.duration_since(t1v).as_micros(),
            t3v.duration_since(t2v).as_micros(),
            t4.duration_since(t3v).as_micros(),
            t4.duration_since(t0v).as_micros(),
        );
    }

    result
}

// ── Fused gate_up GEMM + SiLU×Mul ────────────────────────────────────

/// Context for fused brgemm gate_up + SiLU×Mul dispatch.
#[repr(C)]
struct BrgemmFusedSiluCtx {
    input_ptr: usize,
    brg_ptr: usize,
    output_ptr: usize,
    m: i64,
    dim: usize,  // = intermediate_size (half of gate_up N)
    block_n: usize,
    n_blocks: usize, // blocks in the FIRST half (gate) only
}

/// Worker for fused SiLU: each thread processes gate + up block pairs.
unsafe fn brgemm_fused_silu_work(tid: usize, n_threads: usize, ctx_raw: *const u8) {
    unsafe {
        let ctx = &*(ctx_raw as *const BrgemmFusedSiluCtx);
        let blocks_per_thread = (ctx.n_blocks + n_threads - 1) / n_threads;
        let blk_start = tid * blocks_per_thread;
        let blk_end = ((tid + 1) * blocks_per_thread).min(ctx.n_blocks);
        let n_start = blk_start * ctx.block_n;
        let n_end = (blk_end * ctx.block_n).min(ctx.dim);
        if n_start >= n_end { return; }

        super::ffi::brgemm_bf16_linear_fused_silu_mul(
            ctx.input_ptr as *const std::ffi::c_void,
            ctx.brg_ptr as *mut std::ffi::c_void,
            ctx.output_ptr as *mut std::ffi::c_void,
            ctx.m,
            ctx.dim as i64,
            n_start as i64,
            n_end as i64,
        );
    }
}

/// Fused gate_up GEMM + SiLU×Mul: output[M, dim] = SiLU(gate) * up
/// where gate||up = input × packed_weight^T with N = 2*dim.
///
/// Keeps GEMM F32 accumulators hot in L2 cache when applying SiLU,
/// avoiding the cold-cache read that makes separate silu_mul 2-3x slower.
pub fn brgemm_fused_silu_mul(
    input: &Tensor,
    brg: &BrgemmPackedWeight,
    m: usize,
    k: usize,
    dim: usize, // intermediate_size, brg.n should be 2*dim
) -> Result<Tensor> {
    debug_assert_eq!(brg.n, 2 * dim, "brgemm_fused_silu_mul: packed weight N must be 2*dim");

    let input = input.contiguous()?;
    let (in_storage, in_layout) = input.storage_and_layout();
    let in_data = match &*in_storage {
        candle_core::Storage::Cpu(s) => s.as_slice::<half::bf16>()?,
        _ => candle_core::bail!("brgemm_fused_silu_mul: CPU tensor required"),
    };
    let in_ptr = in_data[in_layout.start_offset()..].as_ptr();

    // Output is [M, dim] (half the size of unfused gate_up)
    let mut out_buf = BRGEMM_OUT_BUF.with_borrow_mut(|buf| {
        let mut taken = std::mem::take(buf);
        let needed = m * dim;
        if taken.capacity() < needed {
            taken.reserve(needed - taken.len());
        }
        unsafe { taken.set_len(needed) };
        taken
    });

    let block_n = 32_usize;
    let n_blocks = (dim + block_n - 1) / block_n; // blocks in gate half only

    let pool = crate::ops::cpu::gemm_pool::gemm_pool();

    if m > BLOCK_M_2D {
        // 2D M×N tiling
        let m_blocks = (m + BLOCK_M_2D - 1) / BLOCK_M_2D;
        let total_blocks = m_blocks * n_blocks;
        let num_threads = pool.num_threads().min(total_blocks).max(1);

        let ctx = BrgemmFusedSilu2dCtx {
            input_ptr: in_ptr as usize,
            brg_ptr: brg.ptr as usize,
            output_ptr: out_buf.as_mut_ptr() as usize,
            k,
            m_total: m,
            dim,
            block_m: BLOCK_M_2D,
            block_n,
            m_blocks,
            n_blocks,
        };

        unsafe {
            pool.dispatch(
                brgemm_fused_silu_work_2d,
                &ctx as *const BrgemmFusedSilu2dCtx as *const u8,
                num_threads,
            );
        }
    } else {
        // 1D N-only tiling
        let num_threads = pool.num_threads().min(n_blocks).max(1);

        let ctx = BrgemmFusedSiluCtx {
            input_ptr: in_ptr as usize,
            brg_ptr: brg.ptr as usize,
            output_ptr: out_buf.as_mut_ptr() as usize,
            m: m as i64,
            dim,
            block_n,
            n_blocks,
        };

        unsafe {
            pool.dispatch(
                brgemm_fused_silu_work,
                &ctx as *const BrgemmFusedSiluCtx as *const u8,
                num_threads,
            );
        }
    }

    drop(in_storage);
    let bf16_vec: Vec<half::bf16> =
        unsafe { std::mem::transmute::<Vec<u16>, Vec<half::bf16>>(out_buf) };
    Tensor::from_vec(bf16_vec, &[m, dim], &candle_core::Device::Cpu)
}

/// Raw brgemm GEMM: input/output as raw u16 slices, no Tensor wrapping.
///
/// output = input[M, K] × packed_weight^T → output[M, N] (BF16).
/// Caller pre-allocates output buffer. Uses GemmPool for dispatch.
///
/// # Safety
/// - `input` must be `[M * K]` contiguous BF16 (u16) elements.
/// - `output` must be `[M * N]` pre-allocated u16 elements.
/// - `brg` must be a valid packed weight with matching K, N.
pub unsafe fn brgemm_gemm_raw(
    input: *const u16,
    brg: &BrgemmPackedWeight,
    output: *mut u16,
    m: usize,
    n: usize,
) {
    unsafe {
        let block_n = 32_usize;
        let n_blocks = (n + block_n - 1) / block_n;

        let pool = crate::ops::cpu::gemm_pool::gemm_pool();

        if m > BLOCK_M_2D {
            let k = brg.k;
            let m_blocks = (m + BLOCK_M_2D - 1) / BLOCK_M_2D;
            let total_blocks = m_blocks * n_blocks;
            let num_threads = pool.num_threads().min(total_blocks).max(1);

            let ctx = BrgemmDispatch2dCtx {
                input_ptr: input as usize,
                brg_ptr: brg.ptr as usize,
                output_ptr: output as usize,
                k,
                m_total: m,
                n_total: n,
                block_m: BLOCK_M_2D,
                block_n,
                m_blocks,
                n_blocks,
            };

            pool.dispatch(
                brgemm_pool_work_2d,
                &ctx as *const BrgemmDispatch2dCtx as *const u8,
                num_threads,
            );
        } else {
            let num_threads = pool.num_threads().min(n_blocks).max(1);

            let ctx = BrgemmDispatchCtx {
                input_ptr: input as usize,
                brg_ptr: brg.ptr as usize,
                output_ptr: output as usize,
                m: m as i64,
                n_total: n,
                block_n,
                n_blocks,
            };

            pool.dispatch(
                brgemm_pool_work,
                &ctx as *const BrgemmDispatchCtx as *const u8,
                num_threads,
            );
        }
    }
}

/// Raw fused gate_up GEMM + SiLU×Mul: output[M, dim] = SiLU(gate) * up.
///
/// # Safety
/// Same as `brgemm_gemm_raw`, plus `brg.n` must equal `2 * dim`.
pub unsafe fn brgemm_fused_silu_mul_raw(
    input: *const u16,
    brg: &BrgemmPackedWeight,
    output: *mut u16,
    m: usize,
    dim: usize,
) {
    unsafe {
        let block_n = 32_usize;
        let n_blocks = (dim + block_n - 1) / block_n;

        let pool = crate::ops::cpu::gemm_pool::gemm_pool();

        if m > BLOCK_M_2D {
            let k = brg.k;
            let m_blocks = (m + BLOCK_M_2D - 1) / BLOCK_M_2D;
            let total_blocks = m_blocks * n_blocks;
            let num_threads = pool.num_threads().min(total_blocks).max(1);

            let ctx = BrgemmFusedSilu2dCtx {
                input_ptr: input as usize,
                brg_ptr: brg.ptr as usize,
                output_ptr: output as usize,
                k,
                m_total: m,
                dim,
                block_m: BLOCK_M_2D,
                block_n,
                m_blocks,
                n_blocks,
            };

            pool.dispatch(
                brgemm_fused_silu_work_2d,
                &ctx as *const BrgemmFusedSilu2dCtx as *const u8,
                num_threads,
            );
        } else {
            let num_threads = pool.num_threads().min(n_blocks).max(1);

            let ctx = BrgemmFusedSiluCtx {
                input_ptr: input as usize,
                brg_ptr: brg.ptr as usize,
                output_ptr: output as usize,
                m: m as i64,
                dim,
                block_n,
                n_blocks,
            };

            pool.dispatch(
                brgemm_fused_silu_work,
                &ctx as *const BrgemmFusedSiluCtx as *const u8,
                num_threads,
            );
        }
    }
}

/// Run the custom small-M BF16 GEMM using raw weight tensor.
/// Input: [M, K], Weight: [N, K] (row-major), Output: [M, N]
fn cpu_gemm_forward(input: &Tensor, weight: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    let input = input.contiguous()?;
    let weight = weight.contiguous()?;

    let (in_storage, in_layout) = input.storage_and_layout();
    let in_offset = in_layout.start_offset();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let w_offset = w_layout.start_offset();

    let in_data = match &*in_storage {
        candle_core::Storage::Cpu(s) => s.as_slice::<half::bf16>()?,
        _ => candle_core::bail!("cpu_gemm_forward: CPU tensor required"),
    };
    let w_data = match &*w_storage {
        candle_core::Storage::Cpu(s) => s.as_slice::<half::bf16>()?,
        _ => candle_core::bail!("cpu_gemm_forward: CPU tensor required"),
    };

    // Safety: bf16 is repr(transparent) over u16
    let in_slice = unsafe {
        std::slice::from_raw_parts(in_data[in_offset..].as_ptr() as *const u16, m * k)
    };
    let w_slice = unsafe {
        std::slice::from_raw_parts(w_data[w_offset..].as_ptr() as *const u16, n * k)
    };

    let mut out_buf: Vec<u16> = Vec::with_capacity(m * n);
    unsafe { out_buf.set_len(m * n) };

    crate::ops::cpu::gemm::bf16_gemm_small_m(&mut out_buf, in_slice, w_slice, m, k, n);

    drop(in_storage);
    drop(w_storage);

    let bf16_vec: Vec<half::bf16> =
        unsafe { std::mem::transmute::<Vec<u16>, Vec<half::bf16>>(out_buf) };
    Tensor::from_vec(bf16_vec, &[m, n], &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_thread_grid() {
        // M=512, N=4864, BLOCK=32: m_blocks=16, n_blocks=152, nth=48
        let (nth_m, nth_n) = compute_thread_grid(16, 152, 48);
        assert_eq!(nth_m * nth_n, 48);
        assert!(nth_m >= 2, "should split M dimension for 16 m_blocks");

        // M=128, N=4864: m_blocks=4, n_blocks=152, nth=48
        let (nth_m, nth_n) = compute_thread_grid(4, 152, 48);
        assert_eq!(nth_m * nth_n, 48);
        assert!(nth_m >= 1);

        // m_blocks=1: should degenerate to 1D
        let (nth_m, nth_n) = compute_thread_grid(1, 152, 48);
        assert_eq!(nth_m, 1);
        assert_eq!(nth_n, 48);

        // n_threads=1: trivial
        let (nth_m, nth_n) = compute_thread_grid(16, 152, 1);
        assert_eq!(nth_m, 1);
        assert_eq!(nth_n, 1);

        // Square-ish: m_blocks=n_blocks
        let (nth_m, nth_n) = compute_thread_grid(16, 16, 16);
        assert_eq!(nth_m * nth_n, 16);
        assert_eq!(nth_m, 4); // sqrt(16) = 4
    }
}

//! GPU GEMM — DeepGEMM → CUTLASS → cuBLAS 3-tier dispatch for Linear layers.
//!
//! `GpuLinear` calls CUTLASS/DeepGEMM FFI directly, without going through
//! `Tensor::matmul()` dispatch.
//!
//! `register_gpu_gemm()` hooks `gemm_dispatch_impl` into candle-core's GEMM
//! dispatch so that every `Tensor::matmul()` on CUDA routes through DeepGEMM
//! (fast path for SM90+ BF16) with CUTLASS + cuBLAS fallbacks.
//!
//! Dispatch order:
//!   1. DeepGEMM — non-batched BF16, SM90+ (fastest, shape-constrained).
//!   2. CUTLASS — SM80+, handles BF16/F16/F32, batched, all transposes.
//!   3. cuBLAS — universal fallback. Needed on Blackwell (SM103) for the
//!      small-M shapes DeepGEMM has no kernel variant for, and for any
//!      arch where the CUTLASS SM80/SM90 cubins don't run.

use prelude_core::tensor::{bail, DType, Module, Result, Tensor};
use crate::device::{self as cb, DeviceRepr, DevicePtr};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::ffi::c_void;
use std::os::raw::c_int;

// Per-process cache of shapes that DeepGEMM / CUTLASS already failed on.
// Key: (m, n, k, batch, transa, transb, dtype). We skip the failing backend
// on subsequent calls with the same shape, which matters on Blackwell where
// SM80/SM90 CUTLASS cubins never run and DeepGEMM SM100 kernel tables have
// gaps — each failed attempt is a 10-50µs FFI round-trip × hundreds of
// layers × hundreds of decode steps = noticeable wall-clock overhead.
type ShapeKey = (i32, i32, i32, i32, bool, bool, u32);

thread_local! {
    static DEEPGEMM_FAILED: RefCell<HashSet<ShapeKey>> =
        RefCell::new(HashSet::with_capacity(128));
    static CUTLASS_FAILED: RefCell<HashSet<ShapeKey>> =
        RefCell::new(HashSet::with_capacity(128));
}

/// Register our GEMM dispatch. Must be called before any GPU matmul.
pub fn register_gpu_gemm() {
    candle_core::cuda_backend::gemm_dispatch::register_gemm_dispatch(gemm_dispatch_impl);
    tracing::info!(
        "GPU GEMM backend registered (CUTLASS + cuBLAS fallback{})",
        if cfg!(feature = "deepgemm") { " + DeepGEMM" } else { "" }
    );
}

/// GEMM dispatch implementation.
pub(crate) unsafe fn gemm_dispatch_impl(
    a: *const c_void,
    b: *const c_void,
    d: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    stride_a: i64,
    stride_b: i64,
    stride_d: i64,
    transa: bool,
    transb: bool,
    dtype: u32,
    stream: *const c_void,
) -> i32 {
    // Try DeepGEMM first for non-batched BF16 on SM90+ (fastest path).
    //
    // cuBLAS convention: m=N_features, n=M_tokens, a=weight, b=input.
    // DeepGEMM expects row-major: D[M,N] = A[M,K] @ B[K,N](col-major),
    // where M=tokens, N=features — the natural layout for LLM inference.
    // So we swap (a↔b, m↔n) to give DeepGEMM the correct orientation.
    //
    // After swap: DeepGEMM M=tokens(n), N=features(m), K=K.
    // Features (m) and K are always model-dimension-aligned.
    // Tokens (n) can be any value — DeepGEMM handles partial tiles for M.
    //
    let key: ShapeKey = (m, n, k, batch, transa, transb, dtype);

    if dtype == 0 && batch == 1 && transa && !transb {
        let known_fail = DEEPGEMM_FAILED.with(|s| s.borrow().contains(&key));
        if !known_fail {
            // Swap: DeepGEMM A=input(b), B=weight(a), M=tokens(n), N=features(m)
            let ret = unsafe { prelude_deepgemm::bf16_gemm(
                b as *mut c_void, a as *mut c_void, d,
                n, m, k, stream as *mut c_void,
            ) };
            match &ret {
                Ok(()) => return 0,
                Err(e) => {
                    tracing::warn!("DeepGEMM → CUTLASS fallback: {e}");
                    DEEPGEMM_FAILED.with(|s| { s.borrow_mut().insert(key); });
                }
            }
        }
    }

    // CUTLASS fallback (SM80+, handles BF16/FP16/F32).
    {
        let known_fail = CUTLASS_FAILED.with(|s| s.borrow().contains(&key));
        if !known_fail {
            let ret = prelude_cutlass_gemm::gemm_dispatch(
                a, b, d, m, n, k, batch, lda, ldb, ldd,
                stride_a, stride_b, stride_d,
                transa, transb, dtype, stream,
            );
            match ret {
                Ok(()) => return 0,
                Err(e) => {
                    tracing::warn!("CUTLASS → cuBLAS fallback: {e}");
                    CUTLASS_FAILED.with(|s| { s.borrow_mut().insert(key); });
                }
            }
        }
    }

    // Final fallback: cuBLAS. Universal shape/arch support. Needed on
    // Blackwell (SM103) where compiled CUTLASS SM80/SM90 cubins don't run
    // and DeepGEMM's SM100 kernel table has gaps (e.g. small-M decodes).
    match unsafe {
        cublas_gemm_ex(
            a, b, d, m, n, k, batch, lda, ldb, ldd,
            stride_a, stride_b, stride_d,
            transa, transb, dtype, stream,
        )
    } {
        Ok(()) => 0,
        Err(e) => {
            eprintln!(
                "cuBLAS GEMM error: {e} (m={m} n={n} k={k} batch={batch} dtype={dtype})"
            );
            -1
        }
    }
}

// ── cuBLAS FFI (minimal, enough for GemmEx + GemmStridedBatchedEx) ─────

// cublasOperation_t
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;

// cudaDataType_t
const CUDA_R_16F: c_int = 2;
const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;

// cublasComputeType_t
const CUBLAS_COMPUTE_32F: c_int = 68;

// cublasGemmAlgo_t
const CUBLAS_GEMM_DEFAULT: c_int = -1;

unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut *mut c_void) -> c_int;
    fn cublasSetStream_v2(handle: *mut c_void, stream: *mut c_void) -> c_int;
    fn cublasSetWorkspace_v2(
        handle: *mut c_void, workspace: *mut c_void, workspace_size: usize,
    ) -> c_int;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cublasGemmEx(
        handle: *mut c_void,
        transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int,
        alpha: *const c_void,
        a: *const c_void, a_type: c_int, lda: c_int,
        b: *const c_void, b_type: c_int, ldb: c_int,
        beta: *const c_void,
        c: *mut c_void, c_type: c_int, ldc: c_int,
        compute_type: c_int, algo: c_int,
    ) -> c_int;
    fn cublasGemmStridedBatchedEx(
        handle: *mut c_void,
        transa: c_int, transb: c_int,
        m: c_int, n: c_int, k: c_int,
        alpha: *const c_void,
        a: *const c_void, a_type: c_int, lda: c_int, stride_a: i64,
        b: *const c_void, b_type: c_int, ldb: c_int, stride_b: i64,
        beta: *const c_void,
        c: *mut c_void, c_type: c_int, ldc: c_int, stride_c: i64,
        batch_count: c_int,
        compute_type: c_int, algo: c_int,
    ) -> c_int;
}

// cuBLAS handles are not thread-safe to share, but are cheap to create.
// Keep one per worker thread — the GPU queue uses a single dedicated OS
// thread anyway, so in practice this is a single handle.
thread_local! {
    static CUBLAS_HANDLE: Cell<*mut c_void> = const { Cell::new(std::ptr::null_mut()) };
}

/// Pre-allocated cuBLAS workspace size. 64 MB is the NVIDIA-recommended value
/// for Hopper/Blackwell. Pre-allocating eliminates `cudaErrorStreamCaptureUnsupported`
/// errors that cuBLAS would otherwise throw (via bad_alloc) when its internal
/// workspace allocator tries `cudaMalloc` during an active CUDA graph capture.
const CUBLAS_WORKSPACE_BYTES: usize = 64 * 1024 * 1024;

unsafe fn get_or_init_handle() -> std::result::Result<*mut c_void, String> {
    CUBLAS_HANDLE.with(|slot| {
        let existing = slot.get();
        if !existing.is_null() {
            return Ok(existing);
        }
        let mut handle: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut handle) };
        if status != 0 || handle.is_null() {
            return Err(format!("cublasCreate_v2 failed (status {status})"));
        }

        // Pre-allocate workspace so subsequent cublasGemmEx calls never
        // request device memory during a CUDA graph capture.
        let mut workspace: *mut c_void = std::ptr::null_mut();
        let cuda_status = unsafe { cudaMalloc(&mut workspace, CUBLAS_WORKSPACE_BYTES) };
        if cuda_status != 0 || workspace.is_null() {
            tracing::warn!(
                "cuBLAS workspace preallocation failed (cudaMalloc status {cuda_status}); \
                 GPU matmul inside a captured CUDA graph may fail"
            );
        } else {
            let ws_status = unsafe {
                cublasSetWorkspace_v2(handle, workspace, CUBLAS_WORKSPACE_BYTES)
            };
            if ws_status != 0 {
                tracing::warn!(
                    "cublasSetWorkspace_v2 failed (status {ws_status}); \
                     CUDA graph capture may bad_alloc"
                );
            }
        }

        slot.set(handle);
        Ok(handle)
    })
}

unsafe fn cublas_gemm_ex(
    a: *const c_void,
    b: *const c_void,
    d: *mut c_void,
    m: i32, n: i32, k: i32,
    batch: i32,
    lda: i32, ldb: i32, ldd: i32,
    stride_a: i64, stride_b: i64, stride_d: i64,
    transa: bool, transb: bool,
    dtype: u32,
    stream: *const c_void,
) -> std::result::Result<(), String> {
    let data_ty = match dtype {
        0 => CUDA_R_16BF,
        1 => CUDA_R_16F,
        2 => CUDA_R_32F,
        other => return Err(format!("cuBLAS: unsupported dtype code {other}")),
    };
    let compute_ty = CUBLAS_COMPUTE_32F;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let op_a = if transa { CUBLAS_OP_T } else { CUBLAS_OP_N };
    let op_b = if transb { CUBLAS_OP_T } else { CUBLAS_OP_N };

    let handle = unsafe { get_or_init_handle() }?;
    let status = unsafe { cublasSetStream_v2(handle, stream as *mut c_void) };
    if status != 0 {
        return Err(format!("cublasSetStream_v2 failed (status {status})"));
    }

    let status = unsafe {
        if batch == 1 {
            cublasGemmEx(
                handle, op_a, op_b, m, n, k,
                &alpha as *const f32 as *const c_void,
                a, data_ty, lda,
                b, data_ty, ldb,
                &beta as *const f32 as *const c_void,
                d, data_ty, ldd,
                compute_ty, CUBLAS_GEMM_DEFAULT,
            )
        } else {
            cublasGemmStridedBatchedEx(
                handle, op_a, op_b, m, n, k,
                &alpha as *const f32 as *const c_void,
                a, data_ty, lda, stride_a,
                b, data_ty, ldb, stride_b,
                &beta as *const f32 as *const c_void,
                d, data_ty, ldd, stride_d,
                batch,
                compute_ty, CUBLAS_GEMM_DEFAULT,
            )
        }
    };
    if status != 0 {
        return Err(format!("cublasGemmEx failed (status {status})"));
    }
    Ok(())
}

// ── GpuLinear ─────────────────────────────────────────────────────────────

/// GPU Linear layer — calls CUTLASS/DeepGEMM directly.
///
/// Unlike `NaiveLinear` which goes through `Tensor::matmul()` → registered dispatch,
/// this extracts CUDA pointers and calls the GEMM kernels via FFI. No global
/// registration needed for this path.
#[derive(Debug, Clone)]
pub struct GpuLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl GpuLinear {
    /// Create from weight `[N, K]` and optional bias `[N]` on CUDA.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        if !weight.device().is_cuda() {
            bail!("GpuLinear: weight must be on CUDA device");
        }
        Ok(Self { weight, bias })
    }
}

// GpuLinear is a standalone CUDA linear layer used by CudaOps and benchmarks.
// It does NOT implement LinearBackend (which lives in prelude-core).
// prelude-core's Linear::from_linear() uses NaiveLinear for CUDA,
// relying on register_gpu_gemm() to route matmul through CUTLASS.

impl Module for GpuLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims().to_vec();
        let k = *x_dims.last().unwrap();
        let (n, wk) = self.weight.dims2()?;
        if k != wk {
            bail!("GpuLinear: input dim {k} != weight dim {wk}");
        }
        let m: usize = x_dims[..x_dims.len() - 1].iter().product();

        // Flatten batch dims and ensure contiguous
        let x_flat = if x_dims.len() == 2 && x.is_contiguous() {
            x.clone()
        } else {
            x.reshape((m, k))?.contiguous()?
        };

        // y[M,N] = x[M,K] @ W[N,K]^T
        let y_flat = gpu_matmul_nt(&x_flat, &self.weight, m, n, k)?;

        // Reshape back to original batch dims
        let y = if x_dims.len() > 2 {
            let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
            out_dims.push(n);
            y_flat.reshape(out_dims.as_slice())?
        } else {
            y_flat
        };

        match &self.bias {
            None => Ok(y),
            Some(bias) => y.broadcast_add(bias),
        }
    }
}

/// Compute y[M,N] = x[M,K] @ W[N,K]^T via CUTLASS/DeepGEMM (non-batched).
///
/// Both tensors must be contiguous and on the same CUDA device.
pub(crate) fn gpu_matmul_nt(x: &Tensor, w: &Tensor, m: usize, n: usize, k: usize) -> Result<Tensor> {
    gpu_matmul_nt_batched(x, w, m, n, k, 1)
}

/// Compute y[batch,M,N] = x[batch,M,K] @ W[batch,N,K]^T via CUTLASS/DeepGEMM.
///
/// Both tensors must be contiguous and on the same CUDA device.
pub(crate) fn gpu_matmul_nt_batched(x: &Tensor, w: &Tensor, m: usize, n: usize, k: usize, batch: usize) -> Result<Tensor> {
    let dtype_code = match x.dtype() {
        DType::BF16 => 0u32,
        DType::F16 => 1u32,
        DType::F32 => 2u32,
        dt => bail!("gpu_matmul_nt: unsupported dtype {dt:?}"),
    };
    match x.dtype() {
        DType::BF16 => gpu_matmul_nt_typed::<half::bf16>(x, w, m, n, k, batch, dtype_code),
        DType::F16 => gpu_matmul_nt_typed::<half::f16>(x, w, m, n, k, batch, dtype_code),
        DType::F32 => gpu_matmul_nt_typed::<f32>(x, w, m, n, k, batch, dtype_code),
        dt => bail!("gpu_matmul_nt: unsupported dtype {dt:?}"),
    }
}

/// Typed inner function — extracts CUDA pointers and calls GEMM dispatch.
fn gpu_matmul_nt_typed<T>(
    x: &Tensor,
    w: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    dtype_code: u32,
) -> Result<Tensor>
where
    T: cb::GpuDType + DeviceRepr + candle_core::cuda_backend::CudaDType,
{
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::WrapErr;

    let (x_storage, x_layout) = x.storage_and_layout();
    let x_cuda = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("gpu_matmul_nt: x requires CUDA"),
    };
    let (w_storage, w_layout) = w.storage_and_layout();
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("gpu_matmul_nt: w requires CUDA"),
    };

    let dev = x_cuda.device().clone();
    let stream = dev.cuda_stream();

    let x_slice = x_cuda.as_cuda_slice::<T>()?.slice(x_layout.start_offset()..);
    let w_slice = w_cuda.as_cuda_slice::<T>()?.slice(w_layout.start_offset()..);

    let total = batch * m * n;
    let out = unsafe { dev.alloc::<T>(total) }?;

    let x_ptr = x_slice.device_ptr(&stream).0 as *const c_void;
    let w_ptr = w_slice.device_ptr(&stream).0 as *const c_void;
    let out_ptr = out.device_ptr(&stream).0 as *mut c_void;
    let raw_stream = unsafe { stream.cu_stream() } as *const c_void;

    // TN convention: D[m_c,n_c] = A[N,K]^T @ B[M,K]
    let stride_a = (n * k) as i64;
    let stride_b = (m * k) as i64;
    let stride_d = (m * n) as i64;
    let ret = unsafe {
        gemm_dispatch_impl(
            w_ptr, x_ptr, out_ptr,
            n as i32, m as i32, k as i32,
            batch as i32,
            k as i32, k as i32, n as i32,
            stride_a, stride_b, stride_d,
            true, false,
            dtype_code,
            raw_stream,
        )
    };
    if ret != 0 {
        bail!("GPU GEMM failed (code {ret}) M={m} N={n} K={k} batch={batch}");
    }

    drop(x_storage);
    drop(w_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        (total,),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}


//! GPU GEMM — direct CUTLASS/DeepGEMM dispatch for Linear layers.
//!
//! `GpuLinear` calls CUTLASS/DeepGEMM FFI directly, without going through
//! `Tensor::matmul()` dispatch.
//!
//! `register_gpu_gemm()` is still provided for other `Tensor::matmul()` callers
//! (attention, RoPE, etc.) that haven't been converted yet.

use prelude_core::tensor::{bail, DType, Module, Result, Tensor};
use crate::device::{self as cb, DeviceRepr, DevicePtr};
use std::ffi::c_void;

/// Register our GEMM dispatch. Must be called before any GPU matmul.
///
/// This replaces cuBLAS: all `Tensor::matmul()` on CUDA will route through
/// CUTLASS (SM80+) with optional DeepGEMM fast path (SM90+ BF16).
pub fn register_gpu_gemm() {
    candle_core::cuda_backend::gemm_dispatch::register_gemm_dispatch(gemm_dispatch_impl);
    tracing::info!("GPU GEMM backend registered (CUTLASS{})",
        if cfg!(feature = "deepgemm") { " + DeepGEMM" } else { "" });
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
    if dtype == 0 && batch == 1 && transa && !transb {
        // Swap: DeepGEMM A=input(b), B=weight(a), M=tokens(n), N=features(m)
        let ret = prelude_deepgemm::bf16_gemm(
            b as *mut c_void, a as *mut c_void, d,
            n, m, k, stream as *mut c_void,
        );
        match &ret {
            Ok(()) => return 0,
            Err(e) => {
                tracing::debug!("DeepGEMM → CUTLASS fallback: {e}");
            }
        }
    }

    // CUTLASS fallback (SM80+, handles BF16/FP16/F32)
    {
        let ret = prelude_cutlass_gemm::gemm_dispatch(
            a, b, d, m, n, k, batch, lda, ldb, ldd,
            stride_a, stride_b, stride_d,
            transa, transb, dtype, stream,
        );
        return match ret {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("CUTLASS GEMM error: {e}");
                -1
            }
        };
    }

    #[allow(unreachable_code)]
    -99 // No GEMM backend available
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


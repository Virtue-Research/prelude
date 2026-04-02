//! GPU GEMM — direct CUTLASS/DeepGEMM dispatch for Linear layers.
//!
//! `GpuLinear` calls CUTLASS/DeepGEMM FFI directly, without going through
//! candle's `Tensor::matmul()` monkey-patch.
//!
//! `register_gpu_gemm()` is still provided for other `Tensor::matmul()` callers
//! (attention, RoPE, etc.) that haven't been converted yet.

use candle_core::{DType, Module, Result, Tensor};
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use std::ffi::c_void;

/// Register our GEMM dispatch with candle-core. Must be called before any GPU matmul.
///
/// This replaces cuBLAS: all `Tensor::matmul()` on CUDA will route through
/// CUTLASS (SM80+) with optional DeepGEMM fast path (SM90+ BF16).
pub fn register_gpu_gemm() {
    candle_core::cuda_backend::gemm_dispatch::register_gemm_dispatch(gemm_dispatch_impl);
    tracing::info!("GPU GEMM backend registered (CUTLASS{})",
        if cfg!(feature = "deepgemm") { " + DeepGEMM" } else { "" });
}

/// The actual dispatch implementation. Matches candle-core's GemmDispatchFn signature.
unsafe fn gemm_dispatch_impl(
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
    // candle passes cuBLAS convention: m=N_features, n=M_tokens, a=weight, b=input.
    // DeepGEMM expects row-major: D[M,N] = A[M,K] @ B[K,N](col-major),
    // where M=tokens, N=features — the natural layout for LLM inference.
    // So we swap (a↔b, m↔n) to give DeepGEMM the correct orientation.
    //
    // After swap: DeepGEMM M=tokens(n), N=features(m), K=K.
    // Features (m) and K are always model-dimension-aligned.
    // Tokens (n) can be any value — DeepGEMM handles partial tiles for M.
    #[cfg(feature = "deepgemm")]
    if dtype == 0 && batch == 1 && transa && !transb
    {
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
    #[cfg(feature = "cutlass-gemm")]
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
/// Unlike `CandleLinear` which goes through `Tensor::matmul()` → registered dispatch,
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
            candle_core::bail!("GpuLinear: weight must be on CUDA device");
        }
        Ok(Self { weight, bias })
    }
}

// GpuLinear is a standalone CUDA linear layer used by CudaOps and benchmarks.
// It does NOT implement LinearBackend (which lives in prelude-core).
// prelude-core's Linear::from_candle() uses CandleLinear for CUDA,
// relying on register_gpu_gemm() to route matmul through CUTLASS.

impl Module for GpuLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims().to_vec();
        let k = *x_dims.last().unwrap();
        let (n, wk) = self.weight.dims2()?;
        if k != wk {
            candle_core::bail!("GpuLinear: input dim {k} != weight dim {wk}");
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

/// Compute y[M,N] = x[M,K] @ W[N,K]^T via CUTLASS/DeepGEMM.
///
/// Both tensors must be contiguous and on the same CUDA device.
fn gpu_matmul_nt(x: &Tensor, w: &Tensor, m: usize, n: usize, k: usize) -> Result<Tensor> {
    let dtype_code = match x.dtype() {
        DType::BF16 => 0u32,
        DType::F16 => 1u32,
        DType::F32 => 2u32,
        dt => candle_core::bail!("gpu_matmul_nt: unsupported dtype {dt:?}"),
    };
    match x.dtype() {
        DType::BF16 => gpu_matmul_nt_typed::<half::bf16>(x, w, m, n, k, dtype_code),
        DType::F16 => gpu_matmul_nt_typed::<half::f16>(x, w, m, n, k, dtype_code),
        DType::F32 => gpu_matmul_nt_typed::<f32>(x, w, m, n, k, dtype_code),
        dt => candle_core::bail!("gpu_matmul_nt: unsupported dtype {dt:?}"),
    }
}

/// Typed inner function — extracts CUDA pointers and calls GEMM dispatch.
fn gpu_matmul_nt_typed<T>(
    x: &Tensor,
    w: &Tensor,
    m: usize,
    n: usize,
    k: usize,
    dtype_code: u32,
) -> Result<Tensor>
where
    T: candle_core::cuda_backend::CudaDType
        + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
{
    use candle_core::op::BackpropOp;

    let dev = x.device().as_cuda_device()?;

    // Extract CUDA slices
    let (x_storage, _x_layout) = x.storage_and_layout();
    let x_slice = match &*x_storage {
        candle_core::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
        _ => candle_core::bail!("gpu_matmul_nt: expected CUDA tensor for x"),
    };

    let (w_storage, _w_layout) = w.storage_and_layout();
    let w_slice = match &*w_storage {
        candle_core::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
        _ => candle_core::bail!("gpu_matmul_nt: expected CUDA tensor for w"),
    };

    // Allocate output
    let out = unsafe { dev.alloc::<T>(m * n)? };

    // Get raw device pointers
    let x_ptr = x_slice.device_ptr(x_slice.stream()).0 as *const c_void;
    let w_ptr = w_slice.device_ptr(w_slice.stream()).0 as *const c_void;
    let out_ptr = out.device_ptr(out.stream()).0 as *mut c_void;
    let stream = dev.cuda_stream();
    let raw_stream = unsafe { stream.cu_stream() } as *const c_void;

    // cuBLAS convention: A=weight, B=input, m=N_features, n=M_tokens
    // transa=true because weight [N,K] row-major needs transpose in cuBLAS col-major
    let ret = unsafe {
        gemm_dispatch_impl(
            w_ptr,       // A = weight
            x_ptr,       // B = input
            out_ptr,     // D = output
            n as i32,    // m_cublas = N (features)
            m as i32,    // n_cublas = M (tokens)
            k as i32,
            1,           // batch
            k as i32,    // lda (weight [N,K] row-major)
            k as i32,    // ldb (input [M,K] row-major)
            n as i32,    // ldd
            0, 0, 0,     // batch strides (non-batched)
            true, false, // transa, transb
            dtype_code,
            raw_stream,
        )
    };
    if ret != 0 {
        candle_core::bail!("GPU GEMM failed (code {ret}) M={m} N={n} K={k}");
    }

    // Release storage guards before wrapping output
    drop(x_storage);
    drop(w_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        (m, n),
        BackpropOp::none(),
        false,
    ))
}

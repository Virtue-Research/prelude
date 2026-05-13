use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use cudarc::driver::DevicePtr;
use flashinfer::types::*;
use flashinfer::{KernelRegistry, TVMSafeCallFn};
use prelude_core::tensor::{DType, DeviceExt, Result, Tensor, bail};

const GEMM_WS_BYTES: usize = 32 * 1024 * 1024;
const SCALE_MAJOR_K: &[u8] = b"K\0";

static REGISTRY: OnceLock<KernelRegistry> = OnceLock::new();
static WORKSPACE: Mutex<Option<GemmWorkspace>> = Mutex::new(None);

unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
}

struct GemmWorkspace {
    int_ws: *mut c_void,
    float_ws: *mut c_void,
    device_id: i32,
}

unsafe impl Send for GemmWorkspace {}

impl GemmWorkspace {
    fn new(device_id: i32) -> Result<Self> {
        unsafe {
            cudaSetDevice(device_id);
            let mut int_ws = std::ptr::null_mut();
            let mut float_ws = std::ptr::null_mut();
            if cudaMalloc(&mut int_ws, GEMM_WS_BYTES) != 0 || int_ws.is_null() {
                bail!("FlashInfer FP8 GEMM: cudaMalloc int workspace failed");
            }
            if cudaMalloc(&mut float_ws, GEMM_WS_BYTES) != 0 || float_ws.is_null() {
                cudaFree(int_ws);
                bail!("FlashInfer FP8 GEMM: cudaMalloc float workspace failed");
            }
            Ok(Self {
                int_ws,
                float_ws,
                device_id,
            })
        }
    }
}

impl Drop for GemmWorkspace {
    fn drop(&mut self) {
        unsafe {
            cudaSetDevice(self.device_id);
            if !self.int_ws.is_null() {
                cudaFree(self.int_ws);
            }
            if !self.float_ws.is_null() {
                cudaFree(self.float_ws);
            }
        }
    }
}

fn registry() -> &'static KernelRegistry {
    REGISTRY.get_or_init(KernelRegistry::new)
}

fn with_workspace<T>(
    device_id: i32,
    f: impl FnOnce(*mut c_void, *mut c_void) -> Result<T>,
) -> Result<T> {
    let mut guard = WORKSPACE
        .lock()
        .map_err(|e| candle_core::Error::Msg(format!("FlashInfer FP8 GEMM workspace lock: {e}")))?;
    if guard.as_ref().map(|ws| ws.device_id) != Some(device_id) {
        *guard = Some(GemmWorkspace::new(device_id)?);
    }
    let ws = guard
        .as_ref()
        .expect("FlashInfer FP8 GEMM workspace initialized");
    f(ws.int_ws, ws.float_ws)
}

pub(crate) fn f8_e4m3_dt() -> DLDataType {
    DLDataType {
        code: KDLFLOAT8_E4M3FN,
        bits: 8,
        lanes: 1,
    }
}

pub(crate) fn bf16_dt() -> DLDataType {
    DLDataType {
        code: KDLBFLOAT,
        bits: 16,
        lanes: 1,
    }
}

pub(crate) fn f32_dt() -> DLDataType {
    DLDataType {
        code: KDLFLOAT,
        bits: 32,
        lanes: 1,
    }
}

pub(crate) fn i32_dt() -> DLDataType {
    DLDataType {
        code: KDLINT,
        bits: 32,
        lanes: 1,
    }
}

fn u8_dt() -> DLDataType {
    DLDataType {
        code: KDLUINT,
        bits: 8,
        lanes: 1,
    }
}

pub(crate) fn raw_dl(
    data: *mut c_void,
    device_id: i32,
    dtype: DLDataType,
    shape: &[i64],
) -> DLTensor {
    DLTensor {
        data,
        device: DLDevice {
            device_type: KDLCUDA,
            device_id,
        },
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: std::ptr::null(),
        byte_offset: 0,
    }
}

pub(crate) fn tensor_to_dl(
    tensor: &Tensor,
    dtype: DLDataType,
    device_id: i32,
    ctx: &str,
) -> Result<(DLTensor, Vec<i64>)> {
    let ptr = tensor_device_ptr(tensor, ctx)?;
    let shape: Vec<i64> = tensor.dims().iter().map(|&d| d as i64).collect();
    let dl = raw_dl(ptr, device_id, dtype, &shape);
    Ok((dl, shape))
}

pub(crate) fn tensor_device_ptr(tensor: &Tensor, ctx: &str) -> Result<*mut c_void> {
    let (storage, layout) = tensor.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("{ctx}: tensor must be on CUDA"),
    };
    let dev = cuda.device().clone();
    let stream = dev.cuda_stream();
    let offset = layout.start_offset();
    let base = match tensor.dtype() {
        DType::BF16 => cuda.as_cuda_slice::<half::bf16>()?.device_ptr(&stream).0,
        DType::F16 => cuda.as_cuda_slice::<half::f16>()?.device_ptr(&stream).0,
        DType::F32 => cuda.as_cuda_slice::<f32>()?.device_ptr(&stream).0,
        DType::I32 => cuda.as_cuda_slice::<i32>()?.device_ptr(&stream).0,
        DType::U32 => cuda.as_cuda_slice::<u32>()?.device_ptr(&stream).0,
        DType::U8 => cuda.as_cuda_slice::<u8>()?.device_ptr(&stream).0,
        DType::F8E4M3 => {
            cuda.as_cuda_slice::<float8::F8E4M3>()?
                .device_ptr(&stream)
                .0
        }
        other => bail!("{ctx}: unsupported dtype {other:?}"),
    };
    let ptr = base + (offset * tensor.dtype().size_in_bytes()) as u64;
    Ok(ptr as *mut c_void)
}

fn utility(name: &str) -> Result<TVMSafeCallFn> {
    registry()
        .get_utility(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("FlashInfer utility '{name}' not compiled")))
}

fn mma_sm(m: usize) -> i64 {
    if m >= 256 { 2 } else { 1 }
}

pub(crate) fn fp8_gemm_nt_scalar(
    qinput: &Tensor,
    weight: &Tensor,
    scale_a: &Tensor,
    scale_b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor> {
    if qinput.dtype() != DType::F8E4M3 || weight.dtype() != DType::F8E4M3 {
        bail!(
            "FlashInfer scalar FP8 GEMM: expected F8E4M3 inputs, got {:?} and {:?}",
            qinput.dtype(),
            weight.dtype()
        );
    }
    if scale_a.dtype() != DType::F32 || scale_b.dtype() != DType::F32 {
        bail!(
            "FlashInfer scalar FP8 GEMM: expected F32 scales, got {:?} and {:?}",
            scale_a.dtype(),
            scale_b.dtype()
        );
    }
    if scale_a.shape().elem_count() != 1 || scale_b.shape().elem_count() != 1 {
        bail!(
            "FlashInfer scalar FP8 GEMM: scales must be scalar, got {:?} and {:?}",
            scale_a.dims(),
            scale_b.dims()
        );
    }

    let (storage, _) = qinput.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("FlashInfer scalar FP8 GEMM: input must be on CUDA"),
    };
    let dev = cuda.device().clone();
    let device_id = qinput.device().ordinal() as i32;
    let stream = dev.cuda_stream();
    let raw_stream = stream.cu_stream() as *mut c_void;
    drop(storage);

    let output = unsafe { dev.alloc::<half::bf16>(m * n) }?;
    let out_ptr = output.device_ptr(&stream).0 as *mut c_void;

    let ws_shape = [GEMM_WS_BYTES as i64];
    let a_shape = [m as i64, k as i64];
    let b_shape = [n as i64, k as i64];
    let out_shape = [m as i64, n as i64];
    let a_dl = raw_dl(
        tensor_device_ptr(qinput, "FlashInfer scalar FP8 GEMM A")?,
        device_id,
        f8_e4m3_dt(),
        &a_shape,
    );
    let b_dl = raw_dl(
        tensor_device_ptr(weight, "FlashInfer scalar FP8 GEMM B")?,
        device_id,
        f8_e4m3_dt(),
        &b_shape,
    );
    let (sfa_dl, _sfa_shape) = tensor_to_dl(
        scale_a,
        f32_dt(),
        device_id,
        "FlashInfer scalar FP8 GEMM SFA",
    )?;
    let (sfb_dl, _sfb_shape) = tensor_to_dl(
        scale_b,
        f32_dt(),
        device_id,
        "FlashInfer scalar FP8 GEMM SFB",
    )?;
    let out_dl = raw_dl(out_ptr, device_id, bf16_dt(), &out_shape);

    let func = utility("fp8_gemm")?;
    registry().set_stream(device_id, raw_stream);
    with_workspace(device_id, |_int_ws, float_ws| {
        let ws_dl = raw_dl(float_ws, device_id, u8_dt(), &ws_shape);
        let args = vec![
            TVMFFIAny::dltensor(&a_dl),
            TVMFFIAny::dltensor(&b_dl),
            TVMFFIAny::dltensor(&sfa_dl),
            TVMFFIAny::dltensor(&sfb_dl),
            TVMFFIAny::dltensor(&out_dl),
            TVMFFIAny::dltensor(&ws_dl),
            TVMFFIAny::int64(-1),
        ];
        unsafe { registry().call(func, &args) }.map_err(candle_core::Error::Msg)?;
        Ok(())
    })?;

    let storage = candle_core::CudaStorage::wrap_cuda_slice(output, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        (m, n),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn group_fp8_gemm_nt_groupwise_raw(
    device_id: i32,
    stream: *mut c_void,
    a_ptr: *mut c_void,
    b: &Tensor,
    sfa_ptr: *mut c_void,
    sfb: &Tensor,
    d_ptr: *mut c_void,
    m_indptr_ptr: *mut c_void,
    padded_m: usize,
    num_experts: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if b.dtype() != DType::F8E4M3 || sfb.dtype() != DType::F32 {
        bail!(
            "FlashInfer grouped FP8 GEMM: expected F8 weights and F32 scales, got {:?} and {:?}",
            b.dtype(),
            sfb.dtype()
        );
    }
    if k % 128 != 0 || n % 128 != 0 {
        bail!("FlashInfer grouped FP8 GEMM requires N and K multiples of 128, got N={n}, K={k}");
    }

    let func = utility("group_gemm_fp8_nt_groupwise")?;
    registry().set_stream(device_id, stream);

    let ws_shape = [GEMM_WS_BYTES as i64];
    let a_shape = [padded_m as i64, k as i64];
    let d_shape = [padded_m as i64, n as i64];
    let sfa_shape = [padded_m as i64, (k / 128) as i64];
    let indptr_shape = [(num_experts + 1) as i64];
    let a_dl = raw_dl(a_ptr, device_id, f8_e4m3_dt(), &a_shape);
    let d_dl = raw_dl(d_ptr, device_id, bf16_dt(), &d_shape);
    let sfa_dl = raw_dl(sfa_ptr, device_id, f32_dt(), &sfa_shape);
    let indptr_dl = raw_dl(m_indptr_ptr, device_id, i32_dt(), &indptr_shape);
    let (b_dl, _b_shape) = tensor_to_dl(b, f8_e4m3_dt(), device_id, "FlashInfer grouped FP8 B")?;
    let (sfb_dl, _sfb_shape) =
        tensor_to_dl(sfb, f32_dt(), device_id, "FlashInfer grouped FP8 SFB")?;

    with_workspace(device_id, |int_ws, float_ws| {
        let int_ws_dl = raw_dl(int_ws, device_id, u8_dt(), &ws_shape);
        let float_ws_dl = raw_dl(float_ws, device_id, u8_dt(), &ws_shape);
        let mut args = vec![
            TVMFFIAny::dltensor(&int_ws_dl),
            TVMFFIAny::dltensor(&float_ws_dl),
            TVMFFIAny::dltensor(&a_dl),
            TVMFFIAny::dltensor(&b_dl),
            TVMFFIAny::dltensor(&sfa_dl),
            TVMFFIAny::dltensor(&sfb_dl),
            TVMFFIAny::dltensor(&d_dl),
            TVMFFIAny::dltensor(&indptr_dl),
            TVMFFIAny::int64(n as i64),
            TVMFFIAny::int64(k as i64),
            TVMFFIAny::int64(1),
            TVMFFIAny::int64(128),
            TVMFFIAny::int64(128),
            TVMFFIAny::raw_str(SCALE_MAJOR_K.as_ptr()),
        ];
        if registry().arch() < 120 {
            args.push(TVMFFIAny::int64(mma_sm(padded_m / num_experts.max(1))));
        }
        unsafe { registry().call(func, &args) }.map_err(candle_core::Error::Msg)?;
        Ok(())
    })
}

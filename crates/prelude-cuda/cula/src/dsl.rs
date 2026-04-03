//! CuTe DSL kernel dispatch for cuLA (TVM FFI calling convention).
//!
//! Same pattern as FA4: statically linked AOT kernels called via TVM FFI.
//! build.rs generates `cula_dsl_dispatch.rs` with extern declarations + lookup.

use std::ffi::c_void;

// ── TVM FFI types (identical to FA4) ─────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *const i64,
    pub strides: *const i64,
    pub byte_offset: u64,
}

pub const KDLCUDA: i32 = 2;
pub const KDLBFLOAT: u8 = 4;
pub const KDLFLOAT: u8 = 2;
pub const KDLINT: u8 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TVMFFIAny {
    pub type_index: i32,
    pub zero_padding: u32,
    pub v_union: u64,
}

const KTVMFFI_NONE: i32 = 0;
const KTVMFFI_INT: i32 = 1;
const KTVMFFI_FLOAT: i32 = 3;
const KTVMFFI_DLTENSOR_PTR: i32 = 7;

impl TVMFFIAny {
    pub fn none() -> Self {
        Self { type_index: KTVMFFI_NONE, zero_padding: 0, v_union: 0 }
    }
    pub fn dltensor(tensor: *const DLTensor) -> Self {
        Self { type_index: KTVMFFI_DLTENSOR_PTR, zero_padding: 0, v_union: tensor as u64 }
    }
    pub fn float32(val: f32) -> Self {
        let f64_val = val as f64;
        Self { type_index: KTVMFFI_FLOAT, zero_padding: 0, v_union: f64_val.to_bits() }
    }
    pub fn int32(val: i32) -> Self {
        Self { type_index: KTVMFFI_INT, zero_padding: 0, v_union: val as i64 as u64 }
    }
}

// ── Kernel dispatch ──────────────────────────────────────────────────

pub type TVMSafeCallFn =
    unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;

// Include build.rs-generated dispatch table
include!(concat!(env!("OUT_DIR"), "/cula_dsl_dispatch.rs"));

// ── Extern: TVM FFI runtime + CUDA ──────────────────────────────────

unsafe extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const std::ffi::c_char;
    fn cudaDeviceSynchronize() -> i32;
    fn TVMFFIEnvSetStream(
        device_type: i32, device_id: i32,
        stream: *mut c_void, out_original: *mut *mut c_void,
    ) -> i32;
}

const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

fn detect_gpu_arch() -> u32 {
    unsafe {
        let mut device = 0i32;
        if cudaGetDevice(&mut device) != 0 { return 0; }
        let mut major = 0i32;
        let mut minor = 0i32;
        if cudaDeviceGetAttribute(&mut major, CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR, device) != 0 { return 0; }
        if cudaDeviceGetAttribute(&mut minor, CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR, device) != 0 { return 0; }
        (major * 10 + minor) as u32
    }
}

// ── Kernel registry ──────────────────────────────────────────────────

pub struct DslKernelRegistry {
    arch: u32,
}

impl DslKernelRegistry {
    pub fn new() -> Self {
        let arch = detect_gpu_arch();
        tracing::debug!("cuLA DslKernelRegistry: detected SM{arch}");
        Self { arch }
    }

    pub fn arch(&self) -> u32 {
        self.arch
    }

    /// Look up a DSL kernel variant by name.
    pub fn get(&self, kernel_type: &str, key: &str) -> Option<TVMSafeCallFn> {
        lookup_dsl(kernel_type, key, self.arch)
    }

    pub fn set_stream(&self, device_id: i32, stream: *mut c_void) {
        unsafe {
            TVMFFIEnvSetStream(KDLCUDA, device_id, stream, std::ptr::null_mut());
        }
    }

    /// Call a DSL kernel via TVM FFI.
    ///
    /// # Safety
    /// All TVMFFIAny args must contain valid device pointers.
    pub unsafe fn call_kernel(
        &self,
        func: TVMSafeCallFn,
        args: &mut [TVMFFIAny],
    ) -> Result<(), String> {
        let mut result = TVMFFIAny::none();
        let ret = unsafe {
            func(
                std::ptr::null_mut(),
                args.as_ptr(),
                args.len() as i32,
                &mut result,
            )
        };
        if ret != 0 {
            let detail = unsafe {
                cudaDeviceSynchronize();
                let err = cudaGetLastError();
                if err != 0 {
                    let ptr = cudaGetErrorString(err);
                    if !ptr.is_null() {
                        format!("CUDA error {err}: {}", std::ffi::CStr::from_ptr(ptr).to_string_lossy())
                    } else {
                        format!("CUDA error {err}")
                    }
                } else {
                    "TVM FFI internal failure".to_string()
                }
            };
            return Err(format!("cuLA DSL kernel call failed (code {ret}): {detail}"));
        }
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

pub fn make_dltensor(
    data: *mut c_void, device: DLDevice, dtype: DLDataType,
    shape: &[i64], strides: &[i64],
) -> DLTensor {
    DLTensor {
        data, device,
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
        byte_offset: 0,
    }
}

pub fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

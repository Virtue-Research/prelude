//! CuTe DSL kernel dispatch for cuLA (TVM FFI calling convention).
//!
//! Same pattern as FA4: statically linked AOT kernels called via TVM FFI.
//! build.rs generates `cula_dsl_dispatch.rs` with extern declarations + lookup.

use std::ffi::c_void;

// ── TVM FFI types (from shared prelude-tvm-ffi crate) ───────────────

pub use prelude_tvm_ffi::{
    DLDevice, DLDataType, DLTensor, TVMFFIAny, TVMSafeCallFn,
    KDLCUDA, KDLBFLOAT, KDLFLOAT, KDLINT,
};

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

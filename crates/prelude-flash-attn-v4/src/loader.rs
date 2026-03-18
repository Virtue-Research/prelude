//! Kernel dispatch: statically linked FA4 kernel variants.
//!
//! Kernel .o files are compiled into a static archive (libfa4_kernels.a)
//! and linked into the binary at build time. No runtime dlopen needed.
//!
//! Variant matrix: head_dim × gqa × causal × window × softcap × arch.
//! pack_gqa is auto-derived from gqa_ratio (True when gqa > 1).
//!
//! Multi-arch: kernels for multiple SM architectures (e.g. SM90 + SM120) can
//! coexist in one binary. `KernelRegistry` auto-detects the GPU arch at runtime
//! and dispatches to the correct variant.
//!
//! build.rs generates `fa4_dispatch.rs` with extern "C" declarations for each
//! variant's `__tvm_ffi_{name}` symbol and a lookup function.

use crate::types::TVMFFIAny;
use std::ffi::c_void;

/// TVM safe call function signature (TVMFFISafeCallType).
/// Each kernel variant exports `__tvm_ffi_{variant_name}` with this signature.
/// int func(void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* ret_val);
pub type TVMSafeCallFn =
    unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;

/// Key for looking up a kernel variant.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct KernelKey {
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub causal: bool,
    pub window: bool,
    /// GQA packing optimization — auto-enable when gqa_ratio > 1.
    pub pack_gqa: bool,
    /// Attention logit soft-capping, stored as `f32::to_bits()`. 0 = disabled.
    /// Used by Gemma models (30.0 for Gemma2, 50.0 for Gemma3).
    pub softcap_bits: u32,
}

impl KernelKey {
    /// Convenience: create a key with pack_gqa auto-derived and no softcap.
    pub fn new(head_dim: u32, gqa_ratio: u32, causal: bool, window: bool) -> Self {
        Self {
            head_dim,
            gqa_ratio,
            causal,
            window,
            pack_gqa: gqa_ratio > 1,
            softcap_bits: 0,
        }
    }

    /// Create a key with explicit softcap value.
    pub fn with_softcap(mut self, softcap: Option<f32>) -> Self {
        self.softcap_bits = softcap.map_or(0, |v| v.to_bits());
        self
    }

    /// Get the softcap value, if any.
    pub fn softcap_value(&self) -> Option<f32> {
        if self.softcap_bits == 0 {
            None
        } else {
            Some(f32::from_bits(self.softcap_bits))
        }
    }
}

// Include the build.rs-generated dispatch table
include!(concat!(env!("OUT_DIR"), "/fa4_dispatch.rs"));

// ── GPU arch detection via CUDA runtime API ─────────────────────────

extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

/// Detect the current CUDA device's SM architecture (e.g. 90, 120).
/// Returns 0 if detection fails (no GPU, driver not loaded, etc.).
fn detect_gpu_arch() -> u32 {
    unsafe {
        let mut device = 0i32;
        if cudaGetDevice(&mut device) != 0 {
            return 0;
        }
        let mut major = 0i32;
        let mut minor = 0i32;
        cudaDeviceGetAttribute(&mut major, CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR, device);
        cudaDeviceGetAttribute(&mut minor, CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR, device);
        (major * 10 + minor) as u32
    }
}

/// Collection of statically linked FA4 kernels.
///
/// Multi-arch: kernels for multiple SM architectures can coexist in one binary.
/// The registry auto-detects the GPU arch and dispatches to the correct variant.
pub struct KernelRegistry {
    arch: u32,
}

impl KernelRegistry {
    /// Create a new registry, auto-detecting the current GPU's SM architecture.
    pub fn new() -> Self {
        let arch = detect_gpu_arch();
        tracing::debug!("FA4 KernelRegistry: detected SM{arch}");
        Self { arch }
    }

    /// Create a registry with an explicit SM architecture.
    pub fn with_arch(arch: u32) -> Self {
        Self { arch }
    }

    /// The detected (or explicit) SM architecture.
    pub fn arch(&self) -> u32 {
        self.arch
    }

    /// Look up a kernel variant by key + detected arch.
    pub fn get(&self, key: &KernelKey) -> Option<TVMSafeCallFn> {
        lookup(key, self.arch)
    }

    /// Call a statically linked kernel via TVM FFI packed calling convention.
    ///
    /// # Safety
    /// All TVMFFIAny args must contain valid device pointers.
    pub unsafe fn call_kernel(
        &self,
        func: TVMSafeCallFn,
        args: &mut [TVMFFIAny],
    ) -> Result<(), String> {
        let mut result = TVMFFIAny::none();
        let ret = func(
            std::ptr::null_mut(),
            args.as_ptr(),
            args.len() as i32,
            &mut result,
        );
        if ret != 0 {
            return Err(format!("FA4 kernel call failed (code {ret})"));
        }
        Ok(())
    }
}

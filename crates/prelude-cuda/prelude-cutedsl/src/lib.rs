//! Runtime glue for statically-linked CuTeDSL kernels.
//!
//! Each kernel crate (`cula`, future `flashinfer` DSL phase, `fa4` once
//! it adopts the shared registry) compiles its CuTe DSL kernels ahead
//! of time into `__tvm_ffi_<name>` symbols, runs the generated
//! dispatch table into `OUT_DIR/<crate>_dsl_dispatch.rs`, and then
//! exposes a registry built around this crate's [`DslKernelRegistry`].
//!
//! There are two moving parts:
//!
//! 1. **At build time**: `prelude-kernelbuild::dispatch` emits a
//!    `lookup_dsl(kernel_type, key, arch) -> Option<TVMSafeCallFn>`
//!    function in the consumer crate's `OUT_DIR`. The crate `include!`
//!    s that file from its own module.
//!
//! 2. **At runtime**: the consumer constructs a `DslKernelRegistry`
//!    with `DslKernelRegistry::new(lookup_fn)`, passing its own
//!    generated `lookup_dsl` as the resolver. Callers then go through
//!    the registry for arch probing, stream setup, and kernel lookup.
//!
//! Before this crate existed, `cula/src/dsl.rs` owned the registry
//! type and was tightly coupled to its own generated dispatch table.
//! Lifting the type into a shared crate means future kernel crates
//! (flashinfer's gdn_decode_mtp, etc.) don't have to copy it.
//!
//! ## Crate boundary
//!
//! This crate is **runtime-only** — it's linked into the inference
//! binary. The companion [`prelude_kernelbuild`] crate handles the
//! build-side work (compile .cu files, spawn Python DSL scripts,
//! generate dispatch tables). They share no code; the interface
//! between them is the shape of the generated dispatch function.
//!
//! ## Safety
//!
//! All TVM-FFI kernels expect device pointers and a matching CUDA
//! stream. The registry doesn't validate either — it's a thin
//! dispatcher, not a correctness checker. Consumers are responsible
//! for packing [`TVMFFIAny`] args with valid [`DLTensor`] pointers
//! and calling `set_stream` before invoking a kernel.

use std::ffi::c_void;

// ── TVM FFI type re-exports ─────────────────────────────────────────
//
// These come from the `tvm-static-ffi` crate. Re-exporting them from
// here means consumer code can pull everything under a single
// `use prelude_cutedsl::{...}` without also depending on
// `tvm-static-ffi` directly. The dispatch-table codegen assumes this
// convention too (it emits extern decls referring to a user-supplied
// path for `TVMFFIAny`, which consumers usually spell
// `prelude_cutedsl::TVMFFIAny`).

pub use tvm_static_ffi::{
    DLDataType, DLDevice, DLTensor, TVMFFIAny, TVMSafeCallFn, KDLBFLOAT, KDLCUDA, KDLFLOAT,
    KDLINT,
};

// ── Extern: TVM FFI runtime + CUDA driver bits ──────────────────────
//
// We need a handful of raw CUDA calls for arch probing and error
// reporting, and the single TVM-FFI call to wire a stream into the
// current thread-local env. Keeping them unsafe extern declarations
// rather than depending on the `cudarc` crate means consumers stay
// independent of which CUDA wrapper they prefer.

unsafe extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const std::ffi::c_char;
    fn cudaDeviceSynchronize() -> i32;
    fn TVMFFIEnvSetStream(
        device_type: i32,
        device_id: i32,
        stream: *mut c_void,
        out_original: *mut *mut c_void,
    ) -> i32;
}

const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

/// Probe the current CUDA device's compute capability and return it as
/// a single integer (major×10 + minor), e.g. `90` for Hopper SM90 or
/// `100` for Blackwell SM100. Returns `0` on failure (no CUDA, missing
/// driver, etc.) — callers should treat `0` as "unknown arch".
pub fn detect_gpu_arch() -> u32 {
    unsafe {
        let mut device = 0i32;
        if cudaGetDevice(&mut device) != 0 {
            return 0;
        }
        let mut major = 0i32;
        let mut minor = 0i32;
        if cudaDeviceGetAttribute(&mut major, CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR, device) != 0
        {
            return 0;
        }
        if cudaDeviceGetAttribute(&mut minor, CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR, device) != 0
        {
            return 0;
        }
        (major * 10 + minor) as u32
    }
}

// ── Dispatch table glue ─────────────────────────────────────────────

/// Signature of the generated dispatch lookup function every consumer
/// crate emits from its `build.rs`. The indirection lets one registry
/// type serve multiple consumers — each one passes its own lookup at
/// construction time.
///
/// * `kernel_type` — a free-form string tag identifying the kernel
///   family (e.g. `"kda_decode"`, `"gdn_decode_mtp"`). Consumers can
///   ignore it if their dispatch table only has one family.
/// * `key` — variant identifier (usually the full mangled name of the
///   compiled variant, like `"cula_kda_decode_small_varlen_h16_..."`).
/// * `arch` — compute capability the kernel was compiled for. Runtime
///   callers pass the detected GPU arch; the dispatch table matches
///   exact-equal, so an SM90 kernel isn't returned on SM100.
pub type LookupFn = fn(kernel_type: &str, key: &str, arch: u32) -> Option<TVMSafeCallFn>;

// ── Kernel registry ─────────────────────────────────────────────────

/// A runtime dispatcher that wraps a build-generated `lookup_dsl`
/// function. Typical usage is to park a `OnceLock<DslKernelRegistry>`
/// in the consumer crate and lazily initialise it on first kernel
/// call.
pub struct DslKernelRegistry {
    arch: u32,
    lookup: LookupFn,
}

impl DslKernelRegistry {
    /// Build a new registry bound to a specific lookup function.
    /// Probes the current CUDA device's compute capability once at
    /// construction time; callers that need multi-device support
    /// should construct a registry per device.
    pub fn new(lookup: LookupFn) -> Self {
        let arch = detect_gpu_arch();
        tracing::debug!("DslKernelRegistry: detected SM{arch}");
        Self { arch, lookup }
    }

    /// Compute capability the registry was bound to.
    pub fn arch(&self) -> u32 {
        self.arch
    }

    /// Look up a kernel by `(kernel_type, key)`, returning the
    /// TVM-FFI callable pointer if the dispatch table has a matching
    /// variant compiled for the current arch. `None` is a valid
    /// outcome — consumers typically fall back to a composed / CPU
    /// path when the kernel isn't available.
    pub fn get(&self, kernel_type: &str, key: &str) -> Option<TVMSafeCallFn> {
        (self.lookup)(kernel_type, key, self.arch)
    }

    /// Wire a CUDA stream into the TVM-FFI thread-local env. Must be
    /// called before invoking kernels that read the default-stream
    /// state from TVM-FFI (most CuTeDSL kernels do).
    pub fn set_stream(&self, device_id: i32, stream: *mut c_void) {
        unsafe {
            TVMFFIEnvSetStream(KDLCUDA, device_id, stream, std::ptr::null_mut());
        }
    }

    /// Invoke a kernel via the standard TVM-FFI safe-call calling
    /// convention. Consumers that need richer error handling can call
    /// the raw function pointer themselves (e.g. via
    /// [`tvm_static_ffi::call_tvm_ffi`]), but this helper is enough
    /// for most use cases.
    ///
    /// On failure we synchronize and query `cudaGetLastError()` so the
    /// returned error string contains the actual CUDA failure rather
    /// than a bare "TVM FFI internal failure".
    ///
    /// # Safety
    /// All `TVMFFIAny` args must contain valid device pointers for
    /// the current device, and `func` must be a valid TVM-FFI
    /// callable pointer returned from [`Self::get`].
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
                        format!(
                            "CUDA error {err}: {}",
                            std::ffi::CStr::from_ptr(ptr).to_string_lossy()
                        )
                    } else {
                        format!("CUDA error {err}")
                    }
                } else {
                    "TVM FFI internal failure".to_string()
                }
            };
            return Err(format!("DSL kernel call failed (code {ret}): {detail}"));
        }
        Ok(())
    }
}

// ── DLTensor helpers ────────────────────────────────────────────────

/// Build a [`DLTensor`] around an existing device pointer + shape +
/// strides. The caller owns the backing storage; this only creates a
/// non-owning view.
///
/// `shape` and `strides` must outlive the returned `DLTensor` because
/// DLTensor stores raw `*const i64` pointers into them.
pub fn make_dltensor(
    data: *mut c_void,
    device: DLDevice,
    dtype: DLDataType,
    shape: &[i64],
    strides: &[i64],
) -> DLTensor {
    DLTensor {
        data,
        device,
        ndim: shape.len() as i32,
        dtype,
        shape: shape.as_ptr(),
        strides: strides.as_ptr(),
        byte_offset: 0,
    }
}

/// Compute C-contiguous strides in element counts for a given shape.
/// Equivalent to numpy's `a.strides / a.itemsize` for a contiguous
/// array.
pub fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

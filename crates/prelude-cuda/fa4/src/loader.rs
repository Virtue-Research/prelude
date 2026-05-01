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

/// Dtype for kernel dispatch. Maps to DLDataType codes.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
pub enum KernelDtype {
    BF16 = 0,
    FP16 = 1,
}

/// Key for looking up a kernel variant.
///
/// Mirrors the compile_key dimensions from upstream `flash_attn/cute/interface.py`.
/// Every dimension that can produce a different compiled kernel must be here.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct KernelKey {
    pub head_dim: u32,
    /// Value head dimension. Usually = head_dim. Different for DeepSeek MLA (192, 128).
    pub head_dim_v: u32,
    pub gqa_ratio: u32,
    pub causal: bool,
    pub window: bool,
    /// GQA packing optimization — auto-enable when gqa_ratio > 1.
    pub pack_gqa: bool,
    /// Attention logit soft-capping, stored as `f32::to_bits()`. 0 = disabled.
    /// Used by Gemma models (30.0 for Gemma2, 50.0 for Gemma3).
    pub softcap_bits: u32,
    /// Paged KV cache: K/V read from block-indexed paged cache.
    pub paged: bool,
    /// If paged: true = cp.async (page_size != tile_n), false = TMA (page_size == tile_n).
    /// Ignored when paged=false.
    pub paged_non_tma: bool,
    /// Data type: BF16 or FP16.
    pub dtype: KernelDtype,
    /// Whether seqused_q is passed (prefix cache Q trimming).
    pub has_seqused_q: bool,
}

impl KernelKey {
    /// Convenience: create a key with defaults (bf16, head_dim_v=head_dim,
    /// pack_gqa auto-derived, no softcap, non-paged, no seqused_q).
    pub fn new(head_dim: u32, gqa_ratio: u32, causal: bool, window: bool) -> Self {
        Self {
            head_dim,
            head_dim_v: head_dim,
            gqa_ratio,
            causal,
            window,
            pack_gqa: gqa_ratio > 1,
            softcap_bits: 0,
            paged: false,
            paged_non_tma: false,
            dtype: KernelDtype::BF16,
            has_seqused_q: false,
        }
    }

    /// Set value head dimension (for DeepSeek MLA shape).
    pub fn with_head_dim_v(mut self, head_dim_v: u32) -> Self {
        self.head_dim_v = head_dim_v;
        self
    }

    /// Create a key with explicit softcap value.
    pub fn with_softcap(mut self, softcap: Option<f32>) -> Self {
        self.softcap_bits = softcap.map_or(0, |v| v.to_bits());
        self
    }

    /// Create a key for paged KV attention (TMA path, page_size == tile_n).
    pub fn with_paged(mut self, paged: bool) -> Self {
        self.paged = paged;
        self
    }

    /// Set paged non-TMA (cp.async path, page_size != tile_n).
    pub fn with_paged_non_tma(mut self, non_tma: bool) -> Self {
        self.paged_non_tma = non_tma;
        self
    }

    /// Set dtype.
    pub fn with_dtype(mut self, dtype: KernelDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set whether seqused_q is passed.
    pub fn with_seqused_q(mut self, has_seqused_q: bool) -> Self {
        self.has_seqused_q = has_seqused_q;
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

unsafe extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const std::ffi::c_char;
    fn cudaDeviceSynchronize() -> i32;
    /// Set the CUDA stream for TVM FFI kernel calls.
    /// Kernels call TVMFFIEnvGetStream() internally to get the stream.
    fn TVMFFIEnvSetStream(
        device_type: i32, device_id: i32,
        stream: *mut c_void, out_original: *mut *mut c_void,
    ) -> i32;
    /// Extract error message from TVM FFI's thread-local error state.
    /// Defined in src/tvm_error_helper.cc.
    fn tvm_static_ffi_get_last_error(out_len: *mut usize) -> *const u8;
}

const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

/// Detect the current CUDA device's SM architecture (e.g. 90, 120).
/// Returns 0 if detection fails (no GPU, driver not loaded, etc.).
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

    /// Set the CUDA stream for subsequent kernel calls.
    /// Kernels use TVMFFIEnvGetStream() internally to retrieve it.
    pub fn set_stream(&self, device_id: i32, stream: *mut c_void) {
        const KDLCUDA_DEVICE: i32 = 2;
        unsafe {
            TVMFFIEnvSetStream(KDLCUDA_DEVICE, device_id, stream, std::ptr::null_mut());
        }
    }

    /// Call a statically linked kernel via TVM FFI packed calling convention.
    ///
    /// # Safety
    /// All TVMFFIAny args must contain valid device pointers.
    /// Caller must call `set_stream()` before calling this.
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
                // Try TVM FFI error message first
                let mut msg_len: usize = 0;
                let msg_ptr = tvm_static_ffi_get_last_error(&mut msg_len);
                let tvm_msg = if !msg_ptr.is_null() && msg_len > 0 {
                    String::from_utf8_lossy(std::slice::from_raw_parts(msg_ptr, msg_len)).into_owned()
                } else {
                    String::new()
                };

                cudaDeviceSynchronize();
                let err = cudaGetLastError();
                if !tvm_msg.is_empty() {
                    tvm_msg
                } else if err != 0 {
                    let ptr = cudaGetErrorString(err);
                    if !ptr.is_null() {
                        format!("CUDA error {err}: {}", std::ffi::CStr::from_ptr(ptr).to_string_lossy())
                    } else {
                        format!("CUDA error {err}")
                    }
                } else {
                    "TVM FFI internal failure (no error message available)".to_string()
                }
            };
            return Err(format!("FA4 kernel call failed (code {ret}): {detail}"));
        }
        Ok(())
    }
}

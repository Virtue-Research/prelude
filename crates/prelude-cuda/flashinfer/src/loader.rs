//! Kernel dispatch: statically linked FlashInfer kernel variants.
//!
//! Each FlashInfer module exports 2-3 TVM FFI functions (plan, run/ragged_run/paged_run).
//! The compile script renames symbols per variant to avoid collisions in static linking.
//!
//! build.rs generates `fi_dispatch.rs` with extern "C" declarations and lookup functions.

use crate::types::TVMFFIAny;
use std::ffi::c_void;

/// TVM safe call function signature.
pub type TVMSafeCallFn =
    unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;

/// Data type for kernel dispatch.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
pub enum KernelDtype {
    BF16 = 0,
    FP16 = 1,
}

/// FlashInfer attention backend.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
pub enum Backend {
    /// FA2 kernels (SM80+, Ampere/Ada)
    FA2 = 0,
    /// FA3 kernels (SM90+, Hopper)
    FA3 = 1,
}

/// Mask mode for attention.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(i64)]
pub enum MaskMode {
    NonCausal = 0,
    Causal = 1,
    CustomMask = 2,
    MultiItemScoring = 3,
}

/// Key for looking up a batch prefill kernel variant.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PrefillKey {
    pub dtype: KernelDtype,
    pub head_dim_qk: u32,
    pub head_dim_vo: u32,
    pub sliding_window: bool,
    pub logits_soft_cap: bool,
    pub backend: Backend,
}

/// Key for FP8 E4M3 prefill (SM90 only).
/// Both Q and KV are FP8 E4M3; output is BF16. Symmetric head_dim only.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FP8PrefillKey {
    pub head_dim: u32,
    pub sliding_window: bool,
}

/// Key for looking up a batch decode kernel variant.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DecodeKey {
    pub dtype: KernelDtype,
    pub head_dim_qk: u32,
    pub head_dim_vo: u32,
    pub sliding_window: bool,
    pub logits_soft_cap: bool,
}

/// Key for MLA decode kernel (DeepSeek V2/V3).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MLADecodeKey {
    pub dtype: KernelDtype,
    pub head_dim_ckv: u32,
    pub head_dim_kpe: u32,
}

/// Key for MLA paged attention kernel.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MLAPagedKey {
    pub dtype: KernelDtype,
    pub head_dim_ckv: u32,
    pub head_dim_kpe: u32,
}

/// Function pointers for a batch prefill variant.
/// Each variant exports: plan, ragged_run, paged_run.
pub struct PrefillVariant {
    pub plan: TVMSafeCallFn,
    pub ragged_run: TVMSafeCallFn,
    pub paged_run: TVMSafeCallFn,
}

/// Function pointers for a batch decode variant.
/// Each variant exports: plan, run.
pub struct DecodeVariant {
    pub plan: TVMSafeCallFn,
    pub run: TVMSafeCallFn,
}

/// Function pointers for MLA decode.
pub struct MLADecodeVariant {
    pub plan: TVMSafeCallFn,
    pub run: TVMSafeCallFn,
}

/// Function pointers for MLA paged attention.
pub struct MLAPagedVariant {
    pub plan: TVMSafeCallFn,
    pub run: TVMSafeCallFn,
}

// Include the build.rs-generated dispatch table
include!(concat!(env!("OUT_DIR"), "/fi_dispatch.rs"));

// ── GPU arch detection ──────────────────────────────────────────────

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
    fn tvm_static_ffi_get_last_error(out_len: *mut usize) -> *const u8;
}

const CUDA_DEV_ATTR_MAJOR: i32 = 75;
const CUDA_DEV_ATTR_MINOR: i32 = 76;

fn detect_gpu_arch() -> u32 {
    unsafe {
        let mut device = 0i32;
        if cudaGetDevice(&mut device) != 0 { return 0; }
        let mut major = 0i32;
        let mut minor = 0i32;
        if cudaDeviceGetAttribute(&mut major, CUDA_DEV_ATTR_MAJOR, device) != 0 { return 0; }
        if cudaDeviceGetAttribute(&mut minor, CUDA_DEV_ATTR_MINOR, device) != 0 { return 0; }
        (major * 10 + minor) as u32
    }
}

/// Collection of statically linked FlashInfer kernels.
pub struct KernelRegistry {
    arch: u32,
}

impl KernelRegistry {
    pub fn new() -> Self {
        let arch = detect_gpu_arch();
        tracing::debug!("FlashInfer KernelRegistry: detected SM{arch}");
        Self { arch }
    }

    pub fn with_arch(arch: u32) -> Self {
        Self { arch }
    }

    pub fn arch(&self) -> u32 { self.arch }

    /// Select backend based on GPU arch: FA3 for SM90 (Hopper), FA2 elsewhere.
    ///
    /// FA3 cubins are compiled exclusively for `sm_90`; SM100/103 (Blackwell)
    /// would trigger "no kernel image is available for execution on the device"
    /// at runtime because neither the `sm_90` cubin nor the `sm_80` PTX is a
    /// valid match. FA2 has an `sm_80` PTX fallback that JIT-compiles for any
    /// higher arch, so Blackwell stays on FA2 until we ship FA3 cubins for it.
    pub fn default_backend(&self) -> Backend {
        if self.arch == 90 { Backend::FA3 } else { Backend::FA2 }
    }

    /// Look up a batch prefill variant.
    pub fn get_prefill(&self, key: &PrefillKey) -> Option<PrefillVariant> {
        lookup_prefill(key)
    }

    /// Look up an FP8 E4M3 prefill variant (SM90+ only).
    /// Q and KV are both FP8 E4M3, output is BF16.
    pub fn get_fp8_prefill(&self, key: &FP8PrefillKey) -> Option<PrefillVariant> {
        lookup_prefill_fp8(key)
    }

    /// Look up a batch decode variant.
    pub fn get_decode(&self, key: &DecodeKey) -> Option<DecodeVariant> {
        lookup_decode(key)
    }

    /// Look up an MLA decode variant (DeepSeek V2/V3).
    pub fn get_mla_decode(&self, key: &MLADecodeKey) -> Option<MLADecodeVariant> {
        lookup_mla_decode(key)
    }

    /// Look up an MLA paged attention variant.
    pub fn get_mla_paged(&self, key: &MLAPagedKey) -> Option<MLAPagedVariant> {
        lookup_mla_paged(key)
    }

    /// Look up a utility kernel by name (e.g., "softmax", "rmsnorm", "apply_rope").
    pub fn get_utility(&self, name: &str) -> Option<TVMSafeCallFn> {
        lookup_utility(name)
    }

    /// Set CUDA stream for subsequent kernel calls.
    pub fn set_stream(&self, device_id: i32, stream: *mut c_void) {
        unsafe {
            TVMFFIEnvSetStream(2, device_id, stream, std::ptr::null_mut());
        }
    }

    /// Call a TVM FFI function with packed args.
    ///
    /// # Safety
    /// All TVMFFIAny args must contain valid device pointers.
    pub unsafe fn call(
        &self, func: TVMSafeCallFn, args: &[TVMFFIAny],
    ) -> Result<TVMFFIAny, String> {
        let mut result = TVMFFIAny::none();
        let ret = unsafe {
            func(std::ptr::null_mut(), args.as_ptr(), args.len() as i32, &mut result)
        };
        if ret != 0 {
            let detail = unsafe {
                // Try to get TVM FFI error message first
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
            return Err(format!("FlashInfer kernel call failed (code {ret}): {detail}"));
        }
        Ok(result)
    }
}

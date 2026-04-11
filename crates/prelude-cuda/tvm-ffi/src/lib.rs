//! Minimal ABI for calling statically-linked CuTeDSL / cutlass-dsl AOT
//! kernels from Rust.
//!
//! Scope (deliberately narrow):
//! - DLPack types (`DLDevice`, `DLDataType`, `DLTensor`)
//! - TVM FFI "Any" packed value (`TVMFFIAny`)
//! - TVM `SafeCall` function signature (`TVMSafeCallFn`)
//! - Helper `call_tvm_ffi(func, args)` that invokes a statically-linked
//!   `__tvm_ffi_<name>` symbol and extracts the error message on failure.
//!
//! This crate does NOT provide a TVM runtime, function registry, or object
//! system — that is the job of the upstream `tvm-ffi` crate, which uses
//! dynamic loading (`libtvm_ffi.so`). The two crates serve different use
//! cases and can coexist.
//!
//! The C++ tvm_ffi library is compiled once by this crate's build.rs so the
//! error-retrieval helper has a C ABI to call into.

use std::ffi::c_void;

// ── DLPack types ────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    pub device_type: i32, // kDLCUDA = 2
    pub device_id: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    pub code: u8,  // kDLFloat=2, kDLBfloat=4, kDLInt=0, kDLUInt=1
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

// DLPack constants
pub const KDLCUDA: i32 = 2;
pub const KDLCPU: i32 = 1;
pub const KDLBFLOAT: u8 = 4;
pub const KDLFLOAT: u8 = 2;
pub const KDLINT: u8 = 0;
pub const KDLUINT: u8 = 1;

// ── TVM FFI types ───────────────────────────────────────────────────

/// TVMFFIAny: 16-byte packed value (type_index + value union).
/// Layout: [type_index: i32, padding: u32, value: u64]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TVMFFIAny {
    pub type_index: i32,
    pub zero_padding: u32,
    pub v_union: u64, // reinterpreted as i64/f64/pointer depending on type_index
}

// TVMFFITypeIndex constants
pub const KTVMFFI_NONE: i32 = 0;
pub const KTVMFFI_INT: i32 = 1;
pub const KTVMFFI_BOOL: i32 = 2;
pub const KTVMFFI_FLOAT: i32 = 3;
pub const KTVMFFI_OPAQUE_PTR: i32 = 4;
pub const KTVMFFI_DLTENSOR_PTR: i32 = 7;

/// TVM safe call function signature (TVMFFISafeCallType).
/// Each kernel variant exports `__tvm_ffi_{variant_name}` with this signature.
pub type TVMSafeCallFn =
    unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;

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

    pub fn float64(val: f64) -> Self {
        Self { type_index: KTVMFFI_FLOAT, zero_padding: 0, v_union: val.to_bits() }
    }

    pub fn int32(val: i32) -> Self {
        Self { type_index: KTVMFFI_INT, zero_padding: 0, v_union: val as i64 as u64 }
    }

    pub fn int64(val: i64) -> Self {
        Self { type_index: KTVMFFI_INT, zero_padding: 0, v_union: val as u64 }
    }

    pub fn bool_val(val: bool) -> Self {
        Self { type_index: KTVMFFI_BOOL, zero_padding: 0, v_union: val as u64 }
    }

    pub fn opaque_ptr(ptr: *mut c_void) -> Self {
        Self { type_index: KTVMFFI_OPAQUE_PTR, zero_padding: 0, v_union: ptr as u64 }
    }
}

// ── Error helper ────────────────────────────────────────────────────

unsafe extern "C" {
    fn tvm_static_ffi_get_last_error(out_len: *mut usize) -> *const u8;
}

/// Call a TVM FFI function and convert errors to a Rust Result.
///
/// Returns Ok(()) on success, Err(String) with the TVM error message on failure.
///
/// # Safety
/// The caller must ensure `func` is a valid TVM SafeCall function pointer and
/// that the lifetimes of all `TVMFFIAny` contents in `args` outlive the call.
pub unsafe fn call_tvm_ffi(
    func: TVMSafeCallFn,
    args: &[TVMFFIAny],
) -> Result<(), String> {
    let mut ret = TVMFFIAny::none();
    // SAFETY: caller contract on `func` being a valid SafeCall entry point
    // and on `args` outliving the call.
    let rc = unsafe {
        func(
            std::ptr::null_mut(),
            args.as_ptr(),
            args.len() as i32,
            &mut ret,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        // SAFETY: `tvm_static_ffi_get_last_error` returns a pointer into
        // TVM's thread-local error state, valid until the next TVM call on
        // this thread. We read it into an owned String immediately.
        let (msg_ptr, msg_len) = unsafe {
            let mut len: usize = 0;
            let p = tvm_static_ffi_get_last_error(&mut len);
            (p, len)
        };
        let msg = if !msg_ptr.is_null() && msg_len > 0 {
            // SAFETY: valid UTF-8 slice owned by TVM's thread-local buffer,
            // copied into an owned String before the next FFI call.
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(msg_ptr, msg_len))
                    .to_string()
            }
        } else {
            format!("TVM FFI call failed (rc={rc})")
        };
        Err(msg)
    }
}

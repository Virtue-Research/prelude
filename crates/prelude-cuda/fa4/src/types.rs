//! TVM FFI and DLPack type definitions for calling FA4 kernels.
//!
//! These mirror the C structs from `tvm/ffi/c_api.h` and `dlpack/dlpack.h`.

use std::ffi::c_void;

// ---- DLPack types ----

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
pub const KDLBFLOAT: u8 = 4;
pub const KDLFLOAT: u8 = 2;
pub const KDLINT: u8 = 0;

// ---- TVM FFI types ----

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
pub const KTVMFFI_FLOAT: i32 = 3;
pub const KTVMFFI_OPAQUE_PTR: i32 = 4;
pub const KTVMFFI_DLTENSOR_PTR: i32 = 7;

impl TVMFFIAny {
    /// Create a None value.
    pub fn none() -> Self {
        Self {
            type_index: KTVMFFI_NONE,
            zero_padding: 0,
            v_union: 0,
        }
    }

    /// Create a DLTensor pointer value.
    pub fn dltensor(tensor: *const DLTensor) -> Self {
        Self {
            type_index: KTVMFFI_DLTENSOR_PTR,
            zero_padding: 0,
            v_union: tensor as u64,
        }
    }

    /// Create a float32 value.
    pub fn float32(val: f32) -> Self {
        // TVM FFI stores floats as f64 in the union
        let f64_val = val as f64;
        Self {
            type_index: KTVMFFI_FLOAT,
            zero_padding: 0,
            v_union: f64_val.to_bits(),
        }
    }

    /// Create an opaque pointer value (for CUstream).
    pub fn opaque_ptr(ptr: *mut c_void) -> Self {
        Self {
            type_index: KTVMFFI_OPAQUE_PTR,
            zero_padding: 0,
            v_union: ptr as u64,
        }
    }

    /// Create an int32 value.
    pub fn int32(val: i32) -> Self {
        Self {
            type_index: KTVMFFI_INT,
            zero_padding: 0,
            v_union: val as i64 as u64,
        }
    }
}

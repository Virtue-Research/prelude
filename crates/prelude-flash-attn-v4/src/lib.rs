//! Flash Attention v4 (CuTeDSL AOT) — statically linked kernels.
//!
//! Kernel variants are compiled into .o files with unique symbols
//! (`__tvm_ffi_{variant_name}`) and statically linked into the binary.
//! No dlopen, no runtime .so files needed.
//!
//! # Architecture
//!
//! ```text
//! Build time:
//!   Python (CuTeDSL) → AOT compile → 72 kernel .o files (unique symbols each)
//!   build.rs → ar rcs libfa4_kernels.a *.o → static link with +whole-archive
//!   build.rs → generate fa4_dispatch.rs (extern "C" decls + lookup table)
//!   cc crate → compile vendored tvm_ffi C++ → libtvm_ffi_static.a
//!
//! Runtime:
//!   KernelRegistry::new()  // stateless, all kernels compiled in
//!   registry.get(&key)     // → Option<TVMSafeCallFn>
//!
//!   fa4_varlen_fwd(registry, func, q, k, v, ...)
//!     → pack 17 args as TVMFFIAny array
//!     → func(NULL, args, 17, &result)  // direct call, no dlopen
//! ```
//!
//! # Usage from `attn/flash_v4.rs`:
//! ```ignore
//! let registry = prelude_flash_attn_v4::KernelRegistry::new();
//! let key = KernelKey::new(128, 4, true, false);
//! let func = registry.get(&key).expect("FA4 kernel variant not found");
//!
//! unsafe {
//!     prelude_flash_attn_v4::fa4_varlen_fwd(
//!         &registry, func,
//!         q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
//!         softmax_scale, stream,
//!         cu_seqlens_q_ptr, cu_seqlens_k_ptr,
//!         &q_shape, &k_shape, &o_shape, &lse_shape,
//!         &cu_seqlens_shape,
//!         device_id,
//!     )?;
//! }
//! ```

pub mod loader;
pub mod types;

use std::ffi::c_void;
use types::*;

pub use loader::{KernelKey, KernelRegistry, TVMSafeCallFn};

/// Call FA4 varlen forward attention via TVM FFI.
///
/// # Arguments
/// * `registry` - Kernel registry (stateless, holds no state)
/// * `func` - Kernel function pointer from `registry.get(&key)`
/// * `q_ptr` - Query GPU pointer, shape [total_q, num_heads_q, head_dim]
/// * `k_ptr` - Key GPU pointer, shape [total_k, num_heads_k, head_dim]
/// * `v_ptr` - Value GPU pointer, shape [total_k, num_heads_k, head_dim]
/// * `o_ptr` - Output GPU pointer, shape [total_q, num_heads_q, head_dim]
/// * `lse_ptr` - Log-sum-exp GPU pointer, shape [num_heads_q, total_q] (or null)
/// * `softmax_scale` - 1.0 / sqrt(head_dim)
/// * `stream` - CUDA stream handle
/// * `cu_seqlens_q_ptr` - Cumulative Q sequence lengths, shape [batch+1], I32
/// * `cu_seqlens_k_ptr` - Cumulative K sequence lengths, shape [batch+1], I32
/// * Shapes are passed as [ndim] arrays of i64
/// * `device_id` - CUDA device ID
/// * `seqused_q_ptr` - Optional per-batch Q usage, shape [batch_size], I32
/// * `seqused_k_ptr` - Optional per-batch K usage, shape [batch_size], I32
///
/// # Safety
/// All pointers must be valid CUDA device pointers. Stream must be valid.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fa4_varlen_fwd(
    registry: &KernelRegistry,
    func: TVMSafeCallFn,
    // Tensor data pointers
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    v_ptr: *mut c_void,
    o_ptr: *mut c_void,
    lse_ptr: *mut c_void, // can be null
    // Scalars
    softmax_scale: f32,
    stream: *mut c_void,
    // Varlen
    cu_seqlens_q_ptr: *mut c_void,
    cu_seqlens_k_ptr: *mut c_void,
    // Shapes (caller provides)
    q_shape: &[i64],   // [total_q, num_heads_q, head_dim]
    k_shape: &[i64],   // [total_k, num_heads_k, head_dim]
    o_shape: &[i64],   // [total_q, num_heads_q, head_dim]
    lse_shape: &[i64], // [num_heads, total_q]
    cu_seqlens_shape: &[i64], // [batch_size + 1]
    device_id: i32,
    // Optional window attention
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    // Optional seqused tensors (prefix cache optimization)
    seqused_q_ptr: Option<*mut c_void>,
    seqused_k_ptr: Option<*mut c_void>,
) -> Result<(), String> {
    let device = DLDevice {
        device_type: KDLCUDA,
        device_id,
    };
    let bf16 = DLDataType {
        code: KDLBFLOAT,
        bits: 16,
        lanes: 1,
    };
    let f32_dtype = DLDataType {
        code: KDLFLOAT,
        bits: 32,
        lanes: 1,
    };
    let i32_dtype = DLDataType {
        code: KDLINT,
        bits: 32,
        lanes: 1,
    };

    // Q: [total_q, num_heads_q, head_dim]
    let q_strides = contiguous_strides(q_shape);
    let dl_q = DLTensor {
        data: q_ptr,
        device,
        ndim: q_shape.len() as i32,
        dtype: bf16,
        shape: q_shape.as_ptr(),
        strides: q_strides.as_ptr(),
        byte_offset: 0,
    };

    // K: [total_k, num_heads_k, head_dim]
    let k_strides = contiguous_strides(k_shape);
    let dl_k = DLTensor {
        data: k_ptr,
        device,
        ndim: k_shape.len() as i32,
        dtype: bf16,
        shape: k_shape.as_ptr(),
        strides: k_strides.as_ptr(),
        byte_offset: 0,
    };

    // V: same shape as K
    let v_strides = contiguous_strides(k_shape);
    let dl_v = DLTensor {
        data: v_ptr,
        device,
        ndim: k_shape.len() as i32,
        dtype: bf16,
        shape: k_shape.as_ptr(),
        strides: v_strides.as_ptr(),
        byte_offset: 0,
    };

    // O: same shape as Q
    let o_strides = contiguous_strides(o_shape);
    let dl_o = DLTensor {
        data: o_ptr,
        device,
        ndim: o_shape.len() as i32,
        dtype: bf16,
        shape: o_shape.as_ptr(),
        strides: o_strides.as_ptr(),
        byte_offset: 0,
    };

    // LSE: [num_heads, total_q] F32 (optional)
    let lse_strides = contiguous_strides(lse_shape);
    let dl_lse = DLTensor {
        data: lse_ptr,
        device,
        ndim: lse_shape.len() as i32,
        dtype: f32_dtype,
        shape: lse_shape.as_ptr(),
        strides: lse_strides.as_ptr(),
        byte_offset: 0,
    };

    // cu_seqlens_q: [batch+1] I32
    let cu_q_strides = contiguous_strides(cu_seqlens_shape);
    let dl_cu_q = DLTensor {
        data: cu_seqlens_q_ptr,
        device,
        ndim: 1,
        dtype: i32_dtype,
        shape: cu_seqlens_shape.as_ptr(),
        strides: cu_q_strides.as_ptr(),
        byte_offset: 0,
    };

    // cu_seqlens_k: [batch+1] I32
    let dl_cu_k = DLTensor {
        data: cu_seqlens_k_ptr,
        device,
        ndim: 1,
        dtype: i32_dtype,
        shape: cu_seqlens_shape.as_ptr(),
        strides: cu_q_strides.as_ptr(),
        byte_offset: 0,
    };

    // Build optional seqused DLTensors. Shape = [batch_size] where batch_size = cu_seqlens[0] - 1.
    let batch_size = cu_seqlens_shape[0] - 1;
    let seqused_shape: [i64; 1] = [batch_size];
    let seqused_strides: [i64; 1] = [1];

    let dl_seqused_q = seqused_q_ptr.map(|ptr| DLTensor {
        data: ptr,
        device,
        ndim: 1,
        dtype: i32_dtype,
        shape: seqused_shape.as_ptr(),
        strides: seqused_strides.as_ptr(),
        byte_offset: 0,
    });

    let dl_seqused_k = seqused_k_ptr.map(|ptr| DLTensor {
        data: ptr,
        device,
        ndim: 1,
        dtype: i32_dtype,
        shape: seqused_shape.as_ptr(),
        strides: seqused_strides.as_ptr(),
        byte_offset: 0,
    });

    // Pack arguments in the exact order from args_spec:
    // mQ, mK, mV, mO, mLSE, softmax_scale, stream,
    // mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK, mPageTable,
    // window_size_left, window_size_right, learnable_sink,
    // blocksparse_tensors, aux_tensors
    let mut args: [TVMFFIAny; 17] = [
        TVMFFIAny::dltensor(&dl_q),              // mQ
        TVMFFIAny::dltensor(&dl_k),              // mK
        TVMFFIAny::dltensor(&dl_v),              // mV
        TVMFFIAny::dltensor(&dl_o),              // mO
        if lse_ptr.is_null() {                    // mLSE
            TVMFFIAny::none()
        } else {
            TVMFFIAny::dltensor(&dl_lse)
        },
        TVMFFIAny::float32(softmax_scale),        // softmax_scale
        TVMFFIAny::opaque_ptr(stream),            // stream
        TVMFFIAny::dltensor(&dl_cu_q),            // mCuSeqlensQ
        TVMFFIAny::dltensor(&dl_cu_k),            // mCuSeqlensK
        match &dl_seqused_q {                     // mSeqUsedQ
            Some(t) => TVMFFIAny::dltensor(t),
            None => TVMFFIAny::none(),
        },
        match &dl_seqused_k {                     // mSeqUsedK
            Some(t) => TVMFFIAny::dltensor(t),
            None => TVMFFIAny::none(),
        },
        TVMFFIAny::none(),                        // mPageTable
        match window_size_left {                  // window_size_left
            Some(v) => TVMFFIAny::int32(v),
            None => TVMFFIAny::none(),
        },
        match window_size_right {                 // window_size_right
            Some(v) => TVMFFIAny::int32(v),
            None => TVMFFIAny::none(),
        },
        TVMFFIAny::none(),                        // learnable_sink
        TVMFFIAny::none(),                        // blocksparse_tensors
        TVMFFIAny::none(),                        // aux_tensors
    ];

    registry.call_kernel(func, &mut args)
}

/// Compute contiguous strides for a shape (row-major, element strides not byte strides).
fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

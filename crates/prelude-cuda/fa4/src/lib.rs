//! Flash Attention v4 (CuTeDSL AOT) — statically linked kernels.
//!
//! TVM FFI packed calling convention. AOT kernels accept the full arg list
//! including None slots — the TVM FFI runtime skips None args internally.
//! Stream is set via TVMFFIEnvSetStream before the call.

pub mod loader;
pub mod types;

use std::ffi::c_void;
use types::*;

pub use loader::{KernelDtype, KernelKey, KernelRegistry, TVMSafeCallFn};

// ── Shared helpers ─────────────────────────────────────────────────────

fn half_dl_dtype(dtype: KernelDtype) -> DLDataType {
    match dtype {
        KernelDtype::BF16 => DLDataType {
            code: KDLBFLOAT,
            bits: 16,
            lanes: 1,
        },
        KernelDtype::FP16 => DLDataType {
            code: KDLFLOAT,
            bits: 16,
            lanes: 1,
        },
    }
}

fn make_aux_dtypes() -> (DLDataType, DLDataType) {
    let f32_dt = DLDataType {
        code: KDLFLOAT,
        bits: 32,
        lanes: 1,
    };
    let i32_dt = DLDataType {
        code: KDLINT,
        bits: 32,
        lanes: 1,
    };
    (f32_dt, i32_dt)
}

fn make_dltensor(
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

/// Compute contiguous strides for a shape (row-major, element strides not byte strides).
fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Build the full 16-arg TVM FFI array.
///
/// Arg order matches `cute.compile()` from upstream interface.py (minus stream,
/// which is set via TVMFFIEnvSetStream). None slots are preserved — TVM FFI
/// runtime skips them internally.
///
/// ```text
/// [0]  mQ              (DLTensor)
/// [1]  mK              (DLTensor)
/// [2]  mV              (DLTensor)
/// [3]  mO              (DLTensor)
/// [4]  mLSE            (DLTensor or None)
/// [5]  softmax_scale   (float32)
/// [6]  mCuSeqlensQ     (DLTensor or None)
/// [7]  mCuSeqlensK     (DLTensor or None)
/// [8]  mSeqUsedQ       (DLTensor or None)
/// [9]  mSeqUsedK       (DLTensor or None)
/// [10] mPageTable      (DLTensor or None)
/// [11] window_left     (int32 or None)
/// [12] window_right    (int32 or None)
/// [13] learnable_sink  (None)
/// [14] blocksparse     (None)
/// [15] aux_tensors     (None)
/// ```

// ── Non-paged varlen forward ───────────────────────────────────────────

/// Call FA4 varlen forward attention via TVM FFI (non-paged).
///
/// Q/K/V strides are caller-provided (element strides, not byte strides) so
/// strided views from fused QKV projection work without a prior `.contiguous()`.
/// The FA4 kernel only requires `stride(-1) == 1` on Q/K/V; other dims are free.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fa4_varlen_fwd(
    registry: &KernelRegistry,
    func: TVMSafeCallFn,
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    v_ptr: *mut c_void,
    o_ptr: *mut c_void,
    lse_ptr: *mut c_void,
    softmax_scale: f32,
    stream: *mut c_void,
    cu_seqlens_q_ptr: *mut c_void,
    cu_seqlens_k_ptr: *mut c_void,
    q_shape: &[i64],
    q_strides: &[i64],
    k_shape: &[i64],
    k_strides: &[i64],
    v_shape: &[i64],
    v_strides: &[i64],
    o_shape: &[i64],
    lse_shape: &[i64],
    cu_seqlens_shape: &[i64],
    device_id: i32,
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    seqused_q_ptr: Option<*mut c_void>,
    seqused_k_ptr: Option<*mut c_void>,
    dtype: KernelDtype,
) -> Result<(), String> {
    let device = DLDevice {
        device_type: KDLCUDA,
        device_id,
    };
    let half_dt = half_dl_dtype(dtype);
    let (f32_dtype, i32_dtype) = make_aux_dtypes();

    debug_assert_eq!(
        q_strides.last().copied(),
        Some(1),
        "FA4 requires Q stride(-1) == 1"
    );
    debug_assert_eq!(
        k_strides.last().copied(),
        Some(1),
        "FA4 requires K stride(-1) == 1"
    );
    debug_assert_eq!(
        v_strides.last().copied(),
        Some(1),
        "FA4 requires V stride(-1) == 1"
    );

    let dl_q = make_dltensor(q_ptr, device, half_dt, q_shape, q_strides);
    let dl_k = make_dltensor(k_ptr, device, half_dt, k_shape, k_strides);
    let dl_v = make_dltensor(v_ptr, device, half_dt, v_shape, v_strides);

    let o_strides = contiguous_strides(o_shape);
    let dl_o = make_dltensor(o_ptr, device, half_dt, o_shape, &o_strides);

    let lse_strides = contiguous_strides(lse_shape);
    let dl_lse = make_dltensor(lse_ptr, device, f32_dtype, lse_shape, &lse_strides);

    let cu_q_strides = contiguous_strides(cu_seqlens_shape);
    let dl_cu_q = make_dltensor(
        cu_seqlens_q_ptr,
        device,
        i32_dtype,
        cu_seqlens_shape,
        &cu_q_strides,
    );
    let dl_cu_k = make_dltensor(
        cu_seqlens_k_ptr,
        device,
        i32_dtype,
        cu_seqlens_shape,
        &cu_q_strides,
    );

    let batch_size = cu_seqlens_shape[0] - 1;
    let seqused_shape: [i64; 1] = [batch_size];
    let seqused_strides: [i64; 1] = [1];

    let dl_seqused_q = seqused_q_ptr
        .map(|ptr| make_dltensor(ptr, device, i32_dtype, &seqused_shape, &seqused_strides));
    let dl_seqused_k = seqused_k_ptr
        .map(|ptr| make_dltensor(ptr, device, i32_dtype, &seqused_shape, &seqused_strides));

    registry.set_stream(device_id, stream);

    let mut args: [TVMFFIAny; 16] = [
        TVMFFIAny::dltensor(&dl_q), // 0: mQ
        TVMFFIAny::dltensor(&dl_k), // 1: mK
        TVMFFIAny::dltensor(&dl_v), // 2: mV
        TVMFFIAny::dltensor(&dl_o), // 3: mO
        if lse_ptr.is_null() {
            // 4: mLSE
            TVMFFIAny::none()
        } else {
            TVMFFIAny::dltensor(&dl_lse)
        },
        TVMFFIAny::float32(softmax_scale), // 5: softmax_scale
        TVMFFIAny::dltensor(&dl_cu_q),     // 6: mCuSeqlensQ
        TVMFFIAny::dltensor(&dl_cu_k),     // 7: mCuSeqlensK
        match &dl_seqused_q {
            // 8: mSeqUsedQ
            Some(t) => TVMFFIAny::dltensor(t),
            None => TVMFFIAny::none(),
        },
        match &dl_seqused_k {
            // 9: mSeqUsedK
            Some(t) => TVMFFIAny::dltensor(t),
            None => TVMFFIAny::none(),
        },
        TVMFFIAny::none(), // 10: mPageTable
        match window_size_left {
            // 11: window_size_left
            Some(v) => TVMFFIAny::int32(v),
            None => TVMFFIAny::none(),
        },
        match window_size_right {
            // 12: window_size_right
            Some(v) => TVMFFIAny::int32(v),
            None => TVMFFIAny::none(),
        },
        TVMFFIAny::none(), // 13: learnable_sink
        TVMFFIAny::none(), // 14: blocksparse_tensors
        TVMFFIAny::none(), // 15: aux_tensors
    ];

    unsafe { registry.call_kernel(func, &mut args) }
}

// ── Paged varlen forward ───────────────────────────────────────────────

/// Call FA4 varlen forward attention with paged KV cache via TVM FFI.
///
/// Paged: cu_seqlens_k=None, seqused_k=per-seq lengths, page_table=block table.
/// K/V are 4D: `[num_pages, page_size, num_heads_k, head_dim]`.
///
/// `q_strides` is caller-provided so Q can be a strided view (e.g. from fused
/// QKV narrow). K/V are the paged cache tensors, always allocated contiguous.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fa4_varlen_paged_fwd(
    registry: &KernelRegistry,
    func: TVMSafeCallFn,
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    v_ptr: *mut c_void,
    o_ptr: *mut c_void,
    lse_ptr: *mut c_void,
    softmax_scale: f32,
    stream: *mut c_void,
    cu_seqlens_q_ptr: *mut c_void,
    seqused_k_ptr: *mut c_void,
    page_table_ptr: *mut c_void,
    q_shape: &[i64],
    q_strides: &[i64],
    k_shape: &[i64],
    v_shape: &[i64],
    o_shape: &[i64],
    lse_shape: &[i64],
    cu_seqlens_q_shape: &[i64],
    seqused_k_shape: &[i64],
    page_table_shape: &[i64],
    device_id: i32,
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    dtype: KernelDtype,
) -> Result<(), String> {
    let device = DLDevice {
        device_type: KDLCUDA,
        device_id,
    };
    let half_dt = half_dl_dtype(dtype);
    let (f32_dtype, i32_dtype) = make_aux_dtypes();

    debug_assert_eq!(
        q_strides.last().copied(),
        Some(1),
        "FA4 requires Q stride(-1) == 1"
    );

    let dl_q = make_dltensor(q_ptr, device, half_dt, q_shape, q_strides);

    let k_strides = contiguous_strides(k_shape);
    let dl_k = make_dltensor(k_ptr, device, half_dt, k_shape, &k_strides);

    let v_strides = contiguous_strides(v_shape);
    let dl_v = make_dltensor(v_ptr, device, half_dt, v_shape, &v_strides);

    let o_strides = contiguous_strides(o_shape);
    let dl_o = make_dltensor(o_ptr, device, half_dt, o_shape, &o_strides);

    let lse_strides = contiguous_strides(lse_shape);
    let dl_lse = make_dltensor(lse_ptr, device, f32_dtype, lse_shape, &lse_strides);

    let cu_q_strides = contiguous_strides(cu_seqlens_q_shape);
    let dl_cu_q = make_dltensor(
        cu_seqlens_q_ptr,
        device,
        i32_dtype,
        cu_seqlens_q_shape,
        &cu_q_strides,
    );

    let seqused_k_strides = contiguous_strides(seqused_k_shape);
    let dl_seqused_k = make_dltensor(
        seqused_k_ptr,
        device,
        i32_dtype,
        seqused_k_shape,
        &seqused_k_strides,
    );

    let pt_strides = contiguous_strides(page_table_shape);
    let dl_page_table = make_dltensor(
        page_table_ptr,
        device,
        i32_dtype,
        page_table_shape,
        &pt_strides,
    );

    registry.set_stream(device_id, stream);

    // Same 16-slot layout as non-paged, but different slots filled:
    // cu_seqlens_k=None (paged), seqused_k=filled, page_table=filled
    let mut args: [TVMFFIAny; 16] = [
        TVMFFIAny::dltensor(&dl_q), // 0: mQ
        TVMFFIAny::dltensor(&dl_k), // 1: mK (paged 4D)
        TVMFFIAny::dltensor(&dl_v), // 2: mV (paged 4D)
        TVMFFIAny::dltensor(&dl_o), // 3: mO
        if lse_ptr.is_null() {
            // 4: mLSE
            TVMFFIAny::none()
        } else {
            TVMFFIAny::dltensor(&dl_lse)
        },
        TVMFFIAny::float32(softmax_scale),   // 5: softmax_scale
        TVMFFIAny::dltensor(&dl_cu_q),       // 6: mCuSeqlensQ (varlen Q)
        TVMFFIAny::none(),                   // 7: mCuSeqlensK (None — paged)
        TVMFFIAny::none(),                   // 8: mSeqUsedQ
        TVMFFIAny::dltensor(&dl_seqused_k),  // 9: mSeqUsedK
        TVMFFIAny::dltensor(&dl_page_table), // 10: mPageTable
        match window_size_left {
            // 11: window_size_left
            Some(v) => TVMFFIAny::int32(v),
            None => TVMFFIAny::none(),
        },
        match window_size_right {
            // 12: window_size_right
            Some(v) => TVMFFIAny::int32(v),
            None => TVMFFIAny::none(),
        },
        TVMFFIAny::none(), // 13: learnable_sink
        TVMFFIAny::none(), // 14: blocksparse_tensors
        TVMFFIAny::none(), // 15: aux_tensors
    ];

    unsafe { registry.call_kernel(func, &mut args) }
}

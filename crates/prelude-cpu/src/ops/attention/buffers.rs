//! Thread-local scratch buffers for attention kernels.

use std::cell::UnsafeCell;

// ── Thread-local reusable buffers for tiled attention ───────────────────
// Eliminates heap alloc/dealloc per head call in the tiled attention kernel.
// Each Vec grows monotonically to the max needed size but never shrinks.
//
// Uses UnsafeCell (not RefCell) to avoid closure-based `.with()` API that
// would force the entire kernel body into a closure, inhibiting LLVM
// inlining and register allocation across the call boundary.

pub(super) struct AttnBuffers {
    pub(super) k_buf: Vec<u16>,
    pub(super) v_buf: Vec<u16>,
    pub(super) k_f32: Vec<f32>,
    pub(super) v_f32: Vec<f32>,
    pub(super) s_i: Vec<f32>,
    pub(super) v_prime: Vec<f32>,
    pub(super) s_prime: Vec<f32>,
    pub(super) m_prime: Vec<f32>,
    pub(super) q_bf16_block: Vec<u16>,
    pub(super) q_f32_block: Vec<f32>,
}

impl AttnBuffers {
    fn new() -> Self {
        Self {
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            k_f32: Vec::new(),
            v_f32: Vec::new(),
            s_i: Vec::new(),
            v_prime: Vec::new(),
            s_prime: Vec::new(),
            m_prime: Vec::new(),
            q_bf16_block: Vec::new(),
            q_f32_block: Vec::new(),
        }
    }
}

thread_local! {
    static ATTN_BUFS: UnsafeCell<AttnBuffers> = UnsafeCell::new(AttnBuffers::new());
}

/// Get a mutable reference to the thread-local attention buffers.
///
/// # Safety
/// Caller must ensure no re-entrant calls (single-threaded per GemmPool thread,
/// `prefill_attention_one_head` does not recurse).
#[inline]
pub(super) unsafe fn get_attn_bufs() -> &'static mut AttnBuffers {
    ATTN_BUFS.with(|cell| unsafe { &mut *cell.get() })
}

/// Resize a Vec to at least `len`, without initializing new elements.
/// The Vec grows but never shrinks, amortizing allocation across calls.
#[inline]
pub(super) fn ensure_len_u16(buf: &mut Vec<u16>, len: usize) {
    if buf.len() < len {
        buf.resize(len, 0);
    }
}

#[inline]
pub(super) fn ensure_len_f32(buf: &mut Vec<f32>, len: usize) {
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
}

//! Block structures for GGUF quantization formats.
//!
//! These are `repr(C)` and layout-compatible with llama.cpp / GGUF on-disk format,
//! so we can zero-copy cast `&[u8]` from a GGUF file into `&[BlockQ4_0]` etc.

/// Number of elements per Q8_0 block (also applies to Q4_0).
pub const QK8_0: usize = 32;

/// Q4_0: 32 values → 18 bytes (4.5 bpw).
///
/// Each block stores one FP16 scale (`d`) and 16 bytes of nibble-packed 4-bit weights.
/// Nibble pair `qs[j]` encodes two unsigned values in `[0..15]`; the true signed value
/// is obtained by subtracting 8 → `[-8..+7]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ4_0 {
    /// FP16 scale (delta), stored as raw `u16` bits.
    pub d: u16,
    /// 32 × 4-bit values, two per byte → 16 bytes.
    pub qs: [u8; 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ4_0>() == 18);

/// Q8_0: 32 values → 34 bytes (8.5 bpw).
///
/// Each block stores one FP16 scale (`d`) and 32 signed int8 values.
/// Used as the activation quantization format paired with Q4_0 weights.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8_0 {
    /// FP16 scale (delta), stored as raw `u16` bits.
    pub d: u16,
    /// 32 × signed 8-bit quantized values.
    pub qs: [i8; QK8_0],
}

const _: () = assert!(core::mem::size_of::<BlockQ8_0>() == 34);

// ── FP16 helpers ────────────────────────────────────────────────────────

/// Convert a raw FP16 `u16` bit pattern to `f32`.
#[inline(always)]
pub fn fp16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

/// Convert `f32` to a raw FP16 `u16` bit pattern.
#[inline(always)]
pub fn f32_to_fp16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes() {
        assert_eq!(core::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(core::mem::size_of::<BlockQ8_0>(), 34);
    }

    #[test]
    fn fp16_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.00006103515625] {
            let h = f32_to_fp16(v);
            let back = fp16_to_f32(h);
            assert_eq!(v, back, "fp16 roundtrip failed for {v}");
        }
    }

    #[test]
    fn bytemuck_cast() {
        let zeros = [0u8; 18];
        let block: &BlockQ4_0 = bytemuck::from_bytes(&zeros);
        assert_eq!(block.d, 0);
        assert_eq!(block.qs, [0u8; 16]);
    }
}

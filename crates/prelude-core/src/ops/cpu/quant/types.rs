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

// ── Q4_K / Q8_K (K-quant family: 256 elements per block) ──────────────

/// Number of elements per K-quant block.
pub const QK_K: usize = 256;

/// Packed scale array size for Q4_K (8 scales + 8 mins in 6-bit packing).
pub const K_SCALE_SIZE: usize = 12;

/// Q4_K: 256 values → 144 bytes (4.5 bpw).
///
/// Higher quality than Q4_0: uses 8 sub-block scales + 8 sub-block minimums
/// (each 6-bit, packed into 12 bytes) instead of Q4_0's single scale.
/// Each 32-value chunk has its own scale and minimum offset.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ4K {
    /// FP16 super-block scale (delta).
    pub d: u16,
    /// FP16 super-block minimum.
    pub dmin: u16,
    /// Packed 6-bit scales and mins for 8 sub-blocks (12 bytes).
    pub scales: [u8; K_SCALE_SIZE],
    /// 256 × 4-bit quantized values, two per byte → 128 bytes.
    pub qs: [u8; QK_K / 2],
}

const _: () = assert!(core::mem::size_of::<BlockQ4K>() == 144);

/// Q8_K: 256 values → 292 bytes.
///
/// Activation quantization format paired with Q4_K weights.
/// Uses f32 scale (not f16) and pre-computed block sums for fast dot product.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8K {
    /// f32 scale (stored as raw bits for Pod compatibility).
    pub d: f32,
    /// 256 × signed 8-bit quantized values.
    pub qs: [i8; QK_K],
    /// Pre-computed sums of each 16-element sub-block (16 sums).
    /// Used to efficiently compute the minimum contribution in Q4_K dot product.
    pub bsums: [i16; QK_K / 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ8K>() == 292);

/// Extract the j-th (scale, min) pair from a Q4_K packed scales array.
///
/// The 12-byte `scales` array encodes 8 × 6-bit scales and 8 × 6-bit mins
/// in a bit-packed layout matching llama.cpp's `get_scale_min_k4`.
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

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
        assert_eq!(core::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(core::mem::size_of::<BlockQ8K>(), 292);
    }

    #[test]
    fn get_scale_min_k4_low_indices() {
        // j < 4: scale = lower 6 bits of q[j], min = lower 6 bits of q[j+4]
        let mut scales = [0u8; K_SCALE_SIZE];
        scales[0] = 0b00_101010; // scale=42
        scales[4] = 0b00_010101; // min=21
        let (s, m) = get_scale_min_k4(0, &scales);
        assert_eq!(s, 42);
        assert_eq!(m, 21);
    }

    #[test]
    fn get_scale_min_k4_high_indices() {
        // j >= 4: uses bits from multiple positions
        let mut scales = [0u8; K_SCALE_SIZE];
        // For j=4: d = (q[8] & 0xF) | ((q[0] >> 6) << 4)
        //          m = (q[8] >> 4) | ((q[4] >> 6) << 4)
        scales[0] = 0b11_000000; // upper 2 bits for scale
        scales[4] = 0b10_000000; // upper 2 bits for min
        scales[8] = 0b0111_0101; // low nibble=5 for scale, high nibble=7 for min
        let (s, m) = get_scale_min_k4(4, &scales);
        assert_eq!(s, 0x05 | (0x03 << 4)); // 5 | (3<<4) = 53
        assert_eq!(m, 0x07 | (0x02 << 4)); // 7 | (2<<4) = 39
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

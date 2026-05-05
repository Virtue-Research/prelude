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

/// Q4_1: 32 values → 20 bytes (5.0 bpw).
///
/// Asymmetric 4-bit: each block has scale (`d`) AND minimum (`m`).
/// Value = d * nibble + m, where nibble ∈ [0..15].
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ4_1 {
    /// FP16 scale (delta).
    pub d: u16,
    /// FP16 minimum.
    pub m: u16,
    /// 32 × 4-bit values, two per byte → 16 bytes.
    pub qs: [u8; 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ4_1>() == 20);

/// Q5_0: 32 values → 22 bytes (5.5 bpw).
///
/// Symmetric 5-bit: 4 low bits in `qs`, 1 high bit packed in `qh[4]`.
/// Value = d * (combined_5bit - 16).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ5_0 {
    /// FP16 scale (delta).
    pub d: u16,
    /// High bits: bit j of qh[j/8] is the 5th bit of element j.
    pub qh: [u8; 4],
    /// 32 × 4-bit low values, two per byte → 16 bytes.
    pub qs: [u8; 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ5_0>() == 22);

/// Q5_1: 32 values → 24 bytes (6.0 bpw).
///
/// Asymmetric 5-bit: scale + minimum, with 5th bit in `qh[4]`.
/// Value = d * combined_5bit + m.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ5_1 {
    /// FP16 scale (delta).
    pub d: u16,
    /// FP16 minimum.
    pub m: u16,
    /// High bits: bit j of qh[j/8] is the 5th bit of element j.
    pub qh: [u8; 4],
    /// 32 × 4-bit low values, two per byte → 16 bytes.
    pub qs: [u8; 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ5_1>() == 24);

/// Q8_1: 32 values → 36 bytes (9.0 bpw).
///
/// Activation format paired with Q4_1/Q5_1 weights.
/// Like Q8_0 but also stores `s = d * sum(qs)` for efficient asymmetric dot products.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8_1 {
    /// FP16 scale (delta).
    pub d: u16,
    /// FP16 precomputed `d * sum(qs[i])`.
    pub s: u16,
    /// 32 × signed 8-bit quantized values.
    pub qs: [i8; QK8_0],
}

const _: () = assert!(core::mem::size_of::<BlockQ8_1>() == 36);

// ── K-quant family: 256 elements per block ────────────────────────────

/// Number of elements per K-quant block.
pub const QK_K: usize = 256;

/// Packed scale array size for Q4_K/Q5_K (8 scales + 8 mins in 6-bit packing).
pub const K_SCALE_SIZE: usize = 12;

/// Q2_K: 256 values → 84 bytes (2.625 bpw).
///
/// 2-bit quantization with 16 sub-blocks of 16 elements each.
/// Each sub-block has a 4-bit scale and 4-bit minimum packed in `scales[16]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ2K {
    /// Packed 4-bit scales (low nibble) and mins (high nibble) for 16 sub-blocks.
    pub scales: [u8; QK_K / 16],
    /// 256 × 2-bit quantized values, four per byte → 64 bytes.
    pub qs: [u8; QK_K / 4],
    /// FP16 super-block scale (delta).
    pub d: u16,
    /// FP16 super-block minimum.
    pub dmin: u16,
}

const _: () = assert!(core::mem::size_of::<BlockQ2K>() == 84);

/// Q3_K: 256 values → 110 bytes (3.4375 bpw).
///
/// 3-bit quantization: 2 low bits in `qs`, 1 high bit in `hmask`.
/// 16 sub-blocks of 16 elements, scales are 6-bit signed (stored + 32).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ3K {
    /// High-bit mask: bit j of hmask[l] is the high bit of element l + 32*j.
    pub hmask: [u8; QK_K / 8],
    /// 256 × 2-bit low values, four per byte → 64 bytes.
    pub qs: [u8; QK_K / 4],
    /// Packed 6-bit scales for 16 sub-blocks (12 bytes, same packing as Q4_K).
    pub scales: [u8; 12],
    /// FP16 super-block scale (delta).
    pub d: u16,
}

const _: () = assert!(core::mem::size_of::<BlockQ3K>() == 110);

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

/// Q5_K: 256 values → 176 bytes (5.5 bpw).
///
/// 5-bit quantization: 4 low bits in `qs`, 1 high bit in `qh`.
/// Same scale/min packing as Q4_K (8 sub-blocks, 6-bit scales).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ5K {
    /// FP16 super-block scale (delta).
    pub d: u16,
    /// FP16 super-block minimum.
    pub dmin: u16,
    /// Packed 6-bit scales and mins for 8 sub-blocks (12 bytes).
    pub scales: [u8; K_SCALE_SIZE],
    /// High-bit mask: 1 bit per element → 32 bytes.
    pub qh: [u8; QK_K / 8],
    /// 256 × 4-bit low values, two per byte → 128 bytes.
    pub qs: [u8; QK_K / 2],
}

const _: () = assert!(core::mem::size_of::<BlockQ5K>() == 176);

/// Q6_K: 256 values → 210 bytes (6.5625 bpw).
///
/// 6-bit quantization: 4 low bits in `ql`, 2 high bits in `qh`.
/// 16 sub-blocks of 16 elements each with signed 8-bit scales.
/// No minimum offset — symmetric around zero (values − 32).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ6K {
    /// 256 × lower 4 bits, two per byte → 128 bytes.
    pub ql: [u8; QK_K / 2],
    /// 256 × upper 2 bits, four per byte → 64 bytes.
    pub qh: [u8; QK_K / 4],
    /// Signed 8-bit scales for 16 sub-blocks.
    pub scales: [i8; QK_K / 16],
    /// FP16 super-block scale (delta).
    pub d: u16,
}

const _: () = assert!(core::mem::size_of::<BlockQ6K>() == 210);

/// Q8_K: 256 values → 292 bytes.
///
/// Activation quantization format paired with K-quant weights.
/// Uses f32 scale (not f16) and pre-computed block sums for fast dot product.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8K {
    /// f32 scale (stored as raw bits for Pod compatibility).
    pub d: f32,
    /// 256 × signed 8-bit quantized values.
    pub qs: [i8; QK_K],
    /// Pre-computed sums of each 16-element sub-block (16 sums).
    /// Used to efficiently compute the minimum contribution in dot products.
    pub bsums: [i16; QK_K / 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ8K>() == 292);

// ── IQ (Importance-based Quantization) formats ─────────────────────────

/// Number of elements per IQ4_NL block.
pub const QK4_NL: usize = 32;

/// IQ4_NL non-linear lookup table: 4-bit indices → signed 8-bit values.
///
/// These 16 values are statistically optimal for normal-distributed weights,
/// replacing the uniform `[-8..+7]` mapping used by Q4_0.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// IQ4_NL: 32 values → 18 bytes (4.5 bpw, non-linear).
///
/// Same struct layout as Q4_0 (d + 16 nibble bytes), but nibble values are
/// indices into [`KVALUES_IQ4NL`] instead of uniform `[0..15]` with offset -8.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ4NL {
    /// FP16 scale (delta), stored as raw `u16` bits.
    pub d: u16,
    /// 32 × 4-bit non-linear indices, two per byte → 16 bytes.
    pub qs: [u8; QK4_NL / 2],
}

const _: () = assert!(core::mem::size_of::<BlockIQ4NL>() == 18);

/// IQ4_XS: 256 values → 136 bytes (4.25 bpw, non-linear + per-sub-block scales).
///
/// Uses the same [`KVALUES_IQ4NL`] lookup table as IQ4_NL, but adds per-sub-block
/// 6-bit scales (4 low bits in `scales_l`, 2 high bits packed in `scales_h`).
/// 8 sub-blocks of 32 elements each.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ4XS {
    /// FP16 super-block scale (delta).
    pub d: u16,
    /// Packed upper 2 bits of 8 sub-block scales (2 bits × 8 = 16 bits).
    pub scales_h: u16,
    /// Packed lower 4 bits of 8 sub-block scales (4 bits × 8 = 32 bits = 4 bytes).
    pub scales_l: [u8; QK_K / 64],
    /// 256 × 4-bit non-linear indices, two per byte → 128 bytes.
    pub qs: [u8; QK_K / 2],
}

const _: () = assert!(core::mem::size_of::<BlockIQ4XS>() == 136);

/// IQ3_S: 256 values → 110 bytes (3.4375 bpw).
///
/// 3-bit codebook indices with explicit sign bits and per-sub-block scales.
/// Uses `iq3s_grid` (512-entry) codebook.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ3S {
    pub d: u16,
    pub qs: [u8; QK_K / 4],      // 64 bytes: low 8 bits of 3-bit indices
    pub qh: [u8; QK_K / 32],     // 8 bytes: high bit of indices
    pub signs: [u8; QK_K / 8],   // 32 bytes: sign bits
    pub scales: [u8; QK_K / 64], // 4 bytes: sub-block scales
}

const _: () = assert!(core::mem::size_of::<BlockIQ3S>() == 110);

/// IQ3_XXS: 256 values → 98 bytes (3.0625 bpw).
///
/// 3-bit codebook indices using `iq3xxs_grid` (256-entry) codebook.
/// Scales and signs packed into the qs array.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ3XXS {
    pub d: u16,
    pub qs: [u8; 3 * QK_K / 8], // 96 bytes
}

const _: () = assert!(core::mem::size_of::<BlockIQ3XXS>() == 98);

/// IQ2_S: 256 values → 82 bytes (2.5625 bpw).
///
/// 2-bit codebook with `iq2s_grid` (1024-entry) + sign bits + scales.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ2S {
    pub d: u16,
    pub qs: [u8; QK_K / 4],      // 64 bytes
    pub qh: [u8; QK_K / 32],     // 8 bytes
    pub scales: [u8; QK_K / 32], // 8 bytes
}

const _: () = assert!(core::mem::size_of::<BlockIQ2S>() == 82);

/// IQ2_XS: 256 values → 74 bytes (2.3125 bpw).
///
/// 2-bit codebook with `iq2xs_grid` (512-entry) + per-group scales.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ2XS {
    pub d: u16,
    pub qs: [u16; QK_K / 8],     // 64 bytes (32 × u16)
    pub scales: [u8; QK_K / 32], // 8 bytes
}

const _: () = assert!(core::mem::size_of::<BlockIQ2XS>() == 74);

/// IQ2_XXS: 256 values → 66 bytes (2.0625 bpw).
///
/// 2-bit codebook with `iq2xxs_grid` (256-entry). Scales packed in qs.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ2XXS {
    pub d: u16,
    pub qs: [u16; QK_K / 8], // 64 bytes (32 × u16)
}

const _: () = assert!(core::mem::size_of::<BlockIQ2XXS>() == 66);

/// IQ1_S: 256 values → 50 bytes (1.5625 bpw).
///
/// 1-bit grid-based quantization using `iq1s_grid` (2048-entry) codebook.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ1S {
    pub d: u16,
    pub qs: [u8; QK_K / 8],   // 32 bytes: grid index low bits
    pub qh: [u16; QK_K / 32], // 16 bytes: grid index high bits + scale
}

const _: () = assert!(core::mem::size_of::<BlockIQ1S>() == 50);

/// IQ1_M: 256 values → 56 bytes (1.75 bpw).
///
/// Minimal 1-bit quantization. No FP16 scale field — scale is packed in scales array.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIQ1M {
    pub qs: [u8; QK_K / 8],      // 32 bytes: grid index low bits
    pub qh: [u8; QK_K / 16],     // 16 bytes: grid index high + shift
    pub scales: [u8; QK_K / 32], // 8 bytes: 3-bit block scales
}

const _: () = assert!(core::mem::size_of::<BlockIQ1M>() == 56);

// ── FP4 formats ────────────────────────────────────────────────────────

/// E2M1 lookup table: 4-bit → signed 8-bit. Shared by MXFP4 and NVFP4.
///
/// Values: `[0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]`
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// MXFP4: 32 values → 17 bytes (OCP MX spec).
///
/// E8M0 shared exponent + 32 packed 4-bit E2M1 values.
/// Scale = 2^(e - 127), applied as `scale * 0.5 * kvalues_mxfp4[nibble]`.
pub const QK_MXFP4: usize = 32;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockMXFP4 {
    /// E8M0 shared exponent (pure power-of-2 scale).
    pub e: u8,
    /// 32 × 4-bit E2M1 values, two per byte → 16 bytes.
    pub qs: [u8; QK_MXFP4 / 2],
}

const _: () = assert!(core::mem::size_of::<BlockMXFP4>() == 17);

/// NVFP4: 64 values → 36 bytes (NVIDIA Blackwell spec).
///
/// 4 × UE4M3 per-sub-block scales + 64 packed 4-bit E2M1 values.
/// Sub-block size = 16 elements.
pub const QK_NVFP4: usize = 64;
pub const QK_NVFP4_SUB: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockNVFP4 {
    /// UE4M3 scales: one per 16-element sub-block → 4 bytes.
    pub d: [u8; QK_NVFP4 / QK_NVFP4_SUB],
    /// 64 × 4-bit E2M1 values, two per byte → 32 bytes.
    pub qs: [u8; QK_NVFP4 / 2],
}

const _: () = assert!(core::mem::size_of::<BlockNVFP4>() == 36);

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
        assert_eq!(core::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(core::mem::size_of::<BlockQ5_0>(), 22);
        assert_eq!(core::mem::size_of::<BlockQ5_1>(), 24);
        assert_eq!(core::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(core::mem::size_of::<BlockQ8_1>(), 36);
        assert_eq!(core::mem::size_of::<BlockQ2K>(), 84);
        assert_eq!(core::mem::size_of::<BlockQ3K>(), 110);
        assert_eq!(core::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(core::mem::size_of::<BlockQ5K>(), 176);
        assert_eq!(core::mem::size_of::<BlockQ6K>(), 210);
        assert_eq!(core::mem::size_of::<BlockQ8K>(), 292);
        assert_eq!(core::mem::size_of::<BlockIQ4NL>(), 18);
        assert_eq!(core::mem::size_of::<BlockIQ4XS>(), 136);
        assert_eq!(core::mem::size_of::<BlockIQ3S>(), 110);
        assert_eq!(core::mem::size_of::<BlockIQ3XXS>(), 98);
        assert_eq!(core::mem::size_of::<BlockIQ2S>(), 82);
        assert_eq!(core::mem::size_of::<BlockIQ2XS>(), 74);
        assert_eq!(core::mem::size_of::<BlockIQ2XXS>(), 66);
        assert_eq!(core::mem::size_of::<BlockIQ1S>(), 50);
        assert_eq!(core::mem::size_of::<BlockIQ1M>(), 56);
        assert_eq!(core::mem::size_of::<BlockMXFP4>(), 17);
        assert_eq!(core::mem::size_of::<BlockNVFP4>(), 36);
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

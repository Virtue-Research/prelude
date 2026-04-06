//! CPU dequantization for GGML quantization formats.
//!
//! Reference implementation — correct but not SIMD-optimized.
//! Hot path uses prelude-cpu's optimized kernels or GPU dequant via prelude-cuda.

use super::GgmlDType;
use crate::tensor::{bail, Result};
use half::f16;

/// Dequantize raw bytes to f32 vector.
pub fn dequantize(data: &[u8], dtype: GgmlDType, elem_count: usize) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; elem_count];
    match dtype {
        GgmlDType::Q4_0 => dequantize_q4_0(data, &mut output),
        GgmlDType::Q4_1 => dequantize_q4_1(data, &mut output),
        GgmlDType::Q5_0 => dequantize_q5_0(data, &mut output),
        GgmlDType::Q5_1 => dequantize_q5_1(data, &mut output),
        GgmlDType::Q8_0 => dequantize_q8_0(data, &mut output),
        GgmlDType::Q2K => dequantize_q2k(data, &mut output),
        GgmlDType::Q3K => dequantize_q3k(data, &mut output),
        GgmlDType::Q4K => dequantize_q4k(data, &mut output),
        GgmlDType::Q5K => dequantize_q5k(data, &mut output),
        GgmlDType::Q6K => dequantize_q6k(data, &mut output),
        GgmlDType::Q8K => dequantize_q8k(data, &mut output),
        _ => bail!("dequantize: unsupported dtype {dtype}"),
    }?;
    Ok(output)
}

// ── Q4_0: 4-bit quantization, 32 elements per block ─────────────
// Layout: f16 scale (2B) + 16 bytes of 4-bit pairs = 18 bytes/block

fn dequantize_q4_0(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 32;
    let type_size = 18;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..];
        for j in 0..16 {
            let v0 = (qs[j] & 0x0F) as i32 - 8;
            let v1 = ((qs[j] >> 4) & 0x0F) as i32 - 8;
            output[i * block_size + j] = v0 as f32 * d;
            output[i * block_size + j + 16] = v1 as f32 * d;
        }
    }
    Ok(())
}

// ── Q4_1: 4-bit with min, 32 elements per block ─────────────────
// Layout: f16 scale (2B) + f16 min (2B) + 16 bytes = 20 bytes/block

fn dequantize_q4_1(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 32;
    let type_size = 20;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qs = &block[4..];
        for j in 0..16 {
            let v0 = (qs[j] & 0x0F) as f32;
            let v1 = ((qs[j] >> 4) & 0x0F) as f32;
            output[i * block_size + j] = v0 * d + m;
            output[i * block_size + j + 16] = v1 * d + m;
        }
    }
    Ok(())
}

// ── Q5_0: 5-bit quantization, 32 elements per block ─────────────
// Layout: f16 scale (2B) + 4 bytes high bits + 16 bytes low 4 bits = 22 bytes

fn dequantize_q5_0(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 32;
    let type_size = 22;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..];
        for j in 0..16 {
            let xh_0 = ((qh >> j) & 1) as i32;
            let xh_1 = ((qh >> (j + 16)) & 1) as i32;
            let v0 = ((qs[j] & 0x0F) as i32 | (xh_0 << 4)) - 16;
            let v1 = (((qs[j] >> 4) & 0x0F) as i32 | (xh_1 << 4)) - 16;
            output[i * block_size + j] = v0 as f32 * d;
            output[i * block_size + j + 16] = v1 as f32 * d;
        }
    }
    Ok(())
}

// ── Q5_1: 5-bit with min ────────────────────────────────────────

fn dequantize_q5_1(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 32;
    let type_size = 24;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..];
        for j in 0..16 {
            let xh_0 = ((qh >> j) & 1) as u32;
            let xh_1 = ((qh >> (j + 16)) & 1) as u32;
            let v0 = ((qs[j] & 0x0F) as u32) | (xh_0 << 4);
            let v1 = (((qs[j] >> 4) & 0x0F) as u32) | (xh_1 << 4);
            output[i * block_size + j] = v0 as f32 * d + m;
            output[i * block_size + j + 16] = v1 as f32 * d + m;
        }
    }
    Ok(())
}

// ── Q8_0: 8-bit quantization, 32 elements per block ─────────────
// Layout: f16 scale (2B) + 32 signed bytes = 34 bytes

fn dequantize_q8_0(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 32;
    let type_size = 34;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for j in 0..32 {
            output[i * block_size + j] = (block[2 + j] as i8) as f32 * d;
        }
    }
    Ok(())
}

// ── Q2K: 2-bit K-quant, 256 elements per block ──────────────────
// Layout: scales[16] + qs[64] + f16 d + f16 dmin = 84 bytes

fn dequantize_q2k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 256;
    let type_size = 84;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
        let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
        for j in 0..256 {
            let is = j / 128; // which pair of scale nibbles
            let qsi = j / 4;  // which quantized byte
            let shift = (j % 4) * 2; // bit shift within byte
            let sc_idx = (j % 128) / 16;
            let sc = scales[is * 8 + sc_idx];
            let sc_val = (sc & 0x0F) as f32;
            let m_val = ((sc >> 4) & 0x0F) as f32;
            let q = ((qs[qsi] >> shift) & 3) as f32;
            output[i * block_size + j] = d * sc_val * q - dmin * m_val;
        }
    }
    Ok(())
}

// ── Q3K: 3-bit K-quant, 256 elements per block ──────────────────

fn dequantize_q3k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 256;
    let type_size = 110;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        let d = f16::from_le_bytes([block[108], block[109]]).to_f32();

        // Decode 6-bit scales from 12 bytes
        let mut scales = [0i8; 16];
        for j in 0..8 {
            scales[j] = (scales_raw[j] & 0x0F) as i8 - 8;
            scales[j + 8] = ((scales_raw[j] >> 4) & 0x0F) as i8 - 8;
        }

        for j in 0..256 {
            let qsi = j / 4;
            let shift = (j % 4) * 2;
            let q_low = ((qs[qsi] >> shift) & 3) as i32;
            let q_high = ((hmask[j / 8] >> (j % 8)) & 1) as i32;
            let q = q_low | (q_high << 2);
            let sc = scales[j / 16] as f32;
            output[i * block_size + j] = d * sc * (q as f32 - 4.0);
        }
    }
    Ok(())
}

// ── Q4K: 4-bit K-quant, 256 elements per block ──────────────────
// Layout: f16 d + f16 dmin + scales[12] + qs[128] = 144 bytes

fn dequantize_q4k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 256;
    let type_size = 144;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales_raw = &block[4..16];
        let qs = &block[16..144];

        // Decode 6-bit scales and mins from 12 bytes
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..4 {
            sc[j] = scales_raw[j] & 0x3F;
            m[j] = scales_raw[j + 4] & 0x3F;
            sc[j + 4] = (scales_raw[j] >> 6) | ((scales_raw[j + 8] & 0x0F) << 2);
            m[j + 4] = (scales_raw[j + 4] >> 6) | (((scales_raw[j + 8] >> 4) & 0x0F) << 2);
        }

        for j in 0..256 {
            let group = j / 32;
            let within = j % 32;
            let q = if within < 16 {
                (qs[j / 2] & 0x0F) as f32
            } else {
                ((qs[j / 2] >> 4) & 0x0F) as f32
            };
            output[i * block_size + j] = d * sc[group] as f32 * q - dmin * m[group] as f32;
        }
    }
    Ok(())
}

// ── Q5K: 5-bit K-quant, 256 elements per block ──────────────────

fn dequantize_q5k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 256;
    let type_size = 176;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales_raw = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        // Decode scales (same pattern as Q4K)
        let mut sc = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..4 {
            sc[j] = scales_raw[j] & 0x3F;
            m[j] = scales_raw[j + 4] & 0x3F;
            sc[j + 4] = (scales_raw[j] >> 6) | ((scales_raw[j + 8] & 0x0F) << 2);
            m[j + 4] = (scales_raw[j + 4] >> 6) | (((scales_raw[j + 8] >> 4) & 0x0F) << 2);
        }

        for j in 0..256 {
            let group = j / 32;
            let within = j % 32;
            let q_low = if within < 16 {
                (qs[j / 2] & 0x0F) as u32
            } else {
                ((qs[j / 2] >> 4) & 0x0F) as u32
            };
            let q_high = ((qh[j / 8] >> (j % 8)) & 1) as u32;
            let q = q_low | (q_high << 4);
            output[i * block_size + j] = d * sc[group] as f32 * q as f32 - dmin * m[group] as f32;
        }
    }
    Ok(())
}

// ── Q6K: 6-bit K-quant, 256 elements per block ──────────────────
// Layout: ql[128] + qh[64] + scales[16] (signed) + f16 d = 210 bytes

fn dequantize_q6k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 256;
    let type_size = 210;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales: &[i8] = bytemuck::cast_slice(&block[192..208]);
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

        for j in 0..256 {
            let q_low = if j < 128 {
                (ql[j] & 0x0F) as i32
            } else {
                ((ql[j - 128] >> 4) & 0x0F) as i32
            };
            let q_high = ((qh[j / 4] >> ((j % 4) * 2)) & 3) as i32;
            let q = q_low | (q_high << 4);
            let sc = scales[j / 16] as f32;
            output[i * block_size + j] = d * sc * (q as f32 - 32.0);
        }
    }
    Ok(())
}

// ── Q8K: 8-bit K-quant, 256 elements per block ──────────────────
// Layout: f32 d (4B) + qs[256] (signed) + bsums[16] (i16) = 292 bytes

fn dequantize_q8k(data: &[u8], output: &mut [f32]) -> Result<()> {
    let block_size = 256;
    let type_size = 292;
    let n_blocks = output.len() / block_size;
    for i in 0..n_blocks {
        let block = &data[i * type_size..(i + 1) * type_size];
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let qs: &[i8] = bytemuck::cast_slice(&block[4..260]);
        for j in 0..256 {
            output[i * block_size + j] = d * qs[j] as f32;
        }
    }
    Ok(())
}

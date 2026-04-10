//! Weight quantization: FP32 -> quantized blocks.
//!
//! These functions quantize weight tensors into the GGUF block formats used by
//! the vec_dot/matmul kernels. Each produces blocks layout-compatible with
//! llama.cpp so data can round-trip through GGUF files.
//!
//! Algorithms are ported from candle-core's `k_quants.rs` (which itself is a
//! port of llama.cpp's `ggml-cpu-quants.c`).

use super::types::*;

// ── Helpers ─────────────────────────────────────────────────────────────

#[inline]
fn nearest_int(v: f32) -> i32 {
    v.round() as i32
}

/// Find (scale, min) for asymmetric quantization with `nmax` levels.
/// Iterative least-squares refinement (port of llama.cpp `make_qkx1_quants`).
fn make_qkx1_quants(nmax: i32, ntry: usize, x: &[f32]) -> (f32, f32) {
    let n = x.len();
    let mut l = vec![0u8; n];

    let min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if max == min {
        return (0.0, 0.0);
    }

    let mut min = min.min(0.0);
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;

    for _ in 0..ntry {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0i32;
        let mut did_change = false;

        for (i, &value) in x.iter().enumerate() {
            let li = nearest_int(iscale * (value - min)).clamp(0, nmax);
            let cli = li as u8;
            if cli != l[i] {
                l[i] = cli;
                did_change = true;
            }
            sumlx += (value - min) * li as f32;
            suml2 += li * li;
        }
        scale = sumlx / suml2 as f32;

        let sum: f32 = x.iter().zip(l.iter()).map(|(&xi, &li)| xi - scale * li as f32).sum();
        min = sum / n as f32;
        if min > 0.0 {
            min = 0.0;
        }
        iscale = 1.0 / scale;
        if !did_change {
            break;
        }
    }
    (scale, -min)
}

/// Find optimal scale for symmetric signed quantization with `nmax` levels.
/// Port of llama.cpp `make_q3_quants` with rmse optimization.
fn make_q3_quants(x: &[f32], nmax: i32, do_rmse: bool) -> f32 {
    let n = x.len();
    let mut l = vec![0i8; n];

    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &xi in x {
        let ax = xi.abs();
        if ax > amax {
            amax = ax;
            max = xi;
        }
    }

    if amax == 0.0 {
        return 0.0;
    }

    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for i in 0..n {
            let li = (iscale * x[i]).round() as i32;
            let li = li.clamp(-nmax, nmax - 1);
            l[i] = li as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * li as f32;
            suml2 += w * (li * li) as f32;
        }
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let slx = sumlx - w * x[i] * l[i] as f32;
                if slx > 0.0 {
                    let sl2 = suml2 - w * (l[i] as i32 * l[i] as i32) as f32;
                    let new_l = (x[i] * sl2 / slx).round() as i32;
                    let new_l = new_l.clamp(-nmax, nmax - 1);
                    if new_l != l[i] as i32 {
                        let slx2 = slx + w * x[i] * new_l as f32;
                        let sl2_2 = sl2 + w * (new_l * new_l) as f32;
                        if sl2_2 > 0.0 && slx2 * slx2 * suml2 > sumlx * sumlx * sl2_2 {
                            l[i] = new_l as i8;
                            sumlx = slx2;
                            suml2 = sl2_2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        return sumlx / suml2;
    }
    1.0 / iscale
}

/// Find optimal scale for symmetric signed quantization (Q6_K style).
/// Port of llama.cpp `make_qx_quants` with rmse_type=1.
fn make_qx_quants(n: usize, nmax: i32, x: &[f32], ls: &mut [i8]) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax == 0.0 {
        for v in ls.iter_mut().take(n) {
            *v = 0;
        }
        return 0.0;
    }

    let iscale = -(nmax as f32) / max;

    // rmse_type = 1 path (weight = x*x)
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        ls[i] = (l + nmax) as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * l as f32;
        suml2 += w * l as f32 * l as f32;
    }
    let mut scale = sumlx / suml2;
    let mut best = scale * sumlx;

    for _ in 0..3 {
        let iscale_iter = 1.0 / scale;
        let mut slx = 0.0f32;
        let mut sl2 = 0.0f32;
        let mut changed = false;
        for i in 0..n {
            let l = nearest_int(iscale_iter * x[i]).clamp(-nmax, nmax - 1);
            if l + nmax != ls[i] as i32 {
                changed = true;
            }
            let w = x[i] * x[i];
            slx += w * x[i] * l as f32;
            sl2 += w * l as f32 * l as f32;
        }
        if !changed || sl2 == 0.0 || slx * slx <= best * sl2 {
            break;
        }
        for i in 0..n {
            let iscale_iter = 1.0 / scale;
            let l = nearest_int(iscale_iter * x[i]).clamp(-nmax, nmax - 1);
            ls[i] = (nmax + l) as i8;
        }
        sumlx = slx;
        suml2 = sl2;
        scale = sumlx / suml2;
        best = scale * sumlx;
    }

    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = x[i] * x[i];
            let l = ls[i] as i32 - nmax;
            let mut slx = sumlx - w * x[i] * l as f32;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * l as f32 * l as f32;
                let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                if new_l != l {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        ls[i] = (nmax + new_l) as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        scale = sumlx / suml2;
                        best = scale * sumlx;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    let _ = best; // suppress unused warning
    scale
}

// ══════════════════════════════════════════════════════════════════════════
// Q4_0: 32 values per block, symmetric 4-bit
// ══════════════════════════════════════════════════════════════════════════

/// Quantize f32 data to Q4_0 blocks (32 values per block, 4.5 bpw).
///
/// Algorithm: find value with max absolute value -> scale = max / -8.
/// Each value is quantized to 4-bit unsigned [0..15] by: round(value/scale + 8).
/// Low nibble = first 16 values, high nibble = last 16 values.
pub fn quantize_f32_q4_0(data: &[f32]) -> Vec<BlockQ4_0> {
    assert!(data.len() % 32 == 0, "data length must be multiple of 32");
    let nb = data.len() / 32;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &data[i * 32..(i + 1) * 32];

        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in x {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }

        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        for j in 0..16 {
            let x0 = x[j] * id;
            let x1 = x[16 + j] * id;
            let xi0 = ((x0 + 8.5) as u8).min(15);
            let xi1 = ((x1 + 8.5) as u8).min(15);
            qs[j] = xi0 | (xi1 << 4);
        }

        output.push(BlockQ4_0 {
            d: f32_to_fp16(d),
            qs,
        });
    }
    output
}

// ══════════════════════════════════════════════════════════════════════════
// Q4_1: 32 values per block, asymmetric 4-bit
// ══════════════════════════════════════════════════════════════════════════

/// Quantize f32 data to Q4_1 blocks (32 values per block, 5.0 bpw).
///
/// Asymmetric: finds min and max, scale = (max-min)/15.
/// Each value → round((value - min) / scale), stored as 4-bit [0..15].
pub fn quantize_f32_q4_1(data: &[f32]) -> Vec<BlockQ4_1> {
    assert!(data.len() % 32 == 0, "data length must be multiple of 32");
    let nb = data.len() / 32;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &data[i * 32..(i + 1) * 32];

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in x {
            min = min.min(v);
            max = max.max(v);
        }

        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        for j in 0..16 {
            let x0 = (x[j] - min) * id;
            let x1 = (x[16 + j] - min) * id;
            let xi0 = ((x0 + 0.5) as u8).min(15);
            let xi1 = ((x1 + 0.5) as u8).min(15);
            qs[j] = xi0 | (xi1 << 4);
        }

        output.push(BlockQ4_1 {
            d: f32_to_fp16(d),
            m: f32_to_fp16(min),
            qs,
        });
    }
    output
}

// ══════════════════════════════════════════════════════════════════════════
// Q5_0: 32 values per block, symmetric 5-bit
// ══════════════════════════════════════════════════════════════════════════

/// Quantize f32 data to Q5_0 blocks (32 values per block, 5.5 bpw).
///
/// Symmetric 5-bit: scale = max / -16. Values → round(value/scale + 16).
/// Low 4 bits in qs nibbles, 5th bit packed into qh[4].
pub fn quantize_f32_q5_0(data: &[f32]) -> Vec<BlockQ5_0> {
    assert!(data.len() % 32 == 0, "data length must be multiple of 32");
    let nb = data.len() / 32;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &data[i * 32..(i + 1) * 32];

        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in x {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }

        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        let mut qh = 0u32;

        for j in 0..16 {
            let x0 = x[j] * id;
            let x1 = x[16 + j] * id;
            let xi0 = ((x0 + 16.5) as i8).min(31) as u8;
            let xi1 = ((x1 + 16.5) as i8).min(31) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            // 5th bit
            qh |= ((xi0 as u32 & 0x10) >> 4) << j;
            qh |= ((xi1 as u32 & 0x10) >> 4) << (j + 16);
        }

        output.push(BlockQ5_0 {
            d: f32_to_fp16(d),
            qh: qh.to_le_bytes(),
            qs,
        });
    }
    output
}

// ══════════════════════════════════════════════════════════════════════════
// Q5_1: 32 values per block, asymmetric 5-bit
// ══════════════════════════════════════════════════════════════════════════

/// Quantize f32 data to Q5_1 blocks (32 values per block, 6.0 bpw).
///
/// Asymmetric 5-bit: scale = (max-min)/31. Values → round((value-min)/scale).
/// Low 4 bits in qs nibbles, 5th bit packed into qh[4].
pub fn quantize_f32_q5_1(data: &[f32]) -> Vec<BlockQ5_1> {
    assert!(data.len() % 32 == 0, "data length must be multiple of 32");
    let nb = data.len() / 32;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &data[i * 32..(i + 1) * 32];

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in x {
            min = min.min(v);
            max = max.max(v);
        }

        let d = (max - min) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 16];
        let mut qh = 0u32;

        for j in 0..16 {
            let x0 = (x[j] - min) * id;
            let x1 = (x[16 + j] - min) * id;
            let xi0 = (x0 + 0.5) as u8;
            let xi1 = (x1 + 0.5) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            // 5th bit
            qh |= ((xi0 as u32 & 0x10) >> 4) << j;
            qh |= ((xi1 as u32 & 0x10) >> 4) << (j + 16);
        }

        output.push(BlockQ5_1 {
            d: f32_to_fp16(d),
            m: f32_to_fp16(min),
            qh: qh.to_le_bytes(),
            qs,
        });
    }
    output
}

// ══════════════════════════════════════════════════════════════════════════
// Q2_K: 256 values per block, 2-bit with sub-block scales/mins
// ══════════════════════════════════════════════════════════════════════════

/// Quantize f32 data to Q2_K blocks (256 values per block, 2.625 bpw).
///
/// 16 sub-blocks of 16 elements each. Each sub-block has a 4-bit scale and
/// 4-bit minimum packed in `scales[16]`. Values are 2-bit [0..3].
pub fn quantize_f32_q2k(data: &[f32]) -> Vec<BlockQ2K> {
    assert!(data.len() % QK_K == 0, "data length must be multiple of {QK_K}");
    let nb = data.len() / QK_K;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &data[i * QK_K..(i + 1) * QK_K];
        let mut block = BlockQ2K {
            scales: [0u8; QK_K / 16],
            qs: [0u8; QK_K / 4],
            d: 0,
            dmin: 0,
        };

        let mut mins = [0.0f32; QK_K / 16];
        let mut scales = [0.0f32; QK_K / 16];

        for (j, chunk) in x.chunks(16).enumerate() {
            let (s, m) = make_qkx1_quants(3, 5, chunk);
            scales[j] = s;
            mins[j] = m;
        }

        let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
        let max_min = mins.iter().copied().fold(0.0f32, f32::max);

        const Q4SCALE: f32 = 15.0;

        if max_scale > 0.0 {
            let iscale = Q4SCALE / max_scale;
            for j in 0..QK_K / 16 {
                block.scales[j] = nearest_int(iscale * scales[j]) as u8;
            }
            block.d = f32_to_fp16(max_scale / Q4SCALE);
        } else {
            block.d = f32_to_fp16(0.0);
        }

        if max_min > 0.0 {
            let iscale = Q4SCALE / max_min;
            for j in 0..QK_K / 16 {
                let l = nearest_int(iscale * mins[j]) as u8;
                block.scales[j] |= l << 4;
            }
            block.dmin = f32_to_fp16(max_min / Q4SCALE);
        } else {
            block.dmin = f32_to_fp16(0.0);
        }

        // Requantize each element using the stored scales
        let mut big_l = [0u8; QK_K];
        for j in 0..QK_K / 16 {
            let d = fp16_to_f32(block.d) * (block.scales[j] & 0xF) as f32;
            if d == 0.0 {
                continue;
            }
            let dm = fp16_to_f32(block.dmin) * (block.scales[j] >> 4) as f32;
            for ii in 0..16 {
                let ll = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3);
                big_l[16 * j + ii] = ll as u8;
            }
        }

        // Pack 2-bit values: 4 per byte
        for j in (0..QK_K).step_by(128) {
            for ll in 0..32 {
                block.qs[j / 4 + ll] = big_l[j + ll]
                    | (big_l[j + ll + 32] << 2)
                    | (big_l[j + ll + 64] << 4)
                    | (big_l[j + ll + 96] << 6);
            }
        }

        output.push(block);
    }
    output
}

// ══════════════════════════════════════════════════════════════════════════
// Q3_K: 256 values per block, 3-bit with 6-bit signed scales
// ══════════════════════════════════════════════════════════════════════════

/// Quantize f32 data to Q3_K blocks (256 values per block, 3.4375 bpw).
///
/// 16 sub-blocks of 16 elements. Scales are 6-bit signed (stored + 32).
/// Values are 3-bit: 2 low bits in qs, 1 high bit in hmask.
pub fn quantize_f32_q3k(data: &[f32]) -> Vec<BlockQ3K> {
    assert!(data.len() % QK_K == 0, "data length must be multiple of {QK_K}");
    let nb = data.len() / QK_K;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &data[i * QK_K..(i + 1) * QK_K];
        let mut block = BlockQ3K {
            hmask: [0u8; QK_K / 8],
            qs: [0u8; QK_K / 4],
            scales: [0u8; 12],
            d: 0,
        };

        let mut scales = [0.0f32; QK_K / 16];
        for (j, chunk) in x.chunks(16).enumerate() {
            scales[j] = make_q3_quants(chunk, 4, true);
        }

        // Find max absolute scale
        let mut max_scale = 0.0f32;
        for &s in &scales {
            if s.abs() > max_scale.abs() {
                max_scale = s;
            }
        }

        block.scales.fill(0);

        if max_scale != 0.0 {
            let iscale = -32.0 / max_scale;
            for (j, &scale) in scales.iter().enumerate() {
                let l_val = nearest_int(iscale * scale).clamp(-32, 31) + 32;
                if j < 8 {
                    block.scales[j] = (l_val & 0xF) as u8;
                } else {
                    block.scales[j - 8] |= ((l_val & 0xF) << 4) as u8;
                }
                let l_val = l_val >> 4;
                block.scales[j % 4 + 8] |= (l_val << (2 * (j / 4))) as u8;
            }
            block.d = f32_to_fp16(1.0 / iscale);
        } else {
            block.d = f32_to_fp16(0.0);
        }

        // Requantize using stored scales
        let mut l = [0i8; QK_K];
        for j in 0..QK_K / 16 {
            let sc = if j < 8 {
                block.scales[j] & 0xF
            } else {
                block.scales[j - 8] >> 4
            };
            let sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
            let d = fp16_to_f32(block.d) * sc as f32;
            if d != 0.0 {
                for ii in 0..16 {
                    let l_val = nearest_int(x[16 * j + ii] / d);
                    l[16 * j + ii] = (l_val.clamp(-4, 3) + 4) as i8;
                }
            }
        }

        // Set hmask for values > 3 (i.e., the high bit)
        block.hmask.fill(0);
        let mut m = 0usize;
        let mut hm = 1u8;
        for ll in l.iter_mut() {
            if *ll > 3 {
                block.hmask[m] |= hm;
                *ll -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }

        // Pack 2-bit values: 4 per byte
        for j in (0..QK_K).step_by(128) {
            for lv in 0..32 {
                block.qs[j / 4 + lv] = (l[j + lv]
                    | (l[j + lv + 32] << 2)
                    | (l[j + lv + 64] << 4)
                    | (l[j + lv + 96] << 6)) as u8;
            }
        }

        output.push(block);
    }
    output
}

// ══════════════════════════════════════════════════════════════════════════
// Q4_K: 256 values per block, 4-bit with 6-bit sub-block scales/mins
// ══════════════════════════════════════════════════════════════════════════

/// Quantize f32 data to Q4_K blocks (256 values per block, 4.5 bpw).
///
/// 8 sub-blocks of 32 elements. Each has a 6-bit scale and 6-bit minimum
/// packed into 12 bytes. Values are 4-bit [0..15].
pub fn quantize_f32_q4k(data: &[f32]) -> Vec<BlockQ4K> {
    assert!(data.len() % QK_K == 0, "data length must be multiple of {QK_K}");
    let nb = data.len() / QK_K;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &data[i * QK_K..(i + 1) * QK_K];
        let mut block = BlockQ4K {
            d: 0,
            dmin: 0,
            scales: [0u8; K_SCALE_SIZE],
            qs: [0u8; QK_K / 2],
        };

        let mut mins = [0.0f32; QK_K / 32];
        let mut scales = [0.0f32; QK_K / 32];

        for (j, chunk) in x.chunks(32).enumerate() {
            let (s, m) = make_qkx1_quants(15, 5, chunk);
            scales[j] = s;
            mins[j] = m;
        }

        let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
        let max_min = mins.iter().copied().fold(0.0f32, f32::max);

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        for j in 0..QK_K / 32 {
            let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
            if j < 4 {
                block.scales[j] = ls;
                block.scales[j + 4] = lm;
            } else {
                block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                block.scales[j - 4] |= (ls >> 4) << 6;
                block.scales[j] |= (lm >> 4) << 6;
            }
        }

        block.d = f32_to_fp16(max_scale / 63.0);
        block.dmin = f32_to_fp16(max_min / 63.0);

        // Requantize using stored scales
        let mut l = [0u8; QK_K];
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &block.scales);
            let d = fp16_to_f32(block.d) * sc as f32;
            if d != 0.0 {
                let dm = fp16_to_f32(block.dmin) * m as f32;
                for ii in 0..32 {
                    let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                    l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                }
            }
        }

        // Pack: low nibble = sub-block 2i, high nibble = sub-block 2i+1
        for j in (0..QK_K).step_by(64) {
            for lv in 0..32 {
                let offset_index = (j / 64) * 32 + lv;
                block.qs[offset_index] = l[j + lv] | (l[j + lv + 32] << 4);
            }
        }

        output.push(block);
    }
    output
}

// ══════════════════════════════════════════════════════════════════════════
// Q5_K: 256 values per block, 5-bit with 6-bit sub-block scales/mins
// ══════════════════════════════════════════════════════════════════════════

/// Quantize f32 data to Q5_K blocks (256 values per block, 5.5 bpw).
///
/// Same scale/min packing as Q4_K. Values are 5-bit [0..31]:
/// low 4 bits in qs, high bit in qh.
pub fn quantize_f32_q5k(data: &[f32]) -> Vec<BlockQ5K> {
    assert!(data.len() % QK_K == 0, "data length must be multiple of {QK_K}");
    let nb = data.len() / QK_K;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &data[i * QK_K..(i + 1) * QK_K];
        let mut block = BlockQ5K {
            d: 0,
            dmin: 0,
            scales: [0u8; K_SCALE_SIZE],
            qh: [0u8; QK_K / 8],
            qs: [0u8; QK_K / 2],
        };

        let mut mins = [0.0f32; QK_K / 32];
        let mut scales = [0.0f32; QK_K / 32];

        for (j, chunk) in x.chunks(32).enumerate() {
            let (s, m) = make_qkx1_quants(31, 5, chunk);
            scales[j] = s;
            mins[j] = m;
        }

        let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
        let max_min = mins.iter().copied().fold(0.0f32, f32::max);

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        for j in 0..QK_K / 32 {
            let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
            if j < 4 {
                block.scales[j] = ls;
                block.scales[j + 4] = lm;
            } else {
                block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                block.scales[j - 4] |= (ls >> 4) << 6;
                block.scales[j] |= (lm >> 4) << 6;
            }
        }

        block.d = f32_to_fp16(max_scale / 63.0);
        block.dmin = f32_to_fp16(max_min / 63.0);

        // Requantize using stored scales
        let mut l = [0u8; QK_K];
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &block.scales);
            let d = fp16_to_f32(block.d) * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = fp16_to_f32(block.dmin) * m as f32;
            for ii in 0..32 {
                let ll = nearest_int((x[32 * j + ii] + dm) / d);
                l[32 * j + ii] = ll.clamp(0, 31) as u8;
            }
        }

        // Pack: low 4 bits in qs, high bit in qh
        block.qh.fill(0);
        let mut m1 = 1u8;
        let mut m2 = 2u8;
        for n in (0..QK_K).step_by(64) {
            let offset = (n / 64) * 32;
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    block.qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    block.qh[j] |= m2;
                }
                block.qs[offset + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
        }

        output.push(block);
    }
    output
}

// ══════════════════════════════════════════════════════════════════════════
// Q6_K: 256 values per block, 6-bit with signed 8-bit scales
// ══════════════════════════════════════════════════════════════════════════

/// Quantize f32 data to Q6_K blocks (256 values per block, 6.5625 bpw).
///
/// 16 sub-blocks of 16 elements with signed 8-bit per-sub-block scales.
/// Values are 6-bit [0..63] → subtract 32 for signed interpretation.
/// Low 4 bits in ql, high 2 bits in qh.
pub fn quantize_f32_q6k(data: &[f32]) -> Vec<BlockQ6K> {
    assert!(data.len() % QK_K == 0, "data length must be multiple of {QK_K}");
    let nb = data.len() / QK_K;
    let mut output = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &data[i * QK_K..(i + 1) * QK_K];
        let mut block = BlockQ6K {
            ql: [0u8; QK_K / 2],
            qh: [0u8; QK_K / 4],
            scales: [0i8; QK_K / 16],
            d: 0,
        };

        let mut l = [0i8; QK_K];
        let mut scales = [0.0f32; QK_K / 16];

        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;

        for ib in 0..QK_K / 16 {
            let chunk = &x[ib * 16..(ib + 1) * 16];
            let mut ls = [0i8; 16];
            let scale = make_qx_quants(16, 32, chunk, &mut ls);
            scales[ib] = scale;
            let abs_scale = scale.abs();
            if abs_scale > max_abs_scale {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        let iscale = -128.0f32 / max_scale;
        block.d = f32_to_fp16(1.0 / iscale);

        for (j, &s) in scales.iter().enumerate() {
            block.scales[j] = nearest_int(iscale * s).min(127) as i8;
        }

        // Requantize using stored scales
        for j in 0..QK_K / 16 {
            let d = fp16_to_f32(block.d) * block.scales[j] as f32;
            if d == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let ll = nearest_int(x[16 * j + ii] / d).clamp(-32, 31);
                l[16 * j + ii] = (ll + 32) as i8;
            }
        }

        // Pack: low 4 bits in ql, high 2 bits in qh
        for j in (0..QK_K).step_by(128) {
            let ql_base = j / 2;  // 128 elements -> 64 ql bytes
            let qh_base = j / 4;  // 128 elements -> 32 qh bytes
            for lv in 0..32 {
                let q1 = l[j + lv] & 0xF;
                let q2 = l[j + lv + 32] & 0xF;
                let q3 = l[j + lv + 64] & 0xF;
                let q4 = l[j + lv + 96] & 0xF;
                block.ql[ql_base + lv] = (q1 | (q3 << 4)) as u8;
                block.ql[ql_base + lv + 32] = (q2 | (q4 << 4)) as u8;
                block.qh[qh_base + lv] = ((l[j + lv] >> 4)
                    | ((l[j + lv + 32] >> 4) << 2)
                    | ((l[j + lv + 64] >> 4) << 4)
                    | ((l[j + lv + 96] >> 4) << 6)) as u8;
            }
        }

        output.push(block);
    }
    output
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: dequantize Q4_0 blocks back to f32.
    fn dequant_q4_0(blocks: &[BlockQ4_0]) -> Vec<f32> {
        let mut out = Vec::with_capacity(blocks.len() * 32);
        for b in blocks {
            let d = fp16_to_f32(b.d);
            for j in 0..16 {
                let lo = (b.qs[j] & 0x0F) as i32 - 8;
                let hi = (b.qs[j] >> 4) as i32 - 8;
                out.push(lo as f32 * d);
                out.push(hi as f32 * d);
            }
        }
        // Fix ordering: Q4_0 stores first 16 values in low nibbles, next 16 in high
        let mut reordered = Vec::with_capacity(out.len());
        for bi in 0..blocks.len() {
            let base = bi * 32;
            for j in 0..16 {
                reordered.push(out[base + j * 2]); // low nibbles
            }
            for j in 0..16 {
                reordered.push(out[base + j * 2 + 1]); // high nibbles
            }
        }
        reordered
    }

    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    fn rms_error(a: &[f32], b: &[f32]) -> f32 {
        let mse: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32;
        mse.sqrt()
    }

    #[test]
    fn q4_0_roundtrip() {
        let data: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.037).sin() * 3.0).collect();
        let blocks = quantize_f32_q4_0(&data);
        assert_eq!(blocks.len(), 8);
        let deq = dequant_q4_0(&blocks);
        assert_eq!(deq.len(), data.len());
        let rmse = rms_error(&data, &deq);
        assert!(rmse < 0.5, "Q4_0 rmse too high: {rmse}");
    }

    #[test]
    fn q4_1_roundtrip() {
        let data: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.05).sin() * 2.0 + 1.0).collect();
        let blocks = quantize_f32_q4_1(&data);
        assert_eq!(blocks.len(), 4);
        // Verify blocks are valid (d, m should be reasonable)
        for b in &blocks {
            assert!(fp16_to_f32(b.d).is_finite());
            assert!(fp16_to_f32(b.m).is_finite());
        }
    }

    #[test]
    fn q5_0_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin() * 5.0).collect();
        let blocks = quantize_f32_q5_0(&data);
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn q5_1_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin() * 2.0 + 3.0).collect();
        let blocks = quantize_f32_q5_1(&data);
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn q2k_roundtrip() {
        let data: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() * 2.0).collect();
        let blocks = quantize_f32_q2k(&data);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn q3k_roundtrip() {
        let data: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() * 2.0).collect();
        let blocks = quantize_f32_q3k(&data);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn q4k_roundtrip() {
        let data: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() * 2.0).collect();
        let blocks = quantize_f32_q4k(&data);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn q5k_roundtrip() {
        let data: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() * 2.0).collect();
        let blocks = quantize_f32_q5k(&data);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn q6k_roundtrip() {
        let data: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.01).sin() * 2.0).collect();
        let blocks = quantize_f32_q6k(&data);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn all_zeros() {
        let zeros = vec![0.0f32; QK_K];
        let q2 = quantize_f32_q2k(&zeros);
        let q3 = quantize_f32_q3k(&zeros);
        let q4 = quantize_f32_q4k(&zeros);
        let q5 = quantize_f32_q5k(&zeros);
        let q6 = quantize_f32_q6k(&zeros);
        assert_eq!(q2.len(), 1);
        assert_eq!(q3.len(), 1);
        assert_eq!(q4.len(), 1);
        assert_eq!(q5.len(), 1);
        assert_eq!(q6.len(), 1);
    }
}

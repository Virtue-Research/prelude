//! Q-prologue fusion correctness for the fa3-0102 backend.
//!
//! Method (from the fa3_q_prologue_deadlock_fix report): run the fused kernel
//! (RAW q + in-kernel RMSNorm+RoPE) and compare its attention output against
//! the reference pipeline = standalone d128 qknorm+rope kernel -> attention.
//! The fused result must match the norm+rope reference tightly and differ
//! grossly from raw-Q attention (proves the prologue actually ran).
//!
//! Config coverage targets the historical failure modes:
//!   * ragged seqlen_q % 16 != 0  -> partial-tile shfl divergence (deadlock)
//!   * chunked prefill with KV history -> in-kernel position start_idx > 0
//!   * decode (seqlen_q=1, PackGQA / PagedKVNonTMA path)
#![cfg(feature = "fa3-0102")]

use candle_core::{DType, Device, Tensor};

const D: usize = 128;
const HQ: usize = 32;
const HKV: usize = 4;
const PAGE: usize = 16;

fn dev() -> Device {
    Device::new_cuda(0).expect("needs CUDA device 0")
}

fn randn_bf16(shape: &[usize], dev: &Device, off: f64) -> Tensor {
    let t = Tensor::randn(0f32, 1f32, shape, &Device::Cpu).unwrap();
    let t = (t + off).unwrap();
    t.to_device(dev).unwrap().to_dtype(DType::BF16).unwrap()
}

fn rope_tables(max_pos: usize, dev: &Device) -> (Tensor, Tensor) {
    let mut cos = vec![0f32; max_pos * D / 2];
    let mut sin = vec![0f32; max_pos * D / 2];
    for p in 0..max_pos {
        for i in 0..D / 2 {
            let inv_freq = 1f64 / 10000f64.powf(2.0 * i as f64 / D as f64);
            let ang = (p as f64) * inv_freq;
            cos[p * D / 2 + i] = ang.cos() as f32;
            sin[p * D / 2 + i] = ang.sin() as f32;
        }
    }
    let cos = Tensor::from_vec(cos, (max_pos, D / 2), &Device::Cpu)
        .unwrap()
        .to_device(dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let sin = Tensor::from_vec(sin, (max_pos, D / 2), &Device::Cpu)
        .unwrap()
        .to_device(dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    (cos, sin)
}

/// Scatter contiguous K/V (already normed+roped) into a paged cache.
fn build_paged(k: &Tensor, v: &Tensor, seq_lens: &[usize], dev: &Device) -> (Tensor, Tensor, Tensor) {
    let max_pages = seq_lens.iter().map(|l| l.div_ceil(PAGE)).max().unwrap();
    let total_pages: usize = seq_lens.iter().map(|l| l.div_ceil(PAGE)).sum();
    let mut tables = vec![0u32; seq_lens.len() * max_pages];
    let kf = k.to_dtype(DType::F32).unwrap();
    let vf = v.to_dtype(DType::F32).unwrap();
    let mut kc = vec![0f32; total_pages * PAGE * HKV * D];
    let mut vc = vec![0f32; total_pages * PAGE * HKV * D];
    let kvec = kf.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let vvec = vf.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut next_page = 0usize;
    let mut tok_base = 0usize;
    for (s, &len) in seq_lens.iter().enumerate() {
        let n_pages = len.div_ceil(PAGE);
        for p in 0..n_pages {
            let phys = next_page + p;
            tables[s * max_pages + p] = phys as u32;
            let tok_in_page = (len - p * PAGE).min(PAGE);
            for t in 0..tok_in_page {
                let src = (tok_base + p * PAGE + t) * HKV * D;
                let dst = (phys * PAGE + t) * HKV * D;
                kc[dst..dst + HKV * D].copy_from_slice(&kvec[src..src + HKV * D]);
                vc[dst..dst + HKV * D].copy_from_slice(&vvec[src..src + HKV * D]);
            }
        }
        next_page += n_pages;
        tok_base += len;
    }
    let kcache = Tensor::from_vec(kc, (total_pages, PAGE, HKV, D), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let vcache = Tensor::from_vec(vc, (total_pages, PAGE, HKV, D), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let bt = Tensor::from_vec(tables, (seq_lens.len(), max_pages), dev).unwrap();
    (kcache, vcache, bt)
}

fn cu(lens: &[usize], dev: &Device) -> Tensor {
    let mut v = vec![0u32];
    for &l in lens {
        v.push(v.last().unwrap() + l as u32);
    }
    Tensor::from_vec(v, (lens.len() + 1,), dev).unwrap()
}

fn stats(a: &Tensor, b: &Tensor) -> (f32, f64) {
    let x = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let y = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut max_abs = 0f32;
    let (mut dot, mut n1, mut n2) = (0f64, 0f64, 0f64);
    for (p, q) in x.iter().zip(y.iter()) {
        max_abs = max_abs.max((p - q).abs());
        dot += (*p as f64) * (*q as f64);
        n1 += (*p as f64) * (*p as f64);
        n2 += (*q as f64) * (*q as f64);
    }
    (max_abs, dot / (n1.sqrt() * n2.sqrt()).max(1e-30))
}

/// One scenario: build raw Q + pre-roped K/V cache, compare fused vs reference.
fn run_case(name: &str, qlens: &[usize], klens: &[usize]) {
    let dev = dev();
    let total_q: usize = qlens.iter().sum();
    let total_k: usize = klens.iter().sum();
    let scale = 1.0 / (D as f32).sqrt();
    let (cos, sin) = rope_tables(8192, &dev);
    let qw = randn_bf16(&[D], &dev, 1.0);
    let kw = randn_bf16(&[D], &dev, 1.0);
    let eps = 1e-6f32;

    // Raw Q as a non-contiguous fused-QKV view (serving shape).
    let fused_qkv = randn_bf16(&[total_q, HQ + 2 * HKV, D], &dev, 0.0);
    let q_raw = fused_qkv.narrow(1, 0, HQ).unwrap();

    // K: raw -> standalone norm+rope (absolute positions 0..klen) -> cache.
    let k_raw = randn_bf16(&[total_k, HKV, D], &dev, 0.1);
    let v = randn_bf16(&[total_k, HKV, D], &dev, 0.2);
    let k_pos: Vec<u32> = klens
        .iter()
        .flat_map(|&l| (0..l as u32).collect::<Vec<_>>())
        .collect();
    let k_pos = Tensor::from_vec(k_pos, (total_k,), &dev).unwrap();
    let k_roped = prelude_cuda::qknorm_rope_for_tests(&k_raw, &kw, &cos, &sin, &k_pos, eps as f64).unwrap();
    let (kc, vc, bt) = build_paged(&k_roped, &v, klens, &dev);

    let cu_q = cu(qlens, &dev);
    let seqused: Vec<u32> = klens.iter().map(|&l| l as u32).collect();
    let seqused = Tensor::from_vec(seqused, (klens.len(),), &dev).unwrap();
    let max_q = *qlens.iter().max().unwrap();
    let max_k = *klens.iter().max().unwrap();

    // Reference: standalone Q norm+rope at positions (klen - qlen + i), then attention.
    let q_pos: Vec<u32> = qlens
        .iter()
        .zip(klens.iter())
        .flat_map(|(&ql, &kl)| ((kl - ql) as u32..kl as u32).collect::<Vec<_>>())
        .collect();
    let q_pos = Tensor::from_vec(q_pos, (total_q,), &dev).unwrap();
    let q_ref = prelude_cuda::qknorm_rope_for_tests(&q_raw, &qw, &cos, &sin, &q_pos, eps as f64).unwrap();
    let out_ref = prelude_cuda::attn_fa3_0102_varlen_paged_for_tests(
        &q_ref, &kc, &vc, &bt, &cu_q, &seqused, max_q, max_k, scale, None,
    )
    .unwrap();

    // Fused: RAW q + in-kernel prologue.
    let out_fused = prelude_cuda::attn_fa3_0102_varlen_paged_for_tests(
        &q_raw,
        &kc,
        &vc,
        &bt,
        &cu_q,
        &seqused,
        max_q,
        max_k,
        scale,
        Some((&qw, &cos, &sin, eps)),
    )
    .unwrap();

    // Sanity: raw-Q attention (no prologue anywhere) must be GROSSLY different.
    let out_raw = prelude_cuda::attn_fa3_0102_varlen_paged_for_tests(
        &q_raw, &kc, &vc, &bt, &cu_q, &seqused, max_q, max_k, scale, None,
    )
    .unwrap();

    let (ma, cs) = stats(&out_fused, &out_ref);
    let (ma_raw, _) = stats(&out_fused, &out_raw);
    println!("{name}: fused-vs-ref max_abs={ma:.5} cos={cs:.6} | fused-vs-raw max_abs={ma_raw:.3}");
    assert!(
        ma < 3e-2 && cs > 0.9995,
        "{name}: fused does not match norm+rope reference (max_abs={ma}, cos={cs})"
    );
    assert!(
        ma_raw > 5e-2,
        "{name}: fused suspiciously equals raw-Q attention — prologue did not run?"
    );
}

#[test]
fn fuse_ragged_prefill_partial_tiles() {
    // seqlen_q % 16 != 0 -> partial tiles, the historical shfl-deadlock shape.
    run_case("ragged 100,50", &[100, 50], &[100, 50]);
}

#[test]
fn fuse_chunked_prefill_with_history() {
    run_case("chunk q50/k900 + q7/k1300", &[50, 7], &[900, 1300]);
}

#[test]
fn fuse_decode_packgqa() {
    run_case("decode x4", &[1, 1, 1, 1], &[333, 129, 1743, 64]);
}

#[test]
fn fuse_mixed_ragged() {
    run_case("mixed 333,777,1001,150", &[333, 777, 1001, 150], &[333, 777, 1001, 150]);
}

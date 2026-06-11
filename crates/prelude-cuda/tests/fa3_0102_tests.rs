//! Correctness tests for the candle-fa3-0102 backend (vendored vLLM 0.22 FA3
//! hopper kernel) against an f32 SDPA reference. Requires an SM90 GPU.
//!
//! Covers the four serving shapes:
//!   1. ragged multi-sequence varlen causal prefill (non-paged)
//!   2. paged prefill, TMA path (max_seqlen_q large)
//!   3. paged decode, PagedKVNonTMA path (max_seqlen_q == 1)
//!   4. paged chunked prefill with KV history (bottom-right causal alignment)
#![cfg(feature = "fa3-0102")]

use candle_core::{DType, Device, Tensor};

const HEAD_DIM: usize = 128;
const H_Q: usize = 32;
const H_KV: usize = 4; // topicguard qwen3 GQA ratio 8
const PAGE: usize = 128;

fn dev() -> Device {
    Device::new_cuda(0).expect("needs CUDA device 0")
}

fn randn_bf16(shape: &[usize], dev: &Device, seed_off: f64) -> Tensor {
    // The candle fork removed CUDA rand_normal — generate on CPU, then upload.
    let t = Tensor::randn(0f32, 1f32, shape, &Device::Cpu).unwrap();
    let t = (t + seed_off).unwrap();
    t.to_device(dev).unwrap().to_dtype(DType::BF16).unwrap()
}

/// f32 SDPA reference (computed on CPU — the candle fork's GPU matmul needs a
/// registered GEMM backend) for one sequence with bottom-right-aligned causal
/// mask. q: (sq, hq, d), k/v: (sk, hkv, d) — returns (sq, hq, d) on CPU.
fn sdpa_ref(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool, scale: f32) -> Tensor {
    let cpu = Device::Cpu;
    let (sq, hq, d) = q.dims3().unwrap();
    let (sk, hkv, _) = k.dims3().unwrap();
    let rep = hq / hkv;
    let to_cpu = |t: &Tensor| {
        t.to_device(&cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .transpose(0, 1)
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let qf = to_cpu(q); // (hq, sq, d)
    let kf = to_cpu(k); // (hkv, sk, d)
    let vf = to_cpu(v);
    // expand kv heads to hq
    let kf = kf
        .unsqueeze(1)
        .unwrap()
        .expand(&[hkv, rep, sk, d])
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape(&[hq, sk, d])
        .unwrap();
    let vf = vf
        .unsqueeze(1)
        .unwrap()
        .expand(&[hkv, rep, sk, d])
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape(&[hq, sk, d])
        .unwrap();
    let mut att = (qf
        .matmul(&kf.transpose(1, 2).unwrap().contiguous().unwrap())
        .unwrap()
        * (scale as f64))
        .unwrap(); // (hq, sq, sk)
    if causal {
        // bottom-right alignment: query i attends keys j <= sk - sq + i
        let mut mask = vec![0f32; sq * sk];
        for i in 0..sq {
            for j in 0..sk {
                if j > sk - sq + i {
                    mask[i * sk + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (1, sq, sk), att.device())
            .unwrap()
            .broadcast_as(att.shape())
            .unwrap();
        att = att.broadcast_add(&mask).unwrap();
    }
    let att = {
        let m = att.max_keepdim(2).unwrap();
        let e = att.broadcast_sub(&m).unwrap().exp().unwrap();
        let s = e.sum_keepdim(2).unwrap();
        e.broadcast_div(&s).unwrap().contiguous().unwrap()
    };
    att.matmul(&vf)
        .unwrap()
        .transpose(0, 1)
        .unwrap()
        .contiguous()
        .unwrap() // (sq, hq, d)
}

fn cmp(name: &str, got: &Tensor, want: &Tensor, max_abs_tol: f32) {
    let g = got
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let w = want
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(g.len(), w.len(), "{name}: length mismatch");
    let mut max_abs = 0f32;
    let (mut dot, mut n1, mut n2) = (0f64, 0f64, 0f64);
    for (a, b) in g.iter().zip(w.iter()) {
        max_abs = max_abs.max((a - b).abs());
        dot += (*a as f64) * (*b as f64);
        n1 += (*a as f64) * (*a as f64);
        n2 += (*b as f64) * (*b as f64);
    }
    let cos = dot / (n1.sqrt() * n2.sqrt()).max(1e-30);
    println!("{name}: max_abs={max_abs:.5} cos={cos:.6}");
    assert!(
        max_abs < max_abs_tol && cos > 0.999,
        "{name}: max_abs={max_abs} cos={cos}"
    );
}

/// Scatter contiguous per-sequence K/V (total, hkv, d) into a paged cache
/// (num_pages, PAGE, hkv, d) with sequential block tables. Returns
/// (kcache, vcache, block_tables, max_pages_per_seq).
fn build_paged(
    k: &Tensor,
    v: &Tensor,
    seq_lens: &[usize],
    dev: &Device,
) -> (Tensor, Tensor, Tensor) {
    let max_pages = seq_lens.iter().map(|l| l.div_ceil(PAGE)).max().unwrap();
    let total_pages: usize = seq_lens.iter().map(|l| l.div_ceil(PAGE)).sum();
    // assign pages round-robin-free: seq s gets consecutive pages but shuffled
    // start offsets to make sure the table is actually exercised (non-identity).
    let mut tables = vec![0u32; seq_lens.len() * max_pages];
    let kf = k.to_dtype(DType::F32).unwrap();
    let vf = v.to_dtype(DType::F32).unwrap();
    let mut kc = vec![0f32; total_pages * PAGE * H_KV * HEAD_DIM];
    let mut vc = vec![0f32; total_pages * PAGE * H_KV * HEAD_DIM];
    let kvec = kf.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let vvec = vf.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // reverse page allocation order across sequences so phys != logical
    let mut next_page = total_pages;
    let mut tok_base = 0usize;
    for (s, &len) in seq_lens.iter().enumerate() {
        let n_pages = len.div_ceil(PAGE);
        next_page -= n_pages;
        for p in 0..n_pages {
            let phys = next_page + p;
            tables[s * max_pages + p] = phys as u32;
            let tok_in_page = (len - p * PAGE).min(PAGE);
            for t in 0..tok_in_page {
                let src = (tok_base + p * PAGE + t) * H_KV * HEAD_DIM;
                let dst = (phys * PAGE + t) * H_KV * HEAD_DIM;
                kc[dst..dst + H_KV * HEAD_DIM].copy_from_slice(&kvec[src..src + H_KV * HEAD_DIM]);
                vc[dst..dst + H_KV * HEAD_DIM].copy_from_slice(&vvec[src..src + H_KV * HEAD_DIM]);
            }
        }
        tok_base += len;
    }
    let kcache = Tensor::from_vec(kc, (total_pages, PAGE, H_KV, HEAD_DIM), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let vcache = Tensor::from_vec(vc, (total_pages, PAGE, H_KV, HEAD_DIM), dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let bt = Tensor::from_vec(tables, (seq_lens.len(), max_pages), dev).unwrap();
    (kcache, vcache, bt)
}

fn cu_seqlens(lens: &[usize], dev: &Device) -> Tensor {
    let mut cu = vec![0u32];
    for &l in lens {
        cu.push(cu.last().unwrap() + l as u32);
    }
    Tensor::from_vec(cu, (lens.len() + 1,), dev).unwrap()
}

#[test]
fn ragged_varlen_causal_vs_ref() {
    let dev = dev();
    let lens = [600usize, 433, 7, 320]; // ragged, multi-page, includes tiny seq
    let total: usize = lens.iter().sum();
    let q = randn_bf16(&[total, H_Q, HEAD_DIM], &dev, 0.0);
    let k = randn_bf16(&[total, H_KV, HEAD_DIM], &dev, 0.1);
    let v = randn_bf16(&[total, H_KV, HEAD_DIM], &dev, 0.2);
    let cu = cu_seqlens(&lens, &dev);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let max = *lens.iter().max().unwrap();

    let out = prelude_cuda::attn_fa3_0102_varlen_for_tests(&q, &k, &v, &cu, &cu, max, max, scale, true)
        .unwrap();

    let mut off = 0;
    for (i, &len) in lens.iter().enumerate() {
        let qs = q.narrow(0, off, len).unwrap();
        let ks = k.narrow(0, off, len).unwrap();
        let vs = v.narrow(0, off, len).unwrap();
        let want = sdpa_ref(&qs, &ks, &vs, true, scale);
        let got = out.narrow(0, off, len).unwrap();
        cmp(&format!("ragged seq{i} len{len}"), &got, &want, 2e-2);
        off += len;
    }
}

#[test]
fn paged_prefill_tma_vs_ref() {
    let dev = dev();
    // full prefill: seqused_k == q len per seq; max_seqlen_q big => TMA paged
    let lens = [513usize, 1024, 130];
    let total: usize = lens.iter().sum();
    let q = randn_bf16(&[total, H_Q, HEAD_DIM], &dev, 0.0);
    let k = randn_bf16(&[total, H_KV, HEAD_DIM], &dev, 0.1);
    let v = randn_bf16(&[total, H_KV, HEAD_DIM], &dev, 0.2);
    let (kc, vc, bt) = build_paged(&k, &v, &lens, &dev);
    let cu_q = cu_seqlens(&lens, &dev);
    let seqused: Vec<u32> = lens.iter().map(|&l| l as u32).collect();
    let seqused = Tensor::from_vec(seqused, (lens.len(),), &dev).unwrap();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let max = *lens.iter().max().unwrap();

    let out = prelude_cuda::attn_fa3_0102_varlen_paged_for_tests(
        &q, &kc, &vc, &bt, &cu_q, &seqused, max, max, scale, None,
    )
    .unwrap();

    let mut off = 0;
    for (i, &len) in lens.iter().enumerate() {
        let qs = q.narrow(0, off, len).unwrap();
        let ks = k.narrow(0, off, len).unwrap();
        let vs = v.narrow(0, off, len).unwrap();
        let want = sdpa_ref(&qs, &ks, &vs, true, scale);
        let got = out.narrow(0, off, len).unwrap();
        cmp(&format!("paged-tma seq{i} len{len}"), &got, &want, 2e-2);
        off += len;
    }
}

#[test]
fn paged_decode_nontma_vs_ref() {
    let dev = dev();
    // decode: 1 query token per seq against long cached K => PagedKVNonTMA path
    let klens = [1743usize, 333, 2048, 129];
    let b = klens.len();
    let total_k: usize = klens.iter().sum();
    let q = randn_bf16(&[b, H_Q, HEAD_DIM], &dev, 0.0);
    let k = randn_bf16(&[total_k, H_KV, HEAD_DIM], &dev, 0.1);
    let v = randn_bf16(&[total_k, H_KV, HEAD_DIM], &dev, 0.2);
    let (kc, vc, bt) = build_paged(&k, &v, &klens, &dev);
    let qlens = vec![1usize; b];
    let cu_q = cu_seqlens(&qlens, &dev);
    let seqused: Vec<u32> = klens.iter().map(|&l| l as u32).collect();
    let seqused = Tensor::from_vec(seqused, (b,), &dev).unwrap();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let max_k = *klens.iter().max().unwrap();

    let out = prelude_cuda::attn_fa3_0102_varlen_paged_for_tests(
        &q, &kc, &vc, &bt, &cu_q, &seqused, 1, max_k, scale, None,
    )
    .unwrap();

    let mut koff = 0;
    for (i, &klen) in klens.iter().enumerate() {
        let qs = q.narrow(0, i, 1).unwrap();
        let ks = k.narrow(0, koff, klen).unwrap();
        let vs = v.narrow(0, koff, klen).unwrap();
        let want = sdpa_ref(&qs, &ks, &vs, true, scale);
        let got = out.narrow(0, i, 1).unwrap();
        cmp(&format!("paged-decode seq{i} klen{klen}"), &got, &want, 2e-2);
        koff += klen;
    }
}

#[test]
fn paged_chunked_prefill_history_vs_ref() {
    let dev = dev();
    // chunked prefill: each seq has cached history, q is the latest chunk only.
    // causal mask must be bottom-right aligned.
    let klens = [900usize, 1300];
    let qlens = [50usize, 7];
    let b = klens.len();
    let total_k: usize = klens.iter().sum();
    let total_q: usize = qlens.iter().sum();
    let k = randn_bf16(&[total_k, H_KV, HEAD_DIM], &dev, 0.1);
    let v = randn_bf16(&[total_k, H_KV, HEAD_DIM], &dev, 0.2);
    let q = randn_bf16(&[total_q, H_Q, HEAD_DIM], &dev, 0.0);
    let (kc, vc, bt) = build_paged(&k, &v, &klens, &dev);
    let cu_q = cu_seqlens(&qlens, &dev);
    let seqused: Vec<u32> = klens.iter().map(|&l| l as u32).collect();
    let seqused = Tensor::from_vec(seqused, (b,), &dev).unwrap();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let out = prelude_cuda::attn_fa3_0102_varlen_paged_for_tests(
        &q,
        &kc,
        &vc,
        &bt,
        &cu_q,
        &seqused,
        *qlens.iter().max().unwrap(),
        *klens.iter().max().unwrap(),
        scale,
        None,
    )
    .unwrap();

    let (mut qoff, mut koff) = (0, 0);
    for i in 0..b {
        let qs = q.narrow(0, qoff, qlens[i]).unwrap();
        let ks = k.narrow(0, koff, klens[i]).unwrap();
        let vs = v.narrow(0, koff, klens[i]).unwrap();
        let want = sdpa_ref(&qs, &ks, &vs, true, scale);
        let got = out.narrow(0, qoff, qlens[i]).unwrap();
        cmp(&format!("paged-chunk seq{i} q{} k{}", qlens[i], klens[i]), &got, &want, 2e-2);
        qoff += qlens[i];
        koff += klens[i];
    }
}

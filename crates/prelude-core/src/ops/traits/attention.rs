//! Default composed implementation for attention (matmul SDPA fallback).
//!
//! Used when FlashInfer/FA4 can't handle the head_dim (> 256).
//! Computes entirely in F32 for numerical stability, with per-head matmul
//! to avoid CUTLASS alignment issues. Uses naive F32 GEMM for small matrices.

use crate::tensor::{DType, Result, Tensor};
use super::ops::{MaskType, VarlenParams};

pub fn varlen_attention(q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
    let cu_q: Vec<u32> = params.cu_seqlens_q.to_vec1()?;
    let cu_k: Vec<u32> = params.cu_seqlens_k.to_vec1()?;
    let batch = cu_q.len() - 1;
    let orig_dtype = q.dtype();
    let (n_hq, _hd) = (q.shape().dims()[1], q.shape().dims()[2]);
    let n_hkv = k.shape().dims()[1];
    let gqa = n_hq / n_hkv;
    // Full F32 computation for precision matching HF's FP32-accumulated matmul.
    // The naive F32 GEMM fallback (in CUTLASS dispatch) handles small matrices
    // where CUTLASS tile-based kernels have bugs.
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let mut outs = Vec::with_capacity(batch);
    for b in 0..batch {
        let (qs, qe) = (cu_q[b] as usize, cu_q[b+1] as usize);
        let (ks, ke) = (cu_k[b] as usize, cu_k[b+1] as usize);
        let (sq, sk) = (qe - qs, ke - ks);
        let q_seq = q.narrow(0, qs, sq)?;
        let k_seq = k.narrow(0, ks, sk)?;
        let v_seq = v.narrow(0, ks, sk)?;
        let mut head_outs = Vec::with_capacity(n_hq);
        for h in 0..n_hq {
            let hkv = h / gqa;
            let qh = q_seq.narrow(1, h, 1)?.squeeze(1)?.contiguous()?;
            let kh = k_seq.narrow(1, hkv, 1)?.squeeze(1)?.contiguous()?;
            let vh = v_seq.narrow(1, hkv, 1)?.squeeze(1)?.contiguous()?;
            let scores = (qh.matmul(&kh.t()?)? * params.scale as f64)?;
            let scores = apply_mask_2d(&scores, &params.mask, sq, sk, q.device())?;
            let w = candle_nn::ops::softmax(&scores, 1)?;
            let out = w.matmul(&vh)?;
            head_outs.push(out.unsqueeze(1)?);
        }
        let head_refs: Vec<&Tensor> = head_outs.iter().collect();
        outs.push(Tensor::cat(&head_refs, 1)?);
    }
    let result = if outs.len() == 1 { outs.into_iter().next().unwrap() }
    else { let refs: Vec<&Tensor> = outs.iter().collect(); Tensor::cat(&refs, 0)? };
    result.to_dtype(orig_dtype)
}

fn apply_mask_2d(scores: &Tensor, mask: &MaskType, sq: usize, sk: usize, device: &crate::tensor::Device) -> Result<Tensor> {
    match mask {
        MaskType::Causal if sq == 1 => Ok(scores.clone()),
        MaskType::Causal => {
            let mut d = vec![f32::NEG_INFINITY; sq * sk];
            for i in 0..sq { for j in 0..sk { if j <= i + (sk - sq) { d[i*sk+j] = 0.0; } } }
            let mask = Tensor::from_vec(d, (sq, sk), device)?.to_dtype(scores.dtype())?;
            scores.broadcast_add(&mask)
        }
        MaskType::Bidirectional => Ok(scores.clone()),
        MaskType::Custom(m) => scores.broadcast_add(m),
        MaskType::SlidingWindow { left, right } => {
            let mut d = vec![f32::NEG_INFINITY; sq * sk];
            let ko = sk.saturating_sub(sq);
            for i in 0..sq { for j in 0..sk {
                let rel = (j as i64) - (i as i64 + ko as i64);
                if rel >= -(*left as i64) && rel <= (*right as i64) { d[i*sk+j] = 0.0; }
            }}
            let mask = Tensor::from_vec(d, (sq, sk), device)?.to_dtype(scores.dtype())?;
            scores.broadcast_add(&mask)
        }
    }
}

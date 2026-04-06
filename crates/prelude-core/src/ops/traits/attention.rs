//! Default composed implementation for attention (matmul SDPA fallback).

use crate::tensor::{Result, Tensor};
use super::ops::{MaskType, VarlenParams};

pub fn varlen_attention(q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> {
    let cu_q: Vec<u32> = params.cu_seqlens_q.to_vec1()?;
    let cu_k: Vec<u32> = params.cu_seqlens_k.to_vec1()?;
    let batch = cu_q.len() - 1;
    let (n_hq, hd) = (q.shape().dims()[1], q.shape().dims()[2]);
    let n_hkv = k.shape().dims()[1];
    let gqa = n_hq / n_hkv;
    let mut outs = Vec::with_capacity(batch);
    for b in 0..batch {
        let (qs, qe) = (cu_q[b] as usize, cu_q[b+1] as usize);
        let (ks, ke) = (cu_k[b] as usize, cu_k[b+1] as usize);
        let (sq, sk) = (qe - qs, ke - ks);
        let qt = q.narrow(0, qs, sq)?.transpose(0, 1)?;
        let kt = k.narrow(0, ks, sk)?.transpose(0, 1)?;
        let vt = v.narrow(0, ks, sk)?.transpose(0, 1)?;
        let (kt, vt) = if gqa > 1 {
            (kt.unsqueeze(1)?.repeat(&[1, gqa, 1, 1])?.reshape((n_hq, sk, hd))?,
             vt.unsqueeze(1)?.repeat(&[1, gqa, 1, 1])?.reshape((n_hq, sk, hd))?)
        } else { (kt.contiguous()?, vt.contiguous()?) };
        let scores = (qt.matmul(&kt.transpose(1, 2)?)? * params.scale as f64)?;
        let scores = apply_mask(&scores, &params.mask, sq, sk, q.device())?;
        let w = scores.softmax(2)?;
        outs.push(w.matmul(&vt)?.transpose(0, 1)?);
    }
    if outs.len() == 1 { Ok(outs.into_iter().next().unwrap()) }
    else { let refs: Vec<&Tensor> = outs.iter().collect(); Tensor::cat(&refs, 0) }
}

fn apply_mask(scores: &Tensor, mask: &MaskType, sq: usize, sk: usize, device: &crate::tensor::Device) -> Result<Tensor> {
    match mask {
        MaskType::Causal if sq == 1 => Ok(scores.clone()),
        MaskType::Causal => {
            let mut d = vec![f32::NEG_INFINITY; sq * sk];
            for i in 0..sq { for j in 0..sk { if j <= i + (sk - sq) { d[i*sk+j] = 0.0; } } }
            let mask = Tensor::from_vec(d, (1, sq, sk), device)?.to_dtype(scores.dtype())?;
            scores.broadcast_add(&mask)
        }
        MaskType::Bidirectional => Ok(scores.clone()),
        MaskType::Custom(m) => scores.broadcast_add(&m.unsqueeze(0)?),
        MaskType::SlidingWindow { left, right } => {
            let mut d = vec![f32::NEG_INFINITY; sq * sk];
            let ko = sk.saturating_sub(sq);
            for i in 0..sq { for j in 0..sk {
                let rel = (j as i64) - (i as i64 + ko as i64);
                if rel >= -(*left as i64) && rel <= (*right as i64) { d[i*sk+j] = 0.0; }
            }}
            let mask = Tensor::from_vec(d, (1, sq, sk), device)?.to_dtype(scores.dtype())?;
            scores.broadcast_add(&mask)
        }
    }
}

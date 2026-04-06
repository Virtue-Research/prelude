//! Default composed implementations for normalization ops.

use crate::tensor::{DType, Result, Tensor};

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let var = x32.sqr()?.mean_keepdim(x.rank() - 1)?;
    let normed = x32.broadcast_div(&(var + eps as f64)?.sqrt()?)?;
    (normed * weight.to_dtype(DType::F32)?)?.to_dtype(x.dtype())
}

pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let x32 = x.to_dtype(DType::F32)?;
    let mean = x32.mean_keepdim(x.rank() - 1)?;
    let centered = x32.broadcast_sub(&mean)?;
    let inv_std = (centered.sqr()?.mean_keepdim(x.rank() - 1)? + eps as f64)?.sqrt()?.recip()?;
    let normed = centered.broadcast_mul(&inv_std)?.to_dtype(x.dtype())?;
    let scaled = normed.broadcast_mul(weight)?;
    match bias { Some(b) => scaled.broadcast_add(b), None => Ok(scaled) }
}

pub fn group_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, num_groups: usize, eps: f32) -> Result<Tensor> {
    let s = x.shape().dims().to_vec();
    let (n, c) = (s[0], s[1]);
    let spatial: usize = s[2..].iter().product();
    let grouped = x.reshape((n, num_groups, (c / num_groups) * spatial))?;
    let g32 = grouped.to_dtype(DType::F32)?;
    let mean = g32.mean_keepdim(2)?;
    let centered = g32.broadcast_sub(&mean)?;
    let inv_std = (centered.sqr()?.mean_keepdim(2)? + eps as f64)?.sqrt()?.recip()?;
    let normed = centered.broadcast_mul(&inv_std)?.reshape(s.as_slice())?.to_dtype(x.dtype())?;
    let scaled = normed.broadcast_mul(weight)?;
    match bias { Some(b) => scaled.broadcast_add(b), None => Ok(scaled) }
}

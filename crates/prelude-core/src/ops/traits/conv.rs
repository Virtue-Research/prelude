//! Default composed implementations for convolution ops (im2col + matmul).

use crate::tensor::{Result, Tensor};

pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let (batch, in_c, in_len) = input.dims3()?;
    let (out_c, _, k_size) = weight.dims3()?;
    let out_len = (in_len + 2 * padding - k_size) / stride + 1;
    let input = if padding > 0 {
        input.pad_with_zeros(2, padding, padding)?
    } else {
        input.clone()
    };
    let mut cols = Vec::with_capacity(out_len);
    for i in 0..out_len {
        cols.push(
            input
                .narrow(2, i * stride, k_size)?
                .reshape((batch, in_c * k_size))?,
        );
    }
    let refs: Vec<&Tensor> = cols.iter().collect();
    let cols = Tensor::stack(&refs, 1)?;
    let w = weight
        .reshape((out_c, in_c * k_size))?
        .t()?
        .unsqueeze(0)?
        .broadcast_as((batch, in_c * k_size, out_c))?;
    let out = cols.matmul(&w)?.transpose(1, 2)?;
    match bias {
        Some(b) => out.broadcast_add(&b.reshape((1, out_c, 1))?),
        None => Ok(out),
    }
}

pub fn conv_transpose1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Result<Tensor> {
    let (batch, in_c, in_len) = input.dims3()?;
    let (_, _, k_size) = weight.dims3()?;
    let dilated = if stride > 1 {
        let zp = Tensor::zeros((batch, in_c, stride - 1), input.dtype(), &input.device())?;
        let mut parts: Vec<Tensor> = Vec::with_capacity(in_len * 2 - 1);
        for i in 0..in_len {
            if i > 0 {
                parts.push(zp.clone());
            }
            parts.push(input.narrow(2, i, 1)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 2)?
    } else {
        input.clone()
    };
    let mut fp = Vec::with_capacity(k_size);
    for k in (0..k_size).rev() {
        fp.push(weight.narrow(2, k, 1)?);
    }
    let fr: Vec<&Tensor> = fp.iter().collect();
    let w = Tensor::cat(&fr, 2)?.transpose(0, 1)?;
    let result = conv1d(&dilated, &w, bias, 1, k_size - 1 - padding)?;
    if output_padding > 0 {
        result.pad_with_zeros(2, 0, output_padding)
    } else {
        Ok(result)
    }
}

pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Tensor> {
    let (batch, in_c, in_h, in_w) = input.dims4()?;
    let (out_c, _, kh, kw) = weight.dims4()?;
    let (out_h, out_w) = (
        (in_h + 2 * padding[0] - kh) / stride[0] + 1,
        (in_w + 2 * padding[1] - kw) / stride[1] + 1,
    );
    let input = if padding[0] > 0 || padding[1] > 0 {
        input
            .pad_with_zeros(2, padding[0], padding[0])?
            .pad_with_zeros(3, padding[1], padding[1])?
    } else {
        input.clone()
    };
    let ps = in_c * kh * kw;
    let mut cols = Vec::with_capacity(out_h * out_w);
    for oh in 0..out_h {
        for ow in 0..out_w {
            cols.push(
                input
                    .narrow(2, oh * stride[0], kh)?
                    .narrow(3, ow * stride[1], kw)?
                    .reshape((batch, ps))?,
            );
        }
    }
    let refs: Vec<&Tensor> = cols.iter().collect();
    let cols = Tensor::stack(&refs, 1)?;
    let w = weight
        .reshape((out_c, ps))?
        .t()?
        .unsqueeze(0)?
        .broadcast_as((batch, ps, out_c))?;
    let out = cols
        .matmul(&w)?
        .reshape((batch, out_h, out_w, out_c))?
        .transpose(1, 3)?
        .transpose(2, 3)?;
    match bias {
        Some(b) => out.broadcast_add(&b.reshape((1, out_c, 1, 1))?),
        None => Ok(out),
    }
}

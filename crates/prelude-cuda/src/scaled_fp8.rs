use std::any::Any;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use prelude_core::models::commons::linear::{
    LinearBackend, ScaledFp8LinearFactory, ScaledFp8LinearFactoryEntry, ScaledFp8LinearParts,
};
use prelude_core::tensor::{DType, Device, Module, Result, Tensor, bail};

use crate::fp8_gemm;
use crate::{MOD_FP8_QUANTIZE, PTX_FP8_QUANTIZE};

#[derive(Debug)]
struct GpuScaledFp8Linear {
    weight: Tensor,
    input_scale_tensor: Tensor,
    weight_scale_tensor: Tensor,
    input_scale: f32,
    weight_scale: f32,
    bias: Option<Tensor>,
    n: usize,
    k: usize,
}

impl Clone for GpuScaledFp8Linear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            input_scale_tensor: self.input_scale_tensor.clone(),
            weight_scale_tensor: self.weight_scale_tensor.clone(),
            input_scale: self.input_scale,
            weight_scale: self.weight_scale,
            bias: self.bias.clone(),
            n: self.n,
            k: self.k,
        }
    }
}

impl Module for GpuScaledFp8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims();
        let m: usize = x_dims[..x_dims.len() - 1].iter().product();
        let x_k = *x_dims.last().unwrap();
        if x_k != self.k {
            bail!(
                "scaled FP8 linear: x inner dim {x_k} != weight dim {}",
                self.k
            );
        }

        let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(self.n);

        let x_flat = if x_dims.len() > 2 {
            x.reshape((m, self.k))?
        } else {
            x.clone()
        };
        let x_flat = match x_flat.dtype() {
            DType::BF16 | DType::F16 | DType::F32 => x_flat.contiguous()?,
            _ => x_flat.to_dtype(DType::BF16)?.contiguous()?,
        };

        let m_pad = align4(m);
        let qinput = static_scaled_quantize_padded(&x_flat, self.input_scale, m, self.k, m_pad)?;

        let mut out = fp8_gemm::fp8_gemm_nt_scalar(
            &qinput,
            &self.weight,
            &self.input_scale_tensor,
            &self.weight_scale_tensor,
            m_pad,
            self.n,
            self.k,
        )?;
        if m_pad != m {
            out = out.narrow(0, 0, m)?;
        }
        if out_shape.len() > 2 {
            out = out.reshape(out_shape.as_slice())?;
        }
        match &self.bias {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
        }
    }
}

impl LinearBackend for GpuScaledFp8Linear {
    fn name(&self) -> &str {
        "gpu/scaled-fp8"
    }

    fn is_quantized(&self) -> bool {
        true
    }

    fn scaled_fp8(&self) -> Option<ScaledFp8LinearParts<'_>> {
        Some(ScaledFp8LinearParts {
            weight: &self.weight,
            input_scale: self.input_scale,
            weight_scale: self.weight_scale,
        })
    }

    fn clone_box(&self) -> Box<dyn LinearBackend> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct GpuScaledFp8Factory;

impl ScaledFp8LinearFactory for GpuScaledFp8Factory {
    fn name(&self) -> &str {
        "gpu/scaled-fp8"
    }

    fn can_create(&self, weight: &Tensor, input_scale: &Tensor, weight_scale: &Tensor) -> bool {
        weight.device().is_cuda()
            && weight.dtype() == DType::F8E4M3
            && input_scale.shape().elem_count() == 1
            && weight_scale.shape().elem_count() == 1
    }

    fn create(
        &self,
        weight: Tensor,
        input_scale: Tensor,
        weight_scale: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Box<dyn LinearBackend>> {
        let dims = weight.dims();
        if dims.len() != 2 {
            bail!("scaled FP8 linear weight must be rank-2, got {:?}", dims);
        }
        let (n, k) = (dims[0], dims[1]);
        let input_scale = scalar_f32(&input_scale)?;
        let weight_scale = scalar_f32(&weight_scale)?;
        if input_scale <= 0.0 || weight_scale <= 0.0 {
            bail!(
                "scaled FP8 linear scales must be positive, got input_scale={input_scale}, weight_scale={weight_scale}"
            );
        }

        let weight = weight.contiguous()?;
        if k % 128 != 0 || n % 128 != 0 {
            bail!(
                "scaled FP8 FlashInfer linear requires N and K multiples of 128, got N={n}, K={k}"
            );
        }
        let input_scale_tensor = Tensor::from_vec(vec![input_scale], (1,), weight.device())?;
        let weight_scale_tensor = Tensor::from_vec(vec![weight_scale], (1,), weight.device())?;

        Ok(Box::new(GpuScaledFp8Linear {
            weight,
            input_scale_tensor,
            weight_scale_tensor,
            input_scale,
            weight_scale,
            bias,
            n,
            k,
        }))
    }
}

inventory::submit!(ScaledFp8LinearFactoryEntry::new(&GpuScaledFp8Factory));

fn align4(v: usize) -> usize {
    v.div_ceil(4) * 4
}

fn scalar_f32(t: &Tensor) -> Result<f32> {
    if t.shape().elem_count() != 1 {
        bail!(
            "scaled FP8 scale tensor must have one element, got {:?}",
            t.shape()
        );
    }
    let vals: Vec<f32> = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec1()?;
    Ok(vals[0])
}

pub(crate) fn static_scaled_quantize_padded(
    x: &Tensor,
    scale: f32,
    m: usize,
    k: usize,
    m_pad: usize,
) -> Result<Tensor> {
    if scale <= 0.0 {
        bail!("static_scaled_quantize_padded: scale must be positive, got {scale}");
    }
    if x.dims() != [m, k] {
        bail!(
            "static_scaled_quantize_padded: expected input shape [{m}, {k}], got {:?}",
            x.dims()
        );
    }

    let (x_storage, x_layout) = x.storage_and_layout();
    let x_cuda = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("static_scaled_quantize_padded: requires CUDA input"),
    };
    let dev = x_cuda.device().clone();
    let n = m_pad * k;
    let out = unsafe { dev.alloc::<float8::F8E4M3>(n) }?;

    let threads = 256u32;
    let blocks = (n as u32).div_ceil(threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let func_name = match x_cuda.dtype() {
        DType::BF16 => "static_scaled_bf16_to_fp8_e4m3_padded",
        DType::F16 => "static_scaled_f16_to_fp8_e4m3_padded",
        DType::F32 => "static_scaled_f32_to_fp8_e4m3_padded",
        other => bail!("static_scaled_quantize_padded: unsupported input dtype {other:?}"),
    };
    let func = dev.get_or_load_custom_func(func_name, MOD_FP8_QUANTIZE, PTX_FP8_QUANTIZE)?;
    let inv_scale = scale.recip();
    let m_val = m as u32;
    let k_val = k as u32;
    let m_pad_val = m_pad as u32;

    match x_cuda.dtype() {
        DType::BF16 => {
            let x_slice = x_cuda
                .as_cuda_slice::<half::bf16>()?
                .slice(x_layout.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&x_slice);
            builder.arg(&out);
            builder.arg(&inv_scale);
            builder.arg(&m_val);
            builder.arg(&k_val);
            builder.arg(&m_pad_val);
            unsafe { builder.launch(cfg) }.w()?;
        }
        DType::F16 => {
            let x_slice = x_cuda
                .as_cuda_slice::<half::f16>()?
                .slice(x_layout.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&x_slice);
            builder.arg(&out);
            builder.arg(&inv_scale);
            builder.arg(&m_val);
            builder.arg(&k_val);
            builder.arg(&m_pad_val);
            unsafe { builder.launch(cfg) }.w()?;
        }
        DType::F32 => {
            let x_slice = x_cuda
                .as_cuda_slice::<f32>()?
                .slice(x_layout.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&x_slice);
            builder.arg(&out);
            builder.arg(&inv_scale);
            builder.arg(&m_val);
            builder.arg(&k_val);
            builder.arg(&m_pad_val);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => unreachable!("dtype guarded above"),
    }

    drop(x_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        (m_pad, k),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

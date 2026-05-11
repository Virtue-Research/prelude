use std::any::Any;
use std::ffi::c_void;
use std::sync::Mutex;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use prelude_core::models::commons::linear::{
    LinearBackend, ScaledFp8LinearFactory, ScaledFp8LinearFactoryEntry,
};
use prelude_core::tensor::{bail, DType, Device, Module, Result, Tensor};

use crate::{MOD_FP8_QUANTIZE, PTX_FP8_QUANTIZE};

#[derive(Debug)]
struct GpuScaledFp8Linear {
    weight: Tensor,
    weight_scale_tma: Tensor,
    input_scale: f32,
    weight_scale: f32,
    bias: Option<Tensor>,
    n: usize,
    k: usize,
    fallback_weight: Mutex<Option<Tensor>>,
}

impl Clone for GpuScaledFp8Linear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            weight_scale_tma: self.weight_scale_tma.clone(),
            input_scale: self.input_scale,
            weight_scale: self.weight_scale,
            bias: self.bias.clone(),
            n: self.n,
            k: self.k,
            fallback_weight: Mutex::new(None),
        }
    }
}

impl GpuScaledFp8Linear {
    fn fallback_weight(&self) -> Result<Tensor> {
        let mut fallback_weight = match self.fallback_weight.lock() {
            Ok(fallback_weight) => fallback_weight,
            Err(_) => bail!("scaled FP8 linear fallback weight cache lock poisoned"),
        };
        if let Some(weight) = fallback_weight.as_ref() {
            return Ok(weight.clone());
        }

        let weight = self
            .weight
            .to_dtype(DType::BF16)?
            .affine(self.weight_scale as f64, 0.0)?;
        *fallback_weight = Some(weight.clone());
        Ok(weight)
    }

    fn fallback_forward(&self, x_flat: &Tensor, out_shape: &[usize]) -> Result<Tensor> {
        let weight = self.fallback_weight()?;
        let x_flat = match x_flat.dtype() {
            DType::BF16 => x_flat.clone(),
            _ => x_flat.to_dtype(DType::BF16)?,
        };
        let mut out = x_flat.matmul(&weight.t()?)?;
        if out_shape.len() > 2 {
            out = out.reshape(out_shape)?;
        }
        match &self.bias {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
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

        let qinput = static_scaled_quantize(&x_flat, self.input_scale)?;
        let k_groups = self.k.div_ceil(128);
        let scale_a = filled_f32_tensor(x.device(), k_groups * align4(m), self.input_scale)?;

        match fp8_gemm(
            &qinput,
            &self.weight,
            &scale_a,
            &self.weight_scale_tma,
            m,
            self.n,
            self.k,
        ) {
            Ok(mut out) => {
                if out_shape.len() > 2 {
                    out = out.reshape(out_shape.as_slice())?;
                }
                match &self.bias {
                    Some(bias) => out.broadcast_add(bias),
                    None => Ok(out),
                }
            }
            Err(err) => {
                tracing::debug!("DeepGEMM FP8 fallback to BF16 dense matmul: {err}");
                self.fallback_forward(&x_flat, out_shape.as_slice())
            }
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
        let weight_scale_tma =
            filled_f32_tensor(weight.device(), k.div_ceil(128) * align4(n), weight_scale)?;

        Ok(Box::new(GpuScaledFp8Linear {
            weight,
            weight_scale_tma,
            input_scale,
            weight_scale,
            bias,
            n,
            k,
            fallback_weight: Mutex::new(None),
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

fn filled_f32_tensor(device: &Device, len: usize, value: f32) -> Result<Tensor> {
    Tensor::zeros((len,), DType::F32, device)?.affine(0.0, value as f64)
}

fn static_scaled_quantize(x: &Tensor, scale: f32) -> Result<Tensor> {
    if scale <= 0.0 {
        bail!("static_scaled_quantize: scale must be positive, got {scale}");
    }

    let (x_storage, x_layout) = x.storage_and_layout();
    let x_cuda = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("static_scaled_quantize: requires CUDA input"),
    };
    let dev = x_cuda.device().clone();
    let n = x_layout.shape().elem_count();
    let out = unsafe { dev.alloc::<float8::F8E4M3>(n) }?;

    let threads = 256u32;
    let blocks = (n as u32).div_ceil(threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let func_name = match x_cuda.dtype() {
        DType::BF16 => "static_scaled_bf16_to_fp8_e4m3",
        DType::F16 => "static_scaled_f16_to_fp8_e4m3",
        DType::F32 => "static_scaled_f32_to_fp8_e4m3",
        other => bail!("static_scaled_quantize: unsupported input dtype {other:?}"),
    };
    let func = dev.get_or_load_custom_func(func_name, MOD_FP8_QUANTIZE, PTX_FP8_QUANTIZE)?;
    let inv_scale = scale.recip();
    let n_val = n as u32;

    match x_cuda.dtype() {
        DType::BF16 => {
            let x_slice = x_cuda
                .as_cuda_slice::<half::bf16>()?
                .slice(x_layout.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&x_slice);
            builder.arg(&out);
            builder.arg(&inv_scale);
            builder.arg(&n_val);
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
            builder.arg(&n_val);
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
            builder.arg(&n_val);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => unreachable!("dtype guarded above"),
    }

    let out_shape = x_layout.shape().clone();
    drop(x_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        out_shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

fn fp8_gemm(
    qinput: &Tensor,
    weight: &Tensor,
    scale_a: &Tensor,
    scale_b: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor> {
    let (a_storage, a_layout) = qinput.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (sa_storage, sa_layout) = scale_a.storage_and_layout();
    let (sb_storage, sb_layout) = scale_b.storage_and_layout();

    let a_cuda = match &*a_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("fp8_gemm: activations require CUDA"),
    };
    let w_cuda = match &*w_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("fp8_gemm: weight requires CUDA"),
    };
    let sa_cuda = match &*sa_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("fp8_gemm: activation scale requires CUDA"),
    };
    let sb_cuda = match &*sb_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => bail!("fp8_gemm: weight scale requires CUDA"),
    };

    if a_cuda.dtype() != DType::F8E4M3 || w_cuda.dtype() != DType::F8E4M3 {
        bail!(
            "fp8_gemm: expected F8E4M3 inputs, got {:?} and {:?}",
            a_cuda.dtype(),
            w_cuda.dtype()
        );
    }
    if sa_cuda.dtype() != DType::F32 || sb_cuda.dtype() != DType::F32 {
        bail!(
            "fp8_gemm: expected F32 scales, got {:?} and {:?}",
            sa_cuda.dtype(),
            sb_cuda.dtype()
        );
    }

    let dev = a_cuda.device().clone();
    let stream = dev.cuda_stream();
    let raw_stream = stream.cu_stream() as *mut c_void;

    let a_slice = a_cuda
        .as_cuda_slice::<float8::F8E4M3>()?
        .slice(a_layout.start_offset()..);
    let w_slice = w_cuda
        .as_cuda_slice::<float8::F8E4M3>()?
        .slice(w_layout.start_offset()..);
    let sa_slice = sa_cuda
        .as_cuda_slice::<f32>()?
        .slice(sa_layout.start_offset()..);
    let sb_slice = sb_cuda
        .as_cuda_slice::<f32>()?
        .slice(sb_layout.start_offset()..);

    let mut out = unsafe { dev.alloc::<half::bf16>(m * n) }?;

    let a_ptr = a_slice.device_ptr(&stream).0 as *mut c_void;
    let w_ptr = w_slice.device_ptr(&stream).0 as *mut c_void;
    let sa_ptr = sa_slice.device_ptr(&stream).0 as *mut c_void;
    let sb_ptr = sb_slice.device_ptr(&stream).0 as *mut c_void;
    let out_ptr = out.device_ptr_mut(&stream).0 as *mut c_void;

    unsafe {
        deepgemm::fp8_gemm(
            a_ptr, w_ptr, out_ptr, sa_ptr, sb_ptr, m as i32, n as i32, k as i32, raw_stream,
        )
    }
    .map_err(candle_core::Error::Msg)?;

    drop(a_storage);
    drop(w_storage);
    drop(sa_storage);
    drop(sb_storage);

    let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(out_storage),
        (m, n),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

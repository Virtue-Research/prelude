//! GPU quantized linear backends — registered via inventory when prelude-cuda is linked.
//!
//! Dispatches to quant-gemm (llama.cpp kernels): tiled MMQ for prefill (M>1),
//! MMVQ for decode (M=1).

use std::any::Any;
use std::sync::Arc;

use prelude_core::models::commons::linear::{LinearBackend, QuantFormat, QuantFormatEntry};
use prelude_core::tensor::quantized::{GgmlDType, QTensor};
use prelude_core::tensor::{bail, DType, Module, Result, Tensor};

use crate::device::{self, CuResultExt};

// ── GgmlDType → quant-gemm GgmlType ─────────────────────────────────

fn to_ggml_type(dtype: GgmlDType) -> Option<quant_gemm::GgmlType> {
    use quant_gemm::GgmlType;
    Some(match dtype {
        GgmlDType::Q4_0 => GgmlType::Q4_0,
        GgmlDType::Q4_1 => GgmlType::Q4_1,
        GgmlDType::Q5_0 => GgmlType::Q5_0,
        GgmlDType::Q5_1 => GgmlType::Q5_1,
        GgmlDType::Q8_0 => GgmlType::Q8_0,
        GgmlDType::Q2K  => GgmlType::Q2K,
        GgmlDType::Q3K  => GgmlType::Q3K,
        GgmlDType::Q4K  => GgmlType::Q4K,
        GgmlDType::Q5K  => GgmlType::Q5K,
        GgmlDType::Q6K  => GgmlType::Q6K,
        _ => return None,
    })
}

// ── GPU quantized weight ─────────────────────────────────────────────

/// GPU-resident quantized weight. Raw GGUF blocks live on GPU memory.
/// Forward pass: activations are quantized to Q8_1 on GPU, then tiled MMQ
/// or MMVQ is used for the matmul.
#[derive(Debug, Clone)]
struct GpuQuantLinear {
    /// Raw quantized weight bytes on GPU, as a U8 Tensor.
    gpu_weights: Tensor,
    ggml_type: quant_gemm::GgmlType,
    n: usize,
    k: usize,
}

impl Module for GpuQuantLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::BF16)?;
        let x_dims = x.shape().dims();
        let m: usize = x_dims[..x_dims.len() - 1].iter().product();
        let x_k = *x_dims.last().unwrap();
        if x_k != self.k {
            bail!("GPU quantized matmul: x inner dim {x_k} != weight dim {}", self.k);
        }

        let x_flat = if x_dims.len() > 2 {
            x.reshape((m, self.k))?
        } else {
            x.clone()
        };

        let out = crate::ops::tiled_mmq::tiled_mmq(
            &self.gpu_weights, &x_flat,
            m, self.n, self.k,
            self.ggml_type,
        )?;

        // Reshape output back to [..., N]
        if x_dims.len() > 2 {
            let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
            out_dims.push(self.n);
            out.reshape(out_dims.as_slice())
        } else {
            Ok(out)
        }
    }
}

impl LinearBackend for GpuQuantLinear {
    fn name(&self) -> &str { "gpu/quant-gemm" }
    fn is_quantized(&self) -> bool { true }
    fn clone_box(&self) -> Box<dyn LinearBackend> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

// ── QuantFormat registration ─────────────────────────────────────────

struct GpuQuantFormat;

impl QuantFormat for GpuQuantFormat {
    fn name(&self) -> &str { "gpu/quant-gemm" }

    fn can_handle(&self, dtype: GgmlDType) -> bool {
        to_ggml_type(dtype).is_some()
    }

    fn load(&self, qtensor: Arc<QTensor>) -> Result<Box<dyn LinearBackend>> {
        let ggml_type = to_ggml_type(qtensor.dtype())
            .ok_or_else(|| prelude_core::tensor::Error::Msg(
                format!("GPU quant: unsupported dtype {:?}", qtensor.dtype())
            ))?;

        let shape = qtensor.shape();
        let dims = shape.dims();
        let (n, k) = (dims[0], dims[1]);

        // Upload raw quantized bytes to GPU
        let cuda_dev = prelude_core::tensor::Device::new_cuda(0)
            .map_err(|e| prelude_core::tensor::Error::Msg(format!("CUDA device: {e}")))?;
        let dev = cuda_dev.as_cuda_device()
            .map_err(|e| prelude_core::tensor::Error::Msg(format!("as_cuda_device: {e}")))?;
        let stream = dev.cuda_stream();
        let gpu_data = unsafe { stream.clone_htod(qtensor.data()) }.ce()?;
        let gpu_weights = device::tensor_from_cuda(gpu_data, dev, (qtensor.data().len(),));

        Ok(Box::new(GpuQuantLinear {
            gpu_weights,
            ggml_type,
            n,
            k,
        }))
    }
}

inventory::submit!(QuantFormatEntry::new(&GpuQuantFormat));

//! CPU linear backends — registered via inventory when prelude-cpu is linked.
//!
//! - OnednnLinearFactory: creates OnednnLinear for CPU BF16/F32
//! - Q4_0Format, Q4KFormat: quantized GGUF backends

use std::sync::Arc;
use prelude_core::tensor::{Module, Result, Tensor};
use prelude_core::models::commons::linear::{
    CpuLinearFactory, CpuLinearFactoryEntry, LinearBackend,
};
use prelude_core::models::commons::linear::NaiveLinear;
use std::any::Any;

// ── OneDNN CPU linear factory ───────────────────────────────────────

#[cfg(feature = "onednn")]
struct OnednnLinearFactory;

#[cfg(feature = "onednn")]
impl CpuLinearFactory for OnednnLinearFactory {
    fn create(&self, linear: NaiveLinear) -> Result<Box<dyn LinearBackend>> {
        Ok(Box::new(crate::onednn::OnednnLinear::new(linear)?))
    }
}

#[cfg(feature = "onednn")]
inventory::submit!(CpuLinearFactoryEntry::new(&OnednnLinearFactory));

// ── Q4_0 quantized backend ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct QuantizedWeight {
    blocks: Vec<crate::ops::quant::BlockQ4_0>,
    n: usize,
    k: usize,
}

impl Module for QuantizedWeight {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use prelude_core::tensor::{DType, Device};

        let x = x.to_dtype(DType::F32)?;
        let x_dims = x.shape().dims();
        let m: usize = x_dims[..x_dims.len() - 1].iter().product();
        let x_k = *x_dims.last().unwrap();
        if x_k != self.k {
            prelude_core::tensor::bail!(
                "quantized matmul: x inner dim {x_k} != weight dim {}",
                self.k
            );
        }

        let x_cont = x.flatten_all()?.contiguous()?;
        let x_slice = crate::ops::tensor_as_f32_slice(&x_cont)?;

        let mut out = vec![0.0f32; m * self.n];
        crate::ops::quant::quantized_matmul_f32(x_slice, &self.blocks, &mut out, m, self.n, self.k);

        let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
        out_dims.push(self.n);
        Tensor::from_vec(out, out_dims.as_slice(), &Device::Cpu)
    }
}

impl LinearBackend for QuantizedWeight {
    fn name(&self) -> &str { "quant/q4_0" }
    fn is_quantized(&self) -> bool { true }
    fn clone_box(&self) -> Box<dyn LinearBackend> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

// ── Q4_K quantized backend ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct QuantizedWeightQ4K {
    blocks: Vec<crate::ops::quant::BlockQ4K>,
    n: usize,
    k: usize,
}

impl Module for QuantizedWeightQ4K {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use prelude_core::tensor::{DType, Device};

        let x = x.to_dtype(DType::F32)?;
        let x_dims = x.shape().dims();
        let m: usize = x_dims[..x_dims.len() - 1].iter().product();
        let x_k = *x_dims.last().unwrap();
        if x_k != self.k {
            prelude_core::tensor::bail!(
                "Q4_K matmul: x inner dim {x_k} != weight dim {}",
                self.k
            );
        }

        let x_cont = x.flatten_all()?.contiguous()?;
        let x_slice = crate::ops::tensor_as_f32_slice(&x_cont)?;

        let mut out = vec![0.0f32; m * self.n];
        crate::ops::quant::q4_k::quantized_matmul_q4k(x_slice, &self.blocks, &mut out, m, self.n, self.k);

        let mut out_dims = x_dims[..x_dims.len() - 1].to_vec();
        out_dims.push(self.n);
        Tensor::from_vec(out, out_dims.as_slice(), &Device::Cpu)
    }
}

impl LinearBackend for QuantizedWeightQ4K {
    fn name(&self) -> &str { "quant/q4_k" }
    fn is_quantized(&self) -> bool { true }
    fn clone_box(&self) -> Box<dyn LinearBackend> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

// ── QuantFormat registrations ───────────────────────────────────────

use prelude_core::models::commons::linear::{QuantFormat, QuantFormatEntry};
use prelude_core::tensor::quantized::GgmlDType;
use prelude_core::tensor::Device;

struct CpuQ4_0Format;

impl QuantFormat for CpuQ4_0Format {
    fn name(&self) -> &str { "cpu/q4_0" }

    fn can_handle(&self, dtype: GgmlDType) -> bool {
        dtype == GgmlDType::Q4_0
    }

    fn load(&self, qtensor: Arc<prelude_core::tensor::quantized::QTensor>) -> Result<Box<dyn LinearBackend>> {
        let shape = qtensor.shape();
        let dims = shape.dims();
        let (n, k) = (dims[0], dims[1]);
        let blocks: Vec<crate::ops::quant::BlockQ4_0> = bytemuck::cast_slice(qtensor.data()).to_vec();
        Ok(Box::new(QuantizedWeight { blocks, n, k }))
    }
}

inventory::submit!(QuantFormatEntry::new(&CpuQ4_0Format));

struct CpuQ4KFormat;

impl QuantFormat for CpuQ4KFormat {
    fn name(&self) -> &str { "cpu/q4_k" }

    fn can_handle(&self, dtype: GgmlDType) -> bool {
        dtype == GgmlDType::Q4K
    }

    fn load(&self, qtensor: Arc<prelude_core::tensor::quantized::QTensor>) -> Result<Box<dyn LinearBackend>> {
        let shape = qtensor.shape();
        let dims = shape.dims();
        let (n, k) = (dims[0], dims[1]);
        let blocks: Vec<crate::ops::quant::BlockQ4K> = bytemuck::cast_slice(qtensor.data()).to_vec();
        Ok(Box::new(QuantizedWeightQ4K { blocks, n, k }))
    }
}

inventory::submit!(QuantFormatEntry::new(&CpuQ4KFormat));

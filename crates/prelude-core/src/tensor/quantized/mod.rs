//! Quantized tensor types for GGUF model loading.
//!
//! Provides GgmlDType, QTensor, and dequantization support.

pub mod gguf_file;
mod k_quants;

use crate::tensor::{CpuStorage, DType, Device, Result, Shape, Tensor, Layout};
use std::sync::{Arc, RwLock};
use half::f16;

// ── GgmlDType ────────────────────────────────────────────────────

/// GGML quantization format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
}

impl GgmlDType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            _ => crate::tensor::bail!("unknown GGML dtype: {v}"),
        })
    }

    pub fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0, Self::F16 => 1,
            Self::Q4_0 => 2, Self::Q4_1 => 3,
            Self::Q5_0 => 6, Self::Q5_1 => 7,
            Self::Q8_0 => 8, Self::Q8_1 => 9,
            Self::Q2K => 10, Self::Q3K => 11, Self::Q4K => 12,
            Self::Q5K => 13, Self::Q6K => 14, Self::Q8K => 15,
            Self::BF16 => 30,
        }
    }

    /// Number of elements per quantization block.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1
            | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K
            | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Size of one quantization block in bytes.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,   // 2 + 16
            Self::Q4_1 => 20,   // 2 + 2 + 16
            Self::Q5_0 => 22,   // 2 + 4 + 16
            Self::Q5_1 => 24,   // 2 + 2 + 4 + 16
            Self::Q8_0 => 34,   // 2 + 32
            Self::Q8_1 => 36,   // 2 + 2 + 32
            Self::Q2K => 84,    // 16 + 64 + 2 + 2
            Self::Q3K => 110,   // 32 + 64 + 12 + 2
            Self::Q4K => 144,   // 2 + 2 + 12 + 128
            Self::Q5K => 176,   // 2 + 2 + 12 + 32 + 128
            Self::Q6K => 210,   // 128 + 64 + 16 + 2
            Self::Q8K => 292,   // 4 + 256 + 32
        }
    }

    /// Is this a non-quantized float format?
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}

impl std::fmt::Display for GgmlDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Self::F32 => "F32", Self::F16 => "F16", Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0", Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0", Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0", Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2K", Self::Q3K => "Q3K", Self::Q4K => "Q4K",
            Self::Q5K => "Q5K", Self::Q6K => "Q6K", Self::Q8K => "Q8K",
        })
    }
}

// ── QTensor ──────────────────────────────────────────────────────

/// Quantized tensor: raw bytes in a specific GGML quantization format.
#[derive(Debug)]
pub struct QTensor {
    data: Vec<u8>,
    dtype: GgmlDType,
    shape: Shape,
}

impl QTensor {
    /// Create a QTensor from raw bytes.
    pub fn new(data: Vec<u8>, dtype: GgmlDType, shape: Shape) -> Self {
        Self { data, dtype, shape }
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Raw quantized bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Number of elements (product of shape dimensions).
    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    /// Dequantize to f32 on CPU and wrap as a Tensor.
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let n = self.elem_count();
        let f32_data = if self.dtype.is_float() {
            self.dequantize_float(n)?
        } else {
            k_quants::dequantize(&self.data, self.dtype, n)?
        };
        let storage = CpuStorage::F32(f32_data);
        let layout = Layout::contiguous(self.shape.clone());
        let t = Tensor::from_storage_layout(
            Arc::new(RwLock::new(crate::tensor::Storage::Device(crate::tensor::DeviceStorage::from_cpu(storage)))),
            layout, DType::F32, Device::Cpu,
        );
        if device.is_cuda() {
            t.to_device(device)
        } else {
            Ok(t)
        }
    }

    fn dequantize_float(&self, n: usize) -> Result<Vec<f32>> {
        match self.dtype {
            GgmlDType::F32 => {
                let src: &[f32] = bytemuck::cast_slice(&self.data);
                Ok(src[..n].to_vec())
            }
            GgmlDType::F16 => {
                let src: &[f16] = bytemuck::cast_slice(&self.data);
                Ok(src[..n].iter().map(|x| x.to_f32()).collect())
            }
            GgmlDType::BF16 => {
                let src: &[half::bf16] = bytemuck::cast_slice(&self.data);
                Ok(src[..n].iter().map(|x| x.to_f32()).collect())
            }
            _ => unreachable!(),
        }
    }
}

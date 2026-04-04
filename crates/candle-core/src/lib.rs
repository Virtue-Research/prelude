// Trimmed candle-core for prelude: inference-only, no Metal/Accelerate/MKL backends,
// no autograd, no npy/pickle formats.

pub mod backend;
pub mod conv;
mod convert;
pub mod cpu;
pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
mod custom_op;
mod device;
pub mod display;
mod dtype;
pub mod dummy_cuda_backend;
pub mod dummy_dtype;
mod dummy_metal_backend;
pub mod error;
mod indexer;
pub mod layout;
pub mod op;
pub mod quantized;
pub mod safetensors;
pub mod scalar;
pub mod shape;
mod sort;
mod storage;
mod strided_index;
mod tensor;
mod tensor_cat;
pub mod utils;

#[cfg(feature = "cudnn")]
pub use cuda_backend::cudnn;

pub use cpu_backend::{CpuStorage, CpuStorageRef};
pub use custom_op::{CustomOp1, CustomOp2, CustomOp3, InplaceOp1, InplaceOp2, InplaceOp3};
pub use device::{Device, DeviceLocation, NdArray};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use error::{Context, Error, Result};
pub use indexer::{IndexOp, TensorIndexer};
pub use layout::Layout;
pub use shape::{Shape, D};
pub use storage::Storage;
pub use strided_index::{StridedBlocks, StridedIndex};
pub use tensor::{Tensor, TensorId};

#[cfg(feature = "cuda")]
pub use cuda_backend as cuda;

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend as cuda;

pub use cuda::{CudaDevice, CudaStorage};

pub use dummy_metal_backend::{MetalDevice, MetalError, MetalStorage};

pub trait ToUsize2 {
    fn to_usize2(self) -> (usize, usize);
}

impl ToUsize2 for usize {
    fn to_usize2(self) -> (usize, usize) {
        (self, self)
    }
}

impl ToUsize2 for (usize, usize) {
    fn to_usize2(self) -> (usize, usize) {
        self
    }
}

/// Defining a module with forward method using a single argument.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}

impl<M: Module> Module for Option<&M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

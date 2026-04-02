use crate::tensor::{Result, Tensor};

pub trait ActivationOps: Send + Sync {
    fn silu(&self, x: &Tensor) -> Result<Tensor>;
    fn gelu(&self, x: &Tensor) -> Result<Tensor>;
    fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor>;
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor>;
}

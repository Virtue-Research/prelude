use crate::tensor::{Result, Tensor};

pub trait ActivationOps: Send + Sync {
    fn silu(&self, x: &Tensor) -> Result<Tensor>;
    fn gelu(&self, x: &Tensor) -> Result<Tensor>;
    fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor>;
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor>;

    fn sigmoid(&self, x: &Tensor) -> Result<Tensor> {
        // 1 / (1 + exp(-x))
        (x.neg()?.exp()? + 1.0)?.recip()
    }

    fn log_softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        let max = x.max_keepdim(dim)?;
        let diff = x.broadcast_sub(&max)?;
        let sum_exp = diff.exp()?.sum_keepdim(dim)?;
        diff.broadcast_sub(&sum_exp.log()?)
    }
}

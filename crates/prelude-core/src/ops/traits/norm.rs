use candle_core::{Result, Tensor};

pub trait NormOps: Send + Sync {
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor>;
    fn group_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        num_groups: usize,
        eps: f32,
    ) -> Result<Tensor>;
}

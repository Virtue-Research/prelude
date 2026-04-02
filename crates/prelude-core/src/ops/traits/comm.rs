use crate::tensor::{Result, Tensor};

pub trait CommOps: Send + Sync {
    fn world_size(&self) -> usize;
    fn rank(&self) -> usize;
    fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor>;
    fn all_gather(&self, x: &Tensor, dim: usize) -> Result<Tensor>;
    fn reduce_scatter(&self, x: &Tensor, dim: usize) -> Result<Tensor>;

    fn all_to_all(
        &self,
        x: &Tensor,
        input_splits: &[usize],
        output_splits: &[usize],
    ) -> Result<Tensor>;

    fn send(&self, _x: &Tensor, _dst: usize) -> Result<()> {
        crate::tensor::bail!("point-to-point send not supported on this device")
    }

    fn recv(&self, _src: usize) -> Result<Tensor> {
        crate::tensor::bail!("point-to-point recv not supported on this device")
    }

    fn ep_dispatch_fused(
        &self,
        _x: &Tensor,
        _topk_ids: &Tensor,
        _num_experts: usize,
        _use_fp8: bool,
    ) -> Option<Result<(Tensor, Tensor)>> {
        None
    }

    fn ep_combine_fused(
        &self,
        _x: &Tensor,
        _topk_weights: &Tensor,
        _topk_ids: &Tensor,
    ) -> Option<Result<Tensor>> {
        None
    }
}

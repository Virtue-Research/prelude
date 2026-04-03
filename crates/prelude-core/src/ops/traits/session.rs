use crate::tensor::{Result, Tensor};

pub trait OpsSession: Send + Sync {
    /// Initialize per-forward-pass state. Called before model.forward().
    fn begin_forward(&self) {}

    /// Clear per-forward-pass state. Called after model.forward().
    fn end_forward(&self) {}

    /// Pre-compute paged attention scheduling for the current batch.
    fn precompute_paged_plan(
        &self,
        _block_tables: &Tensor,
        _cu_seqlens_k: &Tensor,
        _block_size: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Query free GPU memory in bytes. Returns `None` if not on GPU.
    fn gpu_free_memory(&self) -> Option<usize> { None }
}

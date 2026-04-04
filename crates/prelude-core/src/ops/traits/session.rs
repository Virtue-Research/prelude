use crate::tensor::{Result, Tensor};

pub trait OpsSession: Send + Sync {
    /// Initialize per-forward-pass state. Called before model.forward().
    fn begin_forward(&self) {}

    /// Clear per-forward-pass state. Called after model.forward().
    fn end_forward(&self) {}

    /// Pre-compute paged attention scheduling for the current batch.
    ///
    /// FlashInfer uses this to build ragged indptr/indices metadata on GPU.
    /// FA4 and other backends may use it for TMA descriptor precomputation.
    fn precompute_paged_plan(
        &self,
        _q_shape: (usize, usize, usize), // (batch_size, num_qo_heads, head_dim)
        _key_cache: &Tensor,
        _cu_seqlens_q: &Tensor,
        _block_tables: &Tensor,
        _cu_seqlens_k: &Tensor,
        _softmax_scale: f32,
    ) -> Result<()> {
        Ok(())
    }

    /// Query free GPU memory in bytes. Returns `None` if not on GPU.
    fn gpu_free_memory(&self) -> Option<usize> { None }
}

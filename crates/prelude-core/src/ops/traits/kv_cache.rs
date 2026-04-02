use candle_core::{DType, Result, Tensor};

/// Cache slot layout descriptor.
pub struct CacheSlotSpec {
    pub slot_size: usize,
    pub dtype: DType,
}

pub trait KvCacheOps: Send + Sync {
    /// Query per-head cache slot layout for KV cache allocation.
    fn cache_slot_spec(&self, head_dim: usize, dtype: DType) -> CacheSlotSpec {
        CacheSlotSpec {
            slot_size: head_dim,
            dtype,
        }
    }

    /// Write K/V to paged cache at given slot positions.
    fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()>;
}

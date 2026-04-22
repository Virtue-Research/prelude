//! Cascade attention: shared-prefix optimization via hierarchical KV cache.
//!
//! Implements the same multi-level cascade workflow as FlashInfer's
//! `MultiLevelCascadeAttentionWrapper`:
//!
//! 1. **Plan** each level independently — only the **last level** gets causal masking.
//! 2. **Run** the last level first (with causal), then run earlier levels and
//!    merge results into the accumulated output via `merge_state_in_place`.
//!
//! This avoids recomputing attention over the shared prefix for every request.
//!
//! ## Upstream reference
//!
//! `flashinfer/cascade.py` — `MultiLevelCascadeAttentionWrapper.run()`:
//! ```python
//! out, lse = self._batch_prefill_wrappers[-1].run(q, kv, return_lse=True)
//! for wrapper in self._batch_prefill_wrappers[:-1]:
//!     out_i, lse_i = wrapper.run(q, kv, return_lse=True)
//!     merge_state_in_place(out, lse, out_i, lse_i)
//! ```

use crate::loader::{KernelRegistry, PrefillKey, PrefillVariant, TVMSafeCallFn};
use crate::types::TVMFFIAny;

/// A single cascade level backed by a prefill variant.
pub struct CascadeLevel {
    /// The prefill variant for this level.
    pub prefill: PrefillVariant,
    /// Whether this is the last level (gets causal masking).
    pub is_last: bool,
}

/// Multi-level cascade attention orchestrator.
///
/// Mirrors FlashInfer's `MultiLevelCascadeAttentionWrapper`:
/// - Each level has its own prefill variant and plan state.
/// - The last level applies causal masking; earlier levels do not.
/// - After running all levels, results are merged via `merge_state_in_place`.
pub struct CascadeAttention {
    levels: Vec<CascadeLevel>,
    merge_state_in_place: TVMSafeCallFn,
    merge_state: TVMSafeCallFn,
    merge_states: TVMSafeCallFn,
}

impl CascadeAttention {
    /// Create a cascade attention with N levels.
    ///
    /// All levels share the same dtype/head_dim configuration.
    /// The last level in `prefill_keys` will receive causal masking.
    ///
    /// # Arguments
    /// * `registry` — kernel registry
    /// * `prefill_keys` — one PrefillKey per level (order: prefix levels first, unique level last)
    pub fn new(registry: &KernelRegistry, prefill_keys: &[PrefillKey]) -> Option<Self> {
        let n = prefill_keys.len();
        if n == 0 {
            return None;
        }

        let mut levels = Vec::with_capacity(n);
        for (i, key) in prefill_keys.iter().enumerate() {
            let prefill = registry.get_prefill(key)?;
            levels.push(CascadeLevel {
                prefill,
                is_last: i == n - 1,
            });
        }

        Some(Self {
            levels,
            merge_state_in_place: registry.get_utility("merge_state_in_place")?,
            merge_state: registry.get_utility("merge_state")?,
            merge_states: registry.get_utility("merge_states")?,
        })
    }

    /// Number of cascade levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get the prefill variant for a specific level.
    pub fn level(&self, idx: usize) -> &CascadeLevel {
        &self.levels[idx]
    }

    /// Merge two attention states: `(v_a, s_a)` + `(v_b, s_b)` → `(v_out, s_out)`.
    ///
    /// Args: `[v_a, s_a, v_b, s_b, v_out, s_out]` as DLTensor TVMFFIAny.
    ///
    /// # Safety
    /// All args must be valid device tensors.
    pub unsafe fn merge(
        &self,
        registry: &KernelRegistry,
        args: &[TVMFFIAny],
    ) -> Result<(), String> {
        unsafe { registry.call(self.merge_state, args)? };
        Ok(())
    }

    /// In-place merge: update `(v, s)` by incorporating `(v_other, s_other)`.
    ///
    /// This is the core operation used in the cascade run loop.
    /// Upstream signature: `merge_state_in_place(v, s, v_other, s_other)`.
    ///
    /// # Safety
    /// All args must be valid device tensors.
    pub unsafe fn merge_in_place(
        &self,
        registry: &KernelRegistry,
        args: &[TVMFFIAny],
    ) -> Result<(), String> {
        unsafe { registry.call(self.merge_state_in_place, args)? };
        Ok(())
    }

    /// Merge multiple attention states at once.
    ///
    /// # Safety
    /// All args must be valid device tensors.
    pub unsafe fn merge_multiple(
        &self,
        registry: &KernelRegistry,
        args: &[TVMFFIAny],
    ) -> Result<(), String> {
        unsafe { registry.call(self.merge_states, args)? };
        Ok(())
    }
}

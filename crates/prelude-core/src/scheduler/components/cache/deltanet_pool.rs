// DeltaNet state pool design inspired by SGLang's MambaPool:
// https://github.com/sgl-project/sglang/blob/78ddf05a/python/sglang/srt/mem_cache/memory_pool.py
// SGLang is licensed under the Apache License, Version 2.0.
//
// This module provides a pre-allocated pool of DeltaNet recurrent and convolutional
// state tensors, enabling multi-request concurrent decode for hybrid DeltaNet models
// (Qwen3.5, Qwen3-Next). Each request is assigned a pool slot during prefill and
// reads/writes state from that slot during decode, removing the batch_size=1 limitation.

use crate::tensor::{DType, Device, Result, Tensor};
use std::collections::VecDeque;

/// Configuration for DeltaNet state dimensions, extracted from model config.
pub struct DeltaNetPoolConfig {
    /// Number of DeltaNet (linear attention) layers in the model.
    pub num_deltanet_layers: usize,
    /// Number of value heads in DeltaNet layers.
    pub num_v_heads: usize,
    /// Key head dimension.
    pub head_k_dim: usize,
    /// Value head dimension.
    pub head_v_dim: usize,
    /// Conv1d input dimension (key_dim + key_dim + value_dim).
    pub conv_dim: usize,
    /// Conv1d kernel size.
    pub conv_kernel: usize,
}

/// Pre-allocated pool for DeltaNet recurrent and convolutional state.
///
/// Unlike the paged KV cache (which grows with sequence length), DeltaNet state
/// is fixed-size per layer per request. A simple free-list slot allocator suffices.
pub struct DeltaNetPool {
    /// Per DeltaNet layer: `[max_slots, num_v_heads, head_v_dim, head_k_dim]`
    /// in F32. Note the (V, K) order — this matches cuLA's `kda_decode` state
    /// contract (`h0_source.shape == (pool_size, HV, V, K)`) so the fused
    /// batched decode kernel can read/write pool slots directly via slot
    /// indices without a transpose copy.
    pub recurrent_states: Vec<Tensor>,
    /// Per DeltaNet layer: `[max_slots, conv_dim, conv_kernel - 1]` in model dtype.
    pub conv_states: Vec<Tensor>,
    /// Number of DeltaNet layers.
    pub num_layers: usize,
    /// Free slot indices available for allocation.
    free_slots: VecDeque<u32>,
    /// Total number of slots in the pool.
    pub max_slots: u32,
}

impl DeltaNetPool {
    /// Create a new DeltaNet state pool.
    ///
    /// `model_dtype` is used for conv_state tensors; recurrent_state always uses F32
    /// for numerical stability (matching the model's internal delta_rule_step behavior).
    pub fn new(
        cfg: &DeltaNetPoolConfig,
        max_slots: u32,
        model_dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut recurrent_states = Vec::with_capacity(cfg.num_deltanet_layers);
        let mut conv_states = Vec::with_capacity(cfg.num_deltanet_layers);

        let conv_state_len = cfg.conv_kernel.saturating_sub(1);

        for _ in 0..cfg.num_deltanet_layers {
            recurrent_states.push(Tensor::zeros(
                (
                    max_slots as usize,
                    cfg.num_v_heads,
                    cfg.head_v_dim,
                    cfg.head_k_dim,
                ),
                DType::F32,
                device,
            )?);
            conv_states.push(Tensor::zeros(
                (max_slots as usize, cfg.conv_dim, conv_state_len),
                model_dtype,
                device,
            )?);
        }

        let free_slots: VecDeque<u32> = (0..max_slots).collect();

        Ok(Self {
            recurrent_states,
            conv_states,
            num_layers: cfg.num_deltanet_layers,
            free_slots,
            max_slots,
        })
    }

    /// Allocate a slot for a new request. Returns `None` if the pool is exhausted.
    ///
    /// The slot's per-layer recurrent and conv state is zeroed before the slot
    /// is handed out so the caller sees a clean state (matches vLLM / SGLang
    /// semantics — requests that reuse a previously freed slot must not see
    /// stale state from the previous occupant).
    pub fn allocate(&mut self) -> Option<u32> {
        let slot = self.free_slots.pop_front()?;
        if let Err(e) = self.reset_slot(slot) {
            // Put the slot back and surface the failure as "no slot". This
            // keeps the signature Option<u32>; the zero-fill is a tiny
            // slice_set per layer and should never realistically fail.
            tracing::error!(?e, slot, "DeltaNetPool::allocate reset_slot failed");
            self.free_slots.push_front(slot);
            return None;
        }
        Some(slot)
    }

    /// Free a slot when a request finishes or is preempted.
    pub fn free(&mut self, slot: u32) {
        self.free_slots.push_back(slot);
    }

    /// Zero the recurrent and conv state at `slot` across every DeltaNet layer.
    /// Uses `slice_set`, which goes through candle's interior-mutable storage
    /// and so only needs `&self`.
    pub fn reset_slot(&self, slot: u32) -> Result<()> {
        let slot = slot as usize;
        for rs in &self.recurrent_states {
            let zero = Tensor::zeros(&rs.dims()[1..], rs.dtype(), rs.device())?;
            rs.slice_set(&zero.unsqueeze(0)?, 0, slot)?;
        }
        for cs in &self.conv_states {
            let zero = Tensor::zeros(&cs.dims()[1..], cs.dtype(), cs.device())?;
            cs.slice_set(&zero.unsqueeze(0)?, 0, slot)?;
        }
        Ok(())
    }

    /// Number of available (unallocated) slots.
    pub fn available(&self) -> usize {
        self.free_slots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DeltaNetPoolConfig {
        DeltaNetPoolConfig {
            num_deltanet_layers: 3,
            num_v_heads: 4,
            head_k_dim: 64,
            head_v_dim: 64,
            conv_dim: 384,
            conv_kernel: 4,
        }
    }

    #[test]
    fn test_pool_creation() {
        let cfg = test_config();
        let pool = DeltaNetPool::new(&cfg, 8, DType::BF16, &Device::Cpu).unwrap();
        assert_eq!(pool.num_layers, 3);
        assert_eq!(pool.max_slots, 8);
        assert_eq!(pool.available(), 8);
        assert_eq!(pool.recurrent_states.len(), 3);
        assert_eq!(pool.conv_states.len(), 3);

        // Check shapes
        assert_eq!(pool.recurrent_states[0].dims(), &[8, 4, 64, 64]);
        assert_eq!(pool.recurrent_states[0].dtype(), DType::F32);
        assert_eq!(pool.conv_states[0].dims(), &[8, 384, 3]);
        assert_eq!(pool.conv_states[0].dtype(), DType::BF16);
    }

    #[test]
    fn test_allocate_and_free() {
        let cfg = test_config();
        let mut pool = DeltaNetPool::new(&cfg, 3, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(pool.available(), 3);

        let s0 = pool.allocate().unwrap();
        assert_eq!(s0, 0);
        assert_eq!(pool.available(), 2);

        let s1 = pool.allocate().unwrap();
        assert_eq!(s1, 1);

        let s2 = pool.allocate().unwrap();
        assert_eq!(s2, 2);
        assert_eq!(pool.available(), 0);

        // Pool exhausted
        assert!(pool.allocate().is_none());

        // Free slot 1, then allocate again
        pool.free(s1);
        assert_eq!(pool.available(), 1);
        let s3 = pool.allocate().unwrap();
        assert_eq!(s3, 1); // reused slot
    }

    #[test]
    fn test_state_read_write() {
        let cfg = test_config();
        let mut pool = DeltaNetPool::new(&cfg, 4, DType::F32, &Device::Cpu).unwrap();

        let slot = pool.allocate().unwrap();
        let layer = 0;

        // Write a state to the pool
        let state = Tensor::ones((4, 64, 64), DType::F32, &Device::Cpu).unwrap();
        pool.recurrent_states[layer]
            .slice_set(&state.unsqueeze(0).unwrap(), 0, slot as usize)
            .unwrap();

        // Read it back
        let read_back = pool.recurrent_states[layer].get(slot as usize).unwrap();
        assert_eq!(read_back.dims(), &[4, 64, 64]);

        // Verify values are 1.0
        let flat: Vec<f32> = read_back.flatten_all().unwrap().to_vec1().unwrap();
        assert!((flat[0] - 1.0).abs() < 1e-6);

        // Verify another slot is still zeros
        let other_slot = pool.allocate().unwrap();
        let other = pool.recurrent_states[layer]
            .get(other_slot as usize)
            .unwrap();
        let flat2: Vec<f32> = other.flatten_all().unwrap().to_vec1().unwrap();
        assert!((flat2[0]).abs() < 1e-6);
    }
}

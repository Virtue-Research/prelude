//! Forward execution paths — all task types (generate, classify, embed)
//! and paged attention operations (prefill, decode).

use std::sync::{Mutex, MutexGuard};

use super::types::PreTokenizedBatchItem;
use crate::cache::deltanet_pool::DeltaNetPool;
use crate::engine::{DType, Device, EngineError, Tensor, tensor_err};

mod classify;
mod embed;
mod generate;
mod paged_decode;
mod paged_mixed;
mod paged_prefill;
mod prefill;
mod prefill_output;

fn cumulative_lengths_u32(
    lengths: impl IntoIterator<Item = usize>,
) -> Result<(Vec<u32>, usize), EngineError> {
    let mut offsets = vec![0u32];
    let mut running = 0u32;
    let mut max_len = 0usize;

    for len in lengths {
        let len_u32 = u32::try_from(len)
            .map_err(|_| EngineError::Internal("sequence length exceeds u32".into()))?;
        running = running
            .checked_add(len_u32)
            .ok_or_else(|| EngineError::Internal("packed sequence length exceeds u32".into()))?;
        offsets.push(running);
        max_len = max_len.max(len);
    }

    Ok((offsets, max_len))
}

fn block_tables_tensor<'a>(
    tables: impl IntoIterator<Item = &'a [u32]>,
    device: &Device,
) -> Result<Tensor, EngineError> {
    let tables: Vec<&[u32]> = tables.into_iter().collect();
    let batch_size = tables.len();
    let max_blocks = tables.iter().map(|bt| bt.len()).max().unwrap_or(0);

    if max_blocks == 0 {
        return Tensor::zeros((batch_size, 0), DType::U32, device).map_err(tensor_err);
    }

    let mut flat: Vec<u32> = Vec::with_capacity(batch_size * max_blocks);
    for table in tables {
        flat.extend_from_slice(table);
        flat.resize(flat.len() + max_blocks - table.len(), 0);
    }

    Tensor::from_vec(flat, (batch_size, max_blocks), device)
        .map_err(tensor_err)?
        .to_dtype(DType::U32)
        .map_err(tensor_err)
}

fn lock_deltanet_pool(
    pool: Option<&Mutex<DeltaNetPool>>,
) -> Result<Option<MutexGuard<'_, DeltaNetPool>>, EngineError> {
    pool.map(|mutex| {
        mutex
            .lock()
            .map_err(|e| EngineError::Internal(format!("deltanet pool lock: {e}")))
    })
    .transpose()
}

fn pretokenized_token_groups<R>(items: &[PreTokenizedBatchItem<R>]) -> Vec<&[Vec<u32>]> {
    items.iter().map(|item| item.token_ids.as_slice()).collect()
}

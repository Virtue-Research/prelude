//! Shared helpers for batch runtime and CPU batch runtime.

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::engine::{
    Engine, EngineError, PreTokenizedClassifyItem, PreTokenizedEmbedItem, PreparedGenerateRequest,
};
use crate::types::{ClassifyResult, EmbedResult, GenerateResult};

use super::request_state::{
    GenerationRequestState, InFlightClassifyRequest, InFlightEmbedRequest,
};

// ---------------------------------------------------------------------------
// Error cloning
// ---------------------------------------------------------------------------

pub(crate) fn clone_engine_error(err: &EngineError) -> EngineError {
    match err {
        EngineError::Unavailable(msg) => EngineError::Unavailable(msg.clone()),
        EngineError::InvalidRequest(msg) => EngineError::InvalidRequest(msg.clone()),
        EngineError::Internal(msg) => EngineError::Internal(msg.clone()),
    }
}

// ---------------------------------------------------------------------------
// Tokenize generation batch
// ---------------------------------------------------------------------------

/// Tokenize generation requests in parallel using rayon.
///
/// Splits cached (pre-tokenized at routing time) from uncached, tokenizes uncached
/// in parallel, merges results, and re-indexes. Returns None if all fail.
pub(crate) fn tokenize_gen_batch(
    engine: &Engine,
    batch: Vec<GenerationRequestState>,
) -> Option<(Vec<GenerationRequestState>, Vec<PreparedGenerateRequest>)> {
    let mut cached_inflight = Vec::new();
    let mut cached_prepared = Vec::new();
    let mut uncached: Vec<(usize, GenerationRequestState)> = Vec::new();

    for (idx, mut req) in batch.into_iter().enumerate() {
        if let Some(mut p) = req.cached_prepared.take() {
            p.request_idx = idx;
            cached_inflight.push(req);
            cached_prepared.push(p);
        } else {
            uncached.push((idx, req));
        }
    }

    // Parallel tokenize uncached
    let uncached_results: Vec<_> = if uncached.is_empty() {
        Vec::new()
    } else {
        uncached
            .par_iter()
            .map(|(idx, req)| {
                engine
                    .prepare_generate_request(req.request(), *idx)
                    .map(|p| (*idx, p))
                    .map_err(|e| (*idx, e))
            })
            .collect()
    };

    let mut final_inflight = Vec::with_capacity(cached_inflight.len() + uncached.len());
    let mut final_prepared = Vec::with_capacity(final_inflight.capacity());

    final_inflight.extend(cached_inflight);
    final_prepared.extend(cached_prepared);

    let mut uncached_map: Vec<_> = uncached.into_iter().collect();
    for result in uncached_results {
        match result {
            Ok((idx, item)) => {
                if let Some(pos) = uncached_map.iter().position(|(i, _)| *i == idx) {
                    let (_, req) = uncached_map.remove(pos);
                    final_inflight.push(req);
                    final_prepared.push(item);
                }
            }
            Err((idx, e)) => {
                if let Some(pos) = uncached_map.iter().position(|(i, _)| *i == idx) {
                    let (_, req) = uncached_map.remove(pos);
                    req.fail(e);
                }
            }
        }
    }

    // Re-index
    for (i, p) in final_prepared.iter_mut().enumerate() {
        p.request_idx = i;
    }

    if final_inflight.is_empty() {
        None
    } else {
        Some((final_inflight, final_prepared))
    }
}

// ---------------------------------------------------------------------------
// Collect classify / embed batches
// ---------------------------------------------------------------------------

/// Collect and tokenize classify requests up to max_batch_size sequences.
pub(crate) fn collect_classify_batch(
    engine: &Engine,
    queue: &mut VecDeque<InFlightClassifyRequest>,
    max_batch_size: usize,
) -> Option<(Vec<InFlightClassifyRequest>, Vec<PreTokenizedClassifyItem>)> {
    if queue.is_empty() {
        return None;
    }

    let mut valid_inflight = Vec::new();
    let mut pretokenized = Vec::new();
    let mut total_sequences = 0usize;
    let mut idx = 0usize;

    while let Some(inflight) = queue.pop_front() {
        match engine.tokenize_batch(&inflight.request.inputs) {
            Ok((token_ids, total_tokens)) => {
                let num_sequences = token_ids.len();
                if total_sequences > 0 && total_sequences + num_sequences > max_batch_size {
                    queue.push_front(inflight);
                    break;
                }
                total_sequences += num_sequences;
                pretokenized.push(PreTokenizedClassifyItem {
                    request_idx: idx,
                    request: inflight.request.clone(),
                    token_ids,
                    total_tokens,
                });
                valid_inflight.push(inflight);
                idx += 1;
            }
            Err(e) => {
                let _ = inflight.response.send(Err(e));
            }
        }
    }

    if valid_inflight.is_empty() {
        None
    } else {
        Some((valid_inflight, pretokenized))
    }
}

/// Collect and tokenize embed requests up to max_batch_size sequences.
pub(crate) fn collect_embed_batch(
    engine: &Engine,
    queue: &mut VecDeque<InFlightEmbedRequest>,
    max_batch_size: usize,
) -> Option<(Vec<InFlightEmbedRequest>, Vec<PreTokenizedEmbedItem>)> {
    if queue.is_empty() {
        return None;
    }

    let mut valid_inflight = Vec::new();
    let mut pretokenized = Vec::new();
    let mut total_sequences = 0usize;
    let mut idx = 0usize;

    while let Some(inflight) = queue.pop_front() {
        match engine.tokenize_batch(&inflight.request.inputs) {
            Ok((token_ids, total_tokens)) => {
                let num_sequences = token_ids.len();
                if total_sequences > 0 && total_sequences + num_sequences > max_batch_size {
                    queue.push_front(inflight);
                    break;
                }
                total_sequences += num_sequences;
                pretokenized.push(PreTokenizedEmbedItem {
                    request_idx: idx,
                    request: inflight.request.clone(),
                    token_ids,
                    total_tokens,
                });
                valid_inflight.push(inflight);
                idx += 1;
            }
            Err(e) => {
                let _ = inflight.response.send(Err(e));
            }
        }
    }

    if valid_inflight.is_empty() {
        None
    } else {
        Some((valid_inflight, pretokenized))
    }
}

// ---------------------------------------------------------------------------
// Dispatch results
// ---------------------------------------------------------------------------

/// Dispatch generation results to callers. No deferred-abort checking.
pub(crate) fn dispatch_generate_results(
    inflight: Vec<GenerationRequestState>,
    result: Result<Vec<GenerateResult>, EngineError>,
) {
    match result {
        Ok(results) if results.len() == inflight.len() => {
            for (req, res) in inflight.into_iter().zip(results) {
                req.finish(Ok(res));
            }
        }
        Ok(results) => {
            let msg = format!(
                "batch output mismatch: expected {}, got {}",
                inflight.len(),
                results.len()
            );
            for req in inflight {
                req.fail(EngineError::Internal(msg.clone()));
            }
        }
        Err(e) => {
            for req in inflight {
                req.fail(clone_engine_error(&e));
            }
        }
    }
}

/// Dispatch classify results to callers via oneshot channels.
pub(crate) fn dispatch_classify_results(
    inflight: Vec<InFlightClassifyRequest>,
    result: Result<Vec<ClassifyResult>, EngineError>,
) {
    match result {
        Ok(results) if results.len() == inflight.len() => {
            for (req, res) in inflight.into_iter().zip(results) {
                let _ = req.response.send(Ok(res));
            }
        }
        Ok(results) => {
            let msg = format!(
                "classify batch output size mismatch, expected {}, got {}",
                inflight.len(),
                results.len()
            );
            for req in inflight {
                let _ = req.response.send(Err(EngineError::Internal(msg.clone())));
            }
        }
        Err(e) => {
            for req in inflight {
                let _ = req.response.send(Err(clone_engine_error(&e)));
            }
        }
    }
}

/// Dispatch embed results to callers via oneshot channels.
pub(crate) fn dispatch_embed_results(
    inflight: Vec<InFlightEmbedRequest>,
    result: Result<Vec<EmbedResult>, EngineError>,
) {
    match result {
        Ok(results) if results.len() == inflight.len() => {
            for (req, res) in inflight.into_iter().zip(results) {
                let _ = req.response.send(Ok(res));
            }
        }
        Ok(results) => {
            let msg = format!(
                "embed batch output size mismatch, expected {}, got {}",
                inflight.len(),
                results.len()
            );
            for req in inflight {
                let _ = req.response.send(Err(EngineError::Internal(msg.clone())));
            }
        }
        Err(e) => {
            for req in inflight {
                let _ = req.response.send(Err(clone_engine_error(&e)));
            }
        }
    }
}

//! CUDA Executor: dedicated GPU worker thread + CUDA graph replay.
//!
//! The CudaExecutor owns a dedicated GPU worker thread. `submit()` sends
//! work via a channel (non-blocking). The worker thread owns the CUDA graph
//! cache and tries graph replay for decode steps before falling back to eager.

use std::sync::Arc;
use std::time::Instant;

use prelude_core::engine::executor::{
    ExecutionHandle, Executor, ForwardBatch, ModelOutput, StepRequest,
};
use prelude_core::engine::{Engine, EngineError, OwnedBatchDecodeSeq};
use prelude_core::tensor::{D, Tensor};

use crate::cuda_graph::DecodeGraphCache;

struct GpuWork {
    batch: ForwardBatch,
    result_tx: tokio::sync::oneshot::Sender<Result<ModelOutput, EngineError>>,
}

/// CUDA execution strategy: dedicated GPU worker thread with graph cache.
pub struct CudaExecutor {
    _engine: Arc<Engine>,
    work_tx: tokio::sync::mpsc::UnboundedSender<GpuWork>,
    _worker: std::thread::JoinHandle<()>,
}

impl CudaExecutor {
    pub fn new(engine: Arc<Engine>) -> Self {
        let (work_tx, mut work_rx) = tokio::sync::mpsc::unbounded_channel::<GpuWork>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let worker_engine = engine.clone();
        let worker = std::thread::Builder::new()
            .name("gpu-executor".into())
            .spawn(move || {
                // Initialize CUDA graph cache. Models that don't support
                // graph capture (e.g. MoE with dynamic expert routing) set
                // supports_cuda_graph=false in their RuntimeCaps — skip
                // warmup entirely to avoid wasted work and spurious kernel
                // dispatch failures during stream capture.
                let block_size = worker_engine
                    .cache
                    .paged_pool
                    .as_ref()
                    .map(|p| p.block_size)
                    .unwrap_or(128);
                let has_deltanet = worker_engine.cache.deltanet_pool.is_some();
                let supports_graph = worker_engine.runtime_caps().supports_cuda_graph;
                let mut graph_cache = DecodeGraphCache::new(
                    &worker_engine.engine_config,
                    block_size,
                    has_deltanet,
                    supports_graph,
                );
                if supports_graph {
                    graph_cache.warmup_all(&worker_engine);
                }
                warmup_mixed_prefill(&worker_engine, false);
                warmup_mixed_prefill(&worker_engine, true);
                let _ = ready_tx.send(());

                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("GPU executor worker runtime");
                rt.block_on(async {
                    while let Some(work) = work_rx.recv().await {
                        let result = execute_work(&worker_engine, work.batch, &mut graph_cache);
                        let _ = work.result_tx.send(result);
                    }
                });
                tracing::debug!("GPU executor worker exited");
            })
            .expect("spawn GPU executor worker thread");
        let _ = ready_rx.recv();
        Self {
            _engine: engine,
            work_tx,
            _worker: worker,
        }
    }
}

impl Executor for CudaExecutor {
    fn submit(&self, batch: ForwardBatch) -> Result<ExecutionHandle, EngineError> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        self.work_tx
            .send(GpuWork { batch, result_tx })
            .map_err(|_| EngineError::Internal("GPU worker thread has exited".into()))?;
        Ok(ExecutionHandle::new(result_rx))
    }
}

/// Execute a forward batch, using CUDA graph replay for decode when possible.
fn execute_work(
    engine: &Engine,
    batch: ForwardBatch,
    graph_cache: &mut DecodeGraphCache,
) -> Result<ModelOutput, EngineError> {
    match batch {
        ForwardBatch::Decode {
            tokens,
            positions,
            block_tables,
            deltanet_slots,
            sample_greedy,
        } => {
            let bs = tokens.len();
            let t0 = Instant::now();

            // Build OwnedBatchDecodeSeq for graph cache lookup
            let seqs: Vec<OwnedBatchDecodeSeq> = tokens
                .iter()
                .enumerate()
                .map(|(i, &token)| OwnedBatchDecodeSeq {
                    token,
                    position: positions[i],
                    context_len: positions[i] + 1,
                    block_table: block_tables[i].clone(),
                    deltanet_slot: deltanet_slots.as_ref().and_then(|s| s.get(i).copied()),
                })
                .collect();

            // Try CUDA graph replay first
            if let Some(result) = graph_cache.try_replay(engine, &seqs) {
                let logits = result?;
                let sampled_tokens = if sample_greedy {
                    greedy_argmax_tokens(&logits)
                } else {
                    None
                };
                let elapsed_us = t0.elapsed().as_micros();
                tracing::debug!(
                    bs,
                    elapsed_us,
                    sample_greedy,
                    path = "cuda_graph",
                    "decode step"
                );
                return Ok(ModelOutput {
                    logits,
                    item_seq_counts: vec![],
                    prefill_results: vec![],
                    sampled_tokens,
                });
            }

            // Fallback to eager execution
            let mut result = engine.forward_batch(ForwardBatch::Decode {
                tokens,
                positions,
                block_tables,
                deltanet_slots,
                sample_greedy,
            });
            if let Ok(output) = result.as_mut() {
                if sample_greedy {
                    output.sampled_tokens = greedy_argmax_tokens(&output.logits);
                }
            }
            let elapsed_us = t0.elapsed().as_micros();
            tracing::debug!(bs, elapsed_us, sample_greedy, path = "eager", "decode step");
            result
        }
        ForwardBatch::Mixed {
            requests,
            sample_greedy,
        } => {
            let t0 = Instant::now();
            let mut result = engine.forward_batch(ForwardBatch::Mixed {
                requests,
                sample_greedy,
            });
            if let Ok(output) = result.as_mut() {
                if sample_greedy {
                    output.sampled_tokens = greedy_argmax_tokens(&output.logits);
                }
            }
            let elapsed_us = t0.elapsed().as_micros();
            tracing::debug!(elapsed_us, sample_greedy, path = "mixed", "forward step");
            result
        }
        other => {
            let t0 = Instant::now();
            let result = engine.forward_batch(other);
            let elapsed_us = t0.elapsed().as_micros();
            tracing::debug!(elapsed_us, path = "mixed", "forward step");
            result
        }
    }
}

fn warmup_mixed_prefill(engine: &Engine, needs_kv_cache: bool) {
    let token_budget = engine.engine_config.runtime.profile_tokens.clamp(4, 8192);
    let batch = 2usize;
    let seq_len = (token_budget / batch).clamp(2, 4096);
    let path = if needs_kv_cache {
        "paged_prefill"
    } else {
        "ragged_prefill"
    };
    let t0 = Instant::now();

    let mut block_tables = vec![Vec::new(); batch];
    if needs_kv_cache {
        let Some(bm_mutex) = engine.cache.block_manager_arc() else {
            tracing::debug!("mixed prefill warmup skipped: block manager unavailable");
            return;
        };
        let mut bm = match bm_mutex.lock() {
            Ok(bm) => bm,
            Err(e) => {
                tracing::warn!(error = %e, "mixed prefill warmup skipped: block manager lock");
                return;
            }
        };
        for table in &mut block_tables {
            let Some(allocated) = bm.allocate_for_tokens(seq_len) else {
                tracing::warn!(
                    seq_len,
                    batch,
                    "mixed prefill warmup skipped: insufficient KV blocks"
                );
                drop(bm);
                free_warmup_blocks(engine, &block_tables);
                return;
            };
            *table = allocated;
        }
    }

    let mut deltanet_slots = vec![None; batch];
    if needs_kv_cache && let Some(pool_mutex) = &engine.cache.deltanet_pool {
        let mut pool = match pool_mutex.lock() {
            Ok(pool) => pool,
            Err(e) => {
                tracing::warn!(error = %e, "mixed prefill warmup skipped: DeltaNet pool lock");
                free_warmup_blocks(engine, &block_tables);
                return;
            }
        };
        for slot in &mut deltanet_slots {
            let Some(allocated) = pool.allocate() else {
                tracing::warn!(
                    seq_len,
                    batch,
                    "mixed prefill warmup skipped: insufficient DeltaNet slots"
                );
                drop(pool);
                free_warmup_deltanet_slots(engine, &deltanet_slots);
                free_warmup_blocks(engine, &block_tables);
                return;
            };
            *slot = Some(allocated);
        }
    }

    let requests = (0..batch)
        .map(|i| StepRequest {
            tokens: vec![0; seq_len],
            context_len: seq_len,
            position_start: 0,
            block_table: block_tables[i].clone(),
            is_prefill_final: true,
            is_prefill_partial: false,
            deltanet_slot: deltanet_slots[i],
            prompt_logprobs: None,
            needs_kv_cache,
        })
        .collect();

    let result = engine.forward_batch(ForwardBatch::Mixed {
        requests,
        sample_greedy: true,
    });

    free_warmup_deltanet_slots(engine, &deltanet_slots);
    free_warmup_blocks(engine, &block_tables);

    match result {
        Ok(_) => tracing::info!(
            path,
            batch,
            seq_len,
            elapsed_ms = t0.elapsed().as_millis(),
            "mixed prefill warmup complete"
        ),
        Err(e) => tracing::warn!(path, batch, seq_len, error = %e, "mixed prefill warmup failed"),
    }
}

fn free_warmup_blocks(engine: &Engine, block_tables: &[Vec<u32>]) {
    let Some(bm_mutex) = engine.cache.block_manager_arc() else {
        return;
    };
    if let Ok(mut bm) = bm_mutex.lock() {
        for table in block_tables {
            bm.free(table);
        }
    }
}

fn free_warmup_deltanet_slots(engine: &Engine, slots: &[Option<u32>]) {
    let Some(pool_mutex) = &engine.cache.deltanet_pool else {
        return;
    };
    if let Ok(mut pool) = pool_mutex.lock() {
        for slot in slots.iter().flatten() {
            pool.free(*slot);
        }
    }
}

fn greedy_argmax_tokens(logits: &Tensor) -> Option<Vec<u32>> {
    logits
        .argmax(D::Minus1)
        .and_then(|tokens| tokens.to_vec1::<u32>())
        .ok()
}

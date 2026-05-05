//! CUDA Executor: dedicated GPU worker thread + CUDA graph replay.
//!
//! The CudaExecutor owns a dedicated GPU worker thread. `submit()` sends
//! work via a channel (non-blocking). The worker thread owns the CUDA graph
//! cache and tries graph replay for decode steps before falling back to eager.

use std::sync::Arc;
use std::time::Instant;

use prelude_core::engine::executor::{ExecutionHandle, Executor, ForwardBatch, ModelOutput};
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

fn greedy_argmax_tokens(logits: &Tensor) -> Option<Vec<u32>> {
    logits
        .argmax(D::Minus1)
        .and_then(|tokens| tokens.to_vec1::<u32>())
        .ok()
}

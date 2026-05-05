//! Device execution trait.
//!
//! All executors (GPU and CPU) use the same async pattern:
//! 1. `submit()` sends work to a dedicated worker thread, returns immediately
//! 2. The caller awaits `handle.recv()` — tokio interleaves other tasks while waiting
//!
//! This gives zero-overhead async for GPU (CPU thread is idle during GPU work)
//! and keeps tokio workers free for SSE streaming, health checks, etc.
//!
//! # Registration
//!
//! Device crates register their executor via `register_executor()` at startup:
//!
//! ```ignore
//! prelude_core::engine::executor::register_executor(ExecutorBackend {
//!     name: "cuda",
//!     priority: 100,
//!     probe: || cuda_available(),
//!     supports: |d| d.is_cuda(),
//!     create: |e| Box::new(CudaExecutor::new(e)),
//! });
//! ```

use crate::tensor::Tensor;

use super::EngineError;

// ── Execution handle ───────────────────────────────────────────────

/// Handle for an in-flight forward pass.
///
/// All executors (GPU and CPU) use a dedicated worker thread and return
/// results via a oneshot channel. Await with `.recv()` in the AR loop.
pub struct ExecutionHandle {
    rx: tokio::sync::oneshot::Receiver<Result<ModelOutput, super::EngineError>>,
}

impl ExecutionHandle {
    pub fn new(
        rx: tokio::sync::oneshot::Receiver<Result<ModelOutput, super::EngineError>>,
    ) -> Self {
        Self { rx }
    }

    /// Await the result. Yields to tokio while the worker thread runs,
    /// allowing SSE streaming handlers to interleave naturally.
    pub async fn recv(self) -> Result<ModelOutput, super::EngineError> {
        self.rx.await.unwrap_or_else(|_| {
            Err(super::EngineError::Internal(
                "executor worker dropped result channel".into(),
            ))
        })
    }
}

// ── Model output ───────────────────────────────────────────────────

/// Output from a single forward pass.
///
/// Sampling and postprocessing normally happen in the scheduling loop (core).
/// Device executors may optionally return greedy argmax tokens to keep the
/// per-step hot path on the worker.
#[derive(Debug)]
pub struct ModelOutput {
    /// Raw logits: `[batch_size, vocab_size]` (for decode/prefill/classify)
    /// or `[batch_size, hidden_dim]` (for embed).
    pub logits: Tensor,
    /// Number of sequences per input item (for one-shot classify/embed grouping).
    /// Empty for Prefill/Decode batches.
    pub item_seq_counts: Vec<usize>,
    /// Per-request prefill metadata (block tables, prompt_len, etc.).
    /// Populated only for Prefill batches; empty for Decode/OneShot.
    pub prefill_results: Vec<super::BatchPrefillResult>,
    /// Optional row-aligned greedy tokens.
    /// Populated only when the executor was asked to use its greedy fast path.
    pub sampled_tokens: Option<Vec<u32>>,
}

// ── Forward batch ──────────────────────────────────────────────────

/// What the Executor should run. Built by the scheduling loop from
/// `ArSequenceState` + `SchedulerStep`, then passed to `submit()`.
/// Per-request info for a unified mixed forward batch.
#[derive(Debug)]
pub struct StepRequest {
    /// Tokens to process this step (chunk for prefill, 1 for decode).
    pub tokens: Vec<u32>,
    /// Total KV context length (including tokens from previous chunks + this step's tokens).
    pub context_len: usize,
    /// Starting position for this request's new tokens.
    pub position_start: usize,
    /// Block table (accumulated from previous chunks).
    pub block_table: Vec<u32>,
    /// Is this a prefill final chunk? (needs first-token processing in AR loop)
    pub is_prefill_final: bool,
    /// Is this a partial prefill chunk? (discard sampled token)
    pub is_prefill_partial: bool,
    /// DeltaNet pool slot.
    pub deltanet_slot: Option<u32>,
    /// Whether to compute per-token prompt logprobs (for PPL / echo).
    pub prompt_logprobs: Option<u32>,
    /// Whether this request needs paged KV cache writes for later chunks/decode.
    pub needs_kv_cache: bool,
}

pub enum ForwardBatch {
    /// Unified mixed batch: prefill chunks (Q=chunk_len) + decode (Q=1)
    /// in a single forward pass. Used by chunked-prefill scheduler.
    Mixed {
        requests: Vec<StepRequest>,
        /// Rows that need sampling are pure greedy with no logprobs.
        sample_greedy: bool,
    },
    /// Pure decode batch (Q=1 for all): eligible for CUDA graph replay.
    Decode {
        tokens: Vec<u32>,
        positions: Vec<usize>,
        block_tables: Vec<Vec<u32>>,
        /// DeltaNet pool slots for hybrid models (None = non-hybrid).
        deltanet_slots: Option<Vec<u32>>,
        /// Pure greedy decode with no logprobs can be sampled in the executor.
        sample_greedy: bool,
    },
    /// One-shot forward for classify/embed (no decode loop).
    /// Groups of token sequences — each group is one input item which may
    /// contain multiple segments (e.g., query + passage for reranking).
    OneShot {
        token_groups: Vec<Vec<Vec<u32>>>,
        task: super::TaskKind,
    },
}

// ── Executor trait ─────────────────────────────────────────────────

/// Device execution strategy.
///
/// Implementors handle:
/// - Building device-specific tensors from `ForwardBatch`
/// - Running `model.forward()` on the device
/// - Device-specific optimizations (GPU queue, CUDA graph, HIP graph, ...)
///
/// The trait is intentionally minimal — two methods. All scheduling paradigm
/// logic (AR, diffusion LLM, TTS, ...) lives in `engine/run/` and calls
/// these two methods uniformly.
pub trait Executor: Send + Sync + 'static {
    /// Submit a forward pass to the device. Returns immediately.
    ///
    /// Both GPU and CPU executors send work to a dedicated worker thread
    /// and return an `ExecutionHandle` backed by a oneshot channel.
    /// The caller awaits the result with `handle.recv().await`.
    fn submit(&self, batch: ForwardBatch) -> Result<ExecutionHandle, EngineError>;
}

// ── Registration ───────────────────────────────────────────────────

use crate::tensor::Device;
use std::sync::Mutex;

/// Factory function signature: takes shared engine state, returns a device executor.
pub type ExecutorFactory = fn(engine: std::sync::Arc<super::engine::Engine>) -> Box<dyn Executor>;

/// A registered executor backend with priority-based selection.
pub struct ExecutorBackend {
    /// Human-readable name (e.g. "cuda", "cpu").
    pub name: &'static str,
    /// Higher priority wins when multiple executors match.
    pub priority: u32,
    /// Runtime probe: returns `true` if this executor is usable.
    pub probe: fn() -> bool,
    /// Returns `true` if this executor handles the given device kind.
    pub supports: fn(&Device) -> bool,
    /// Factory that creates the executor for an engine.
    pub create: ExecutorFactory,
}

static EXECUTOR_REGISTRY: Mutex<Vec<ExecutorBackend>> = Mutex::new(Vec::new());

/// Register an executor backend. Call during startup before engine creation.
pub fn register_executor(entry: ExecutorBackend) {
    EXECUTOR_REGISTRY.lock().unwrap().push(entry);
}

/// Create the best matching executor for the engine's device, or return `None`.
pub fn create_executor(engine: std::sync::Arc<super::engine::Engine>) -> Option<Box<dyn Executor>> {
    let device = engine.device().clone();
    let backends = EXECUTOR_REGISTRY.lock().unwrap();
    let mut best: Option<&ExecutorBackend> = None;
    for b in backends.iter() {
        if (b.supports)(&device) && (b.probe)() {
            if best.map_or(true, |cur| b.priority > cur.priority) {
                best = Some(b);
            }
        }
    }
    best.map(|b| {
        tracing::info!(
            "executor for {:?}: {} (priority {})",
            device,
            b.name,
            b.priority
        );
        (b.create)(engine)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test executor: computes result in submit, sends via oneshot (same as real executors).
    struct TestExecutor {
        vocab_size: usize,
    }

    impl Executor for TestExecutor {
        fn submit(&self, batch: ForwardBatch) -> Result<ExecutionHandle, EngineError> {
            let (batch_size, item_seq_counts) = match &batch {
                ForwardBatch::Mixed { requests, .. } => (requests.len(), vec![]),
                ForwardBatch::Decode { tokens, .. } => (tokens.len(), vec![]),
                ForwardBatch::OneShot { token_groups, .. } => {
                    let counts: Vec<usize> = token_groups.iter().map(|g| g.len()).collect();
                    let total: usize = counts.iter().sum();
                    (total, counts)
                }
            };
            let logits = Tensor::zeros(
                (batch_size, self.vocab_size),
                crate::tensor::DType::F32,
                &crate::tensor::Device::Cpu,
            )
            .map_err(|e| EngineError::Internal(format!("{e}")))?;

            let (tx, rx) = tokio::sync::oneshot::channel();
            let _ = tx.send(Ok(ModelOutput {
                logits,
                item_seq_counts,
                prefill_results: vec![],
                sampled_tokens: None,
            }));
            Ok(ExecutionHandle::new(rx))
        }
    }

    /// Helper: submit + recv in a blocking tokio context.
    fn submit_recv(
        executor: &TestExecutor,
        batch: ForwardBatch,
    ) -> Result<ModelOutput, EngineError> {
        let handle = executor.submit(batch)?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(handle.recv())
    }

    #[test]
    fn submit_recv_prefill() {
        let output = submit_recv(
            &TestExecutor { vocab_size: 100 },
            ForwardBatch::Mixed {
                requests: vec![],
                sample_greedy: false,
            },
        )
        .unwrap();
        assert_eq!(output.logits.dims(), &[0, 100]);
    }

    #[test]
    fn submit_recv_decode() {
        let output = submit_recv(
            &TestExecutor { vocab_size: 50 },
            ForwardBatch::Decode {
                tokens: vec![1, 2, 3],
                positions: vec![10, 20, 30],
                block_tables: vec![vec![0, 1], vec![0, 2], vec![0, 3]],
                deltanet_slots: None,
                sample_greedy: false,
            },
        )
        .unwrap();
        assert_eq!(output.logits.dims(), &[3, 50]);
    }

    #[test]
    fn submit_recv_oneshot() {
        let output = submit_recv(
            &TestExecutor { vocab_size: 8 },
            ForwardBatch::OneShot {
                token_groups: vec![vec![vec![1, 2, 3]], vec![vec![4, 5], vec![6, 7]]],
                task: crate::engine::TaskKind::Classify,
            },
        )
        .unwrap();
        assert_eq!(output.logits.dims(), &[3, 8]);
        assert_eq!(output.item_seq_counts, vec![1, 2]);
    }

    #[test]
    fn submit_recv_decode_with_deltanet() {
        let output = submit_recv(
            &TestExecutor { vocab_size: 32 },
            ForwardBatch::Decode {
                tokens: vec![10, 20],
                positions: vec![5, 8],
                block_tables: vec![vec![0], vec![1]],
                deltanet_slots: Some(vec![0, 1]),
                sample_greedy: false,
            },
        )
        .unwrap();
        assert_eq!(output.logits.dims(), &[2, 32]);
    }

    #[test]
    fn submit_recv_oneshot_embed() {
        let output = submit_recv(
            &TestExecutor { vocab_size: 64 },
            ForwardBatch::OneShot {
                token_groups: vec![vec![vec![1, 2, 3, 4]]],
                task: crate::engine::TaskKind::Embed,
            },
        )
        .unwrap();
        assert_eq!(output.logits.dims(), &[1, 64]);
        assert_eq!(output.item_seq_counts, vec![1]);
    }

    #[test]
    fn submit_recv_empty() {
        let output = submit_recv(
            &TestExecutor { vocab_size: 16 },
            ForwardBatch::OneShot {
                token_groups: vec![],
                task: crate::engine::TaskKind::Classify,
            },
        )
        .unwrap();
        assert_eq!(output.logits.dims(), &[0, 16]);
    }

    #[test]
    fn dropped_sender_returns_error() {
        let (_, rx) = tokio::sync::oneshot::channel::<Result<ModelOutput, EngineError>>();
        let handle = ExecutionHandle::new(rx);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(handle.recv());
        assert!(result.is_err());
    }
}

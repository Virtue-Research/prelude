//! Device execution trait.
//!
//! `Executor` defines how forward passes are submitted to and collected from
//! a device. Device crates (prelude-cuda, prelude-cpu, ...) implement this trait
//! with device-specific optimizations:
//!
//! - **CudaExecutor**: async GPU queue + CUDA graph replay + double buffering
//! - **CpuExecutor**: synchronous `block_in_place` execution
//!
//! Core's scheduling loops (`engine/run/`) call `submit`/`collect` without knowing
//! which device is running. `submit()` is non-blocking on GPU (queues work and
//! returns immediately), so the scheduling loop naturally overlaps CPU preparation
//! with device execution — no separate batch/continuous variants per device.
//!
//! # Registration
//!
//! Device crates register their executor via `register_executor()` at startup,
//! with priority and probe for automatic backend selection:
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

use std::sync::OnceLock;

use crate::tensor::Tensor;

use super::EngineError;

// ── Execution handle ───────────────────────────────────────────────

/// Opaque handle returned by `Executor::submit()`.
///
/// Represents an in-flight batch on the device. Call `Executor::collect()`
/// to await completion and retrieve the output.
pub struct ExecutionHandle {
    inner: Box<dyn std::any::Any + Send>,
}

impl ExecutionHandle {
    /// Create a new handle wrapping device-specific state.
    pub fn new<T: Send + 'static>(inner: T) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    /// Downcast to the device-specific handle type.
    pub fn downcast<T: 'static>(self) -> Option<T> {
        self.inner.downcast::<T>().ok().map(|b| *b)
    }
}

// ── Model output ───────────────────────────────────────────────────

/// Output from a single forward pass.
///
/// Sampling and postprocessing happen in the scheduling loop (core),
/// not in the Executor.
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
}

// ── Forward batch ──────────────────────────────────────────────────

/// What the Executor should run. Built by the scheduling loop from
/// `ArSequenceState` + `SchedulerStep`, then passed to `submit()`.
pub enum ForwardBatch {
    /// Prefill: process full prompt sequences (generation).
    /// Contains prepared requests with token IDs, sampling params, etc.
    Prefill {
        items: Vec<super::PreparedGenerateRequest>,
    },
    /// Decode: generate one token per active sequence (Q=1).
    /// Each entry is (pending_token, position, block_table).
    Decode {
        tokens: Vec<u32>,
        positions: Vec<usize>,
        block_tables: Vec<Vec<u32>>,
        /// DeltaNet pool slots for hybrid models (None = non-hybrid).
        deltanet_slots: Option<Vec<u32>>,
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
    /// - **GPU**: queues work to a dedicated GPU thread, returns a pending handle.
    /// - **CPU**: runs `model.forward()` synchronously, returns a completed handle.
    fn submit(
        &self,
        batch: ForwardBatch,
    ) -> Result<ExecutionHandle, EngineError>;

    /// Await completion of a submitted forward pass and retrieve output.
    ///
    /// - **GPU**: blocks until the GPU thread signals completion.
    /// - **CPU**: returns immediately (work already done in `submit`).
    fn collect(
        &self,
        handle: ExecutionHandle,
    ) -> Result<ModelOutput, EngineError>;
}

// ── Registration ───────────────────────────────────────────────────

use std::sync::Mutex;
use crate::tensor::Device;

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
        tracing::info!("executor for {device}: {} (priority {})", b.name, b.priority);
        (b.create)(engine)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::SchedulerStep;

    /// Mock result stored in the handle.
    struct MockResult {
        logits: Tensor,
        item_seq_counts: Vec<usize>,
    }

    /// A test executor that returns a fixed logits tensor.
    struct TestExecutor {
        vocab_size: usize,
    }

    impl Executor for TestExecutor {
        fn submit(
            &self,
            batch: ForwardBatch,
        ) -> Result<ExecutionHandle, EngineError> {
            let (batch_size, item_seq_counts) = match &batch {
                ForwardBatch::Prefill { items } => (items.len(), vec![]),
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
            ).map_err(|e| EngineError::Internal(format!("{e}")))?;
            Ok(ExecutionHandle::new(MockResult { logits, item_seq_counts }))
        }

        fn collect(
            &self,
            handle: ExecutionHandle,
        ) -> Result<ModelOutput, EngineError> {
            let result = handle.downcast::<MockResult>()
                .ok_or_else(|| EngineError::Internal("wrong handle type".into()))?;
            Ok(ModelOutput { logits: result.logits, item_seq_counts: result.item_seq_counts, prefill_results: vec![] })
        }
    }

    #[test]
    fn submit_collect_prefill() {
        let executor = TestExecutor { vocab_size: 100 };
        let batch = ForwardBatch::Prefill { items: vec![] };

        let handle = executor.submit(batch).unwrap();
        let output = executor.collect(handle).unwrap();
        assert_eq!(output.logits.dims(), &[0, 100]);
    }

    #[test]
    fn submit_collect_decode() {
        let executor = TestExecutor { vocab_size: 50 };
        let batch = ForwardBatch::Decode {
            tokens: vec![1, 2, 3],
            positions: vec![10, 20, 30],
            block_tables: vec![vec![0, 1], vec![0, 2], vec![0, 3]],
            deltanet_slots: None,
        };

        let handle = executor.submit(batch).unwrap();
        let output = executor.collect(handle).unwrap();
        assert_eq!(output.logits.dims(), &[3, 50]);
    }

    #[test]
    fn submit_collect_oneshot() {
        let executor = TestExecutor { vocab_size: 8 };
        let batch = ForwardBatch::OneShot {
            token_groups: vec![
                vec![vec![1, 2, 3]],           // 1 sequence
                vec![vec![4, 5], vec![6, 7]],  // 2 sequences
            ],
            task: crate::engine::TaskKind::Classify,
        };

        let handle = executor.submit(batch).unwrap();
        let output = executor.collect(handle).unwrap();
        assert_eq!(output.logits.dims(), &[3, 8]); // total 3 sequences
        assert_eq!(output.item_seq_counts, vec![1, 2]);
    }

    #[test]
    fn collect_wrong_handle_type() {
        let executor = TestExecutor { vocab_size: 10 };
        let bad_handle = ExecutionHandle::new(42u32);
        let result = executor.collect(bad_handle);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("wrong handle type"));
    }

    #[test]
    fn handle_downcast_success() {
        let handle = ExecutionHandle::new(String::from("hello"));
        let value = handle.downcast::<String>();
        assert_eq!(value, Some(String::from("hello")));
    }

    #[test]
    fn handle_downcast_wrong_type() {
        let handle = ExecutionHandle::new(42u32);
        let value = handle.downcast::<String>();
        assert!(value.is_none());
    }

    #[test]
    fn submit_collect_decode_with_deltanet_slots() {
        let executor = TestExecutor { vocab_size: 32 };
        let batch = ForwardBatch::Decode {
            tokens: vec![10, 20],
            positions: vec![5, 8],
            block_tables: vec![vec![0], vec![1]],
            deltanet_slots: Some(vec![0, 1]),
        };

        let handle = executor.submit(batch).unwrap();
        let output = executor.collect(handle).unwrap();
        assert_eq!(output.logits.dims(), &[2, 32]);
        assert!(output.item_seq_counts.is_empty());
    }

    #[test]
    fn submit_collect_oneshot_embed() {
        let executor = TestExecutor { vocab_size: 64 };
        let batch = ForwardBatch::OneShot {
            token_groups: vec![
                vec![vec![1, 2, 3, 4]],
            ],
            task: crate::engine::TaskKind::Embed,
        };

        let handle = executor.submit(batch).unwrap();
        let output = executor.collect(handle).unwrap();
        assert_eq!(output.logits.dims(), &[1, 64]);
        assert_eq!(output.item_seq_counts, vec![1]);
    }

    #[test]
    fn submit_collect_oneshot_empty() {
        let executor = TestExecutor { vocab_size: 16 };
        let batch = ForwardBatch::OneShot {
            token_groups: vec![],
            task: crate::engine::TaskKind::Classify,
        };

        let handle = executor.submit(batch).unwrap();
        let output = executor.collect(handle).unwrap();
        assert_eq!(output.logits.dims(), &[0, 16]);
        assert!(output.item_seq_counts.is_empty());
    }
}

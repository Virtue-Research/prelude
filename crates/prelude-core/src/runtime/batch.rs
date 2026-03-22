//! Batch runtime.
//!
//! Double-buffered scheduler loop: prepares batch N+1 while GPU processes batch N.
//! All three task types (generation, classification, embedding) follow a unified
//! pipeline: adaptive wait → background tokenization → GPU submission → dispatch.
//!
//! Each task type implements the `BatchTask` trait, which defines tokenization,
//! GPU submission, postprocessing, and result delivery. `TaskPipeline<T>` drives
//! the lifecycle generically — no closures needed at call sites.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::task::JoinHandle;

use crate::engine::EngineError;
use crate::engine::{
    Engine, PreTokenizedClassifyItem, PreTokenizedEmbedItem, PreparedGenerateRequest,
};
use crate::scheduler::adaptive::AdaptiveSchedulerState;
use crate::runtime::SchedulerConfig;
use crate::engine::{
    RawClassifyOutput, RawEmbedOutput, classify_postprocess, embed_postprocess,
};
use crate::types::{ClassificationInputs, ClassifyResult, EmbedResult, GenerateResult};

#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
use crate::engine::{RawGenerateOutput, generate_postprocess};

use crate::scheduler::scheduled_engine::{
    GenerationRequestState,
    InFlightClassifyRequest, InFlightEmbedRequest, SchedulerMsg,
};
use super::gpu_queue::{submit_classify_batch, submit_embed_batch, submit_generate_batch, GpuQueueTx};

// ---------------------------------------------------------------------------
// PendingSlot — generic GPU in-flight state
// ---------------------------------------------------------------------------

/// A single GPU task in flight: inflight request handles + JoinHandle + metrics.
struct PendingSlot<I, R> {
    inflight: Vec<I>,
    handle: JoinHandle<Result<R, EngineError>>,
    batch_size: usize,
    submit_time: Instant,
}

/// Result of collecting a completed `PendingSlot`.
struct CompletedBatch<I, R> {
    inflight: Vec<I>,
    result: Result<R, EngineError>,
    batch_size: usize,
    gpu_time_ms: f64,
}

impl<I, R: Send + 'static> PendingSlot<I, R> {
    fn new(
        inflight: Vec<I>,
        handle: JoinHandle<Result<R, EngineError>>,
    ) -> Self {
        let batch_size = inflight.len();
        Self {
            inflight,
            handle,
            batch_size,
            submit_time: Instant::now(),
        }
    }

    fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    /// Non-blocking: takes the slot if the handle has finished.
    async fn try_take(slot: &mut Option<Self>) -> Option<CompletedBatch<I, R>> {
        if slot.as_ref().is_some_and(|s| s.is_finished()) {
            Some(Self::collect(slot.take().unwrap()).await)
        } else {
            None
        }
    }

    /// Blocking: awaits the handle and takes the slot.
    async fn take(slot: &mut Option<Self>) -> Option<CompletedBatch<I, R>> {
        let s = slot.take()?;
        Some(Self::collect(s).await)
    }

    async fn collect(s: Self) -> CompletedBatch<I, R> {
        let gpu_time_ms = s.submit_time.elapsed().as_secs_f64() * 1000.0;
        let result = s.handle.await
            .unwrap_or_else(|e| Err(EngineError::Internal(format!("GPU task panicked: {e}"))));
        CompletedBatch {
            inflight: s.inflight,
            result,
            batch_size: s.batch_size,
            gpu_time_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Generic dispatch helpers
// ---------------------------------------------------------------------------

/// Dispatches batch results to callers. `send_ok` delivers a successful result
/// to each inflight handle; `send_err` delivers an error.
fn dispatch_results<I, R>(
    inflight: Vec<I>,
    result: Result<Vec<R>, EngineError>,
    task_name: &str,
    mut send_ok: impl FnMut(I, R),
    mut send_err: impl FnMut(I, EngineError),
) {
    match result {
        Ok(results) if results.len() == inflight.len() => {
            for (item, result) in inflight.into_iter().zip(results) {
                send_ok(item, result);
            }
        }
        Ok(results) => {
            let msg = format!(
                "{task_name} batch output size mismatch, expected {}, got {}",
                inflight.len(),
                results.len()
            );
            for item in inflight {
                send_err(item, EngineError::Internal(msg.clone()));
            }
        }
        Err(e) => {
            for item in inflight {
                send_err(item, e.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BatchTask trait — defines per-task-type behavior
// ---------------------------------------------------------------------------

#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
type GenGpuResult = RawGenerateOutput;
#[cfg(not(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer")))]
type GenGpuResult = Vec<GenerateResult>;

/// Trait that captures all per-task-type differences. `TaskPipeline<T>` drives
/// the lifecycle generically — no closures needed at call sites.
///
/// Per-type wiring:
/// ```text
/// Generation:  tokenize → prepare_generate_request
///              submit   → submit_generate_batch
///              postproc → generate_postprocess (flash-attn-v3) or passthrough
///
/// Classify:    tokenize → tokenize_batch → PreTokenizedClassifyItem
///              submit   → submit_classify_batch
///              postproc → classify_postprocess
///
/// Embed:       tokenize → tokenize_batch → PreTokenizedEmbedItem
///              submit   → submit_embed_batch
///              postproc → embed_postprocess
/// ```
trait BatchTask: SeqCount + Send + Sync + 'static {
    /// Pre-tokenized representation ready for GPU submission.
    type Tokenized: Send + 'static;
    /// Raw GPU output before postprocessing.
    type RawGpuResult: Send + 'static;
    /// Final result type sent back to the caller.
    type Result: Send;

    const TASK_NAME: &'static str;

    fn tokenize_one(engine: &Engine, item: &Self, idx: usize) -> Result<Self::Tokenized, EngineError>;
    fn submit(gpu_tx: &GpuQueueTx, batch: Vec<Self::Tokenized>) -> JoinHandle<Result<Self::RawGpuResult, EngineError>>;
    fn postprocess(raw: Self::RawGpuResult, engine: &Engine) -> Result<Vec<Self::Result>, EngineError>;
    fn send_ok(item: Self, result: Self::Result);
    fn fail(item: Self, err: EngineError);
}

impl BatchTask for GenerationRequestState {
    type Tokenized = PreparedGenerateRequest;
    type RawGpuResult = GenGpuResult;
    type Result = GenerateResult;
    const TASK_NAME: &'static str = "generation";

    fn tokenize_one(engine: &Engine, item: &Self, idx: usize) -> Result<PreparedGenerateRequest, EngineError> {
        engine.prepare_generate_request(item.request(), idx)
    }
    fn submit(gpu_tx: &GpuQueueTx, batch: Vec<PreparedGenerateRequest>) -> JoinHandle<Result<GenGpuResult, EngineError>> {
        submit_generate_batch(gpu_tx, batch)
    }
    fn postprocess(raw: GenGpuResult, engine: &Engine) -> Result<Vec<GenerateResult>, EngineError> {
        #[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]
        return generate_postprocess(raw, &engine.tokenizer, &engine.eos_token_ids);
        #[cfg(not(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer")))]
        { let _ = engine; Ok(raw) }
    }
    fn send_ok(item: Self, result: GenerateResult) { item.finish(Ok(result)); }
    fn fail(item: Self, err: EngineError) { item.fail(err); }
}

impl BatchTask for InFlightClassifyRequest {
    type Tokenized = PreTokenizedClassifyItem;
    type RawGpuResult = RawClassifyOutput;
    type Result = ClassifyResult;
    const TASK_NAME: &'static str = "classify";

    fn tokenize_one(engine: &Engine, item: &Self, idx: usize) -> Result<PreTokenizedClassifyItem, EngineError> {
        let (token_ids, total_tokens) = engine.tokenize_batch(&item.request.inputs)?;
        Ok(PreTokenizedClassifyItem { request_idx: idx, request: item.request.clone(), token_ids, total_tokens })
    }
    fn submit(gpu_tx: &GpuQueueTx, batch: Vec<PreTokenizedClassifyItem>) -> JoinHandle<Result<RawClassifyOutput, EngineError>> {
        submit_classify_batch(gpu_tx, batch)
    }
    fn postprocess(raw: RawClassifyOutput, _engine: &Engine) -> Result<Vec<ClassifyResult>, EngineError> {
        classify_postprocess(raw)
    }
    fn send_ok(item: Self, result: ClassifyResult) { let _ = item.response.send(Ok(result)); }
    fn fail(item: Self, err: EngineError) { let _ = item.response.send(Err(err)); }
}

impl BatchTask for InFlightEmbedRequest {
    type Tokenized = PreTokenizedEmbedItem;
    type RawGpuResult = RawEmbedOutput;
    type Result = EmbedResult;
    const TASK_NAME: &'static str = "embed";

    fn tokenize_one(engine: &Engine, item: &Self, idx: usize) -> Result<PreTokenizedEmbedItem, EngineError> {
        let (token_ids, total_tokens) = engine.tokenize_batch(&item.request.inputs)?;
        Ok(PreTokenizedEmbedItem { request_idx: idx, request: item.request.clone(), token_ids, total_tokens })
    }
    fn submit(gpu_tx: &GpuQueueTx, batch: Vec<PreTokenizedEmbedItem>) -> JoinHandle<Result<RawEmbedOutput, EngineError>> {
        submit_embed_batch(gpu_tx, batch)
    }
    fn postprocess(raw: RawEmbedOutput, _engine: &Engine) -> Result<Vec<EmbedResult>, EngineError> {
        embed_postprocess(raw)
    }
    fn send_ok(item: Self, result: EmbedResult) { let _ = item.response.send(Ok(result)); }
    fn fail(item: Self, err: EngineError) { let _ = item.response.send(Err(err)); }
}

// ---------------------------------------------------------------------------
// TaskPipeline — unified per-task-type pipeline state
// ---------------------------------------------------------------------------

/// A tokenized batch ready for GPU submission.
struct ReadyBatch<T: BatchTask> {
    requests: Vec<T>,
    tokenized: Vec<T::Tokenized>,
}

/// Unified pipeline state for a single task type.
///
/// State flow:
/// ```text
/// queued ──(start_tokenize)──→ tokenize_task
/// tokenize_task ──(collect_tokenized)──→ ready_batches
/// ready_batches ──(submit_ready)──→ gpu_task
/// gpu_task ──(collect_gpu)──→ dispatch to caller
/// ```
struct TaskPipeline<T: BatchTask> {
    /// Incoming requests waiting to be tokenized.
    queued: VecDeque<T>,
    /// Tokenized batches waiting for a free GPU slot.
    ready_batches: VecDeque<ReadyBatch<T>>,
    /// Currently running GPU task (at most one per pipeline).
    gpu_task: Option<PendingSlot<T, T::RawGpuResult>>,
    /// Background tokenization job.
    tokenize_task: Option<JoinHandle<Option<ReadyBatch<T>>>>,
    /// Number of items currently in the tokenize task (for adaptive wait visibility).
    tokenize_count: usize,
    /// Adaptive batch sizing state.
    adaptive: AdaptiveSchedulerState,
}

impl<T: BatchTask> TaskPipeline<T> {
    fn new(adaptive: AdaptiveSchedulerState) -> Self {
        Self {
            queued: VecDeque::new(),
            ready_batches: VecDeque::new(),
            gpu_task: None,
            tokenize_task: None,
            tokenize_count: 0,
            adaptive,
        }
    }

    fn is_idle(&self) -> bool {
        self.queued.is_empty()
            && self.ready_batches.is_empty()
            && self.gpu_task.is_none()
            && self.tokenize_task.is_none()
    }

    fn gpu_busy(&self) -> bool { self.gpu_task.is_some() }
    fn tokenize_busy(&self) -> bool { self.tokenize_task.is_some() }

    /// Total pending items: queued + currently being tokenized.
    fn pending_count(&self) -> usize { self.queued.len() + self.tokenize_count }

    /// Drain items from queued (up to `max_batch_size` sequences) and start
    /// background tokenization via `spawn_blocking`.
    fn start_tokenize(&mut self, max_batch_size: usize, engine: &Arc<Engine>) {
        if self.tokenize_task.is_some() || self.queued.is_empty() {
            return;
        }
        let mut items = Vec::new();
        let mut total_seqs = 0;
        while let Some(item) = self.queued.front() {
            let n = item.seq_count();
            if total_seqs > 0 && total_seqs + n > max_batch_size { break; }
            total_seqs += n;
            items.push(self.queued.pop_front().unwrap());
        }
        if items.is_empty() { return; }
        self.tokenize_count = items.len();
        let e = Arc::clone(engine);
        self.tokenize_task = Some(tokio::task::spawn_blocking(move || {
            tokenize_batch_sync(items, |item, idx| T::tokenize_one(&e, item, idx), T::fail)
        }));
    }

    /// Non-blocking: move completed tokenization result to ready_batches.
    async fn collect_tokenized(&mut self) {
        if self.tokenize_task.as_ref().is_some_and(|h| h.is_finished()) {
            if let Some(h) = self.tokenize_task.take() {
                self.tokenize_count = 0;
                match h.await {
                    Ok(Some(batch)) => self.ready_batches.push_back(batch),
                    Ok(None) => {}
                    Err(e) => tracing::error!(error = %e, "tokenization task panicked"),
                }
            }
        }
    }

    /// Submit ready batches to the GPU queue, merging multiple tokenized
    /// batches into one GPU call (up to `max_batch_size`). Returns batch size
    /// if submitted.
    ///
    /// Defers submission when more items are pending (queued + in-flight
    /// tokenize) than are currently ready.  Rationale:
    ///
    ///   Submit now  → T_gpu(r) + T_gpu(p)      = 2C + (r+p)α
    ///   Defer+merge → T_tok(p) + T_gpu(r+p)    =  C + (r+p)α + T_tok
    ///
    /// Because GPU kernel launch overhead C ≈ 3–5 ms >> T_tok ≈ 1–2 ms,
    /// deferring saves one C per merge.  The loop naturally retries within
    /// microseconds, so the extra latency is negligible.
    fn submit_ready(&mut self, gpu_tx: &GpuQueueTx, max_batch_size: usize) -> Option<usize> {
        if self.gpu_task.is_some() { return None; }
        if self.ready_batches.is_empty() { return None; }

        // Defer: pending items (queued + tokenizing) outnumber ready items.
        let pending = self.pending_count();
        if pending > 0 {
            let ready_total: usize = self.ready_batches.iter().map(|b| b.requests.len()).sum();
            if pending > ready_total {
                return None;
            }
        }

        let first = self.ready_batches.pop_front().unwrap();
        let mut requests = first.requests;
        let mut tokenized = first.tokenized;

        // Merge subsequent ready batches up to max_batch_size.
        while let Some(next) = self.ready_batches.front() {
            if requests.len() + next.requests.len() > max_batch_size {
                break;
            }
            let next = self.ready_batches.pop_front().unwrap();
            requests.extend(next.requests);
            tokenized.extend(next.tokenized);
        }

        let batch_sz = requests.len();
        let handle = T::submit(gpu_tx, tokenized);
        self.gpu_task = Some(PendingSlot::new(requests, handle));
        Some(batch_sz)
    }

    /// Collect finished GPU work and dispatch results to callers.
    ///
    /// Strategy: if a ready batch is already waiting to be submitted, we must
    /// block-wait for the current GPU task to finish so we can free the slot.
    /// Otherwise, just do a non-blocking check to avoid stalling the loop.
    async fn collect_gpu(&mut self, engine: &Engine) {
        let dispatch = |inflight: Vec<T>, raw_result: Result<T::RawGpuResult, EngineError>| {
            let result = raw_result.and_then(|raw| T::postprocess(raw, engine));
            dispatch_results(inflight, result, T::TASK_NAME, T::send_ok, T::fail);
        };
        let needs_gpu_slot = !self.ready_batches.is_empty();
        let completed = if needs_gpu_slot {
            PendingSlot::take(&mut self.gpu_task).await     // block: must free slot
        } else {
            PendingSlot::try_take(&mut self.gpu_task).await  // non-blocking poll
        };
        if let Some(c) = completed {
            self.adaptive.record_gpu_time(c.batch_size, c.gpu_time_ms);
            dispatch(c.inflight, c.result);
        }
    }

    /// Shutdown: finish in-flight work, then fail all remaining requests.
    async fn shutdown(&mut self, engine: &Engine) {
        if let Some(h) = self.tokenize_task.take() { self.tokenize_count = 0; let _ = h.await; }
        if let Some(slot) = self.gpu_task.take() {
            let c = PendingSlot::collect(slot).await;
            let result = c.result.and_then(|raw| T::postprocess(raw, engine));
            dispatch_results(c.inflight, result, T::TASK_NAME, T::send_ok, T::fail);
        }
        let shutdown_err = || EngineError::Unavailable("scheduler loop stopped".into());
        for batch in self.ready_batches.drain(..) {
            for item in batch.requests { T::fail(item, shutdown_err()); }
        }
        for item in self.queued.drain(..) { T::fail(item, shutdown_err()); }
    }
}

// ---------------------------------------------------------------------------
// Scheduler queue state
// ---------------------------------------------------------------------------

struct BatchRuntimeQueues {
    generate: TaskPipeline<GenerationRequestState>,
    classify: TaskPipeline<InFlightClassifyRequest>,
    embed: TaskPipeline<InFlightEmbedRequest>,
}

impl BatchRuntimeQueues {
    fn dispatch_and_track(&mut self, msg: SchedulerMsg) {
        let now = Instant::now();
        match msg {
            SchedulerMsg::NewRequest(inflight) => {
                self.generate.queued.push_back(inflight);
                self.generate.adaptive.record_arrivals(1, now);
            }
            SchedulerMsg::NewClassifyRequest(inflight) => {
                self.classify.queued.push_back(inflight);
                self.classify.adaptive.record_arrivals(1, now);
            }
            SchedulerMsg::NewEmbedRequest(inflight) => {
                self.embed.queued.push_back(inflight);
                self.embed.adaptive.record_arrivals(1, now);
            }
            // Batch runtime only handles prefill-only (max_new=1) generation;
            // abort is a no-op since there's no multi-step decode to cancel.
            SchedulerMsg::Abort(_) => {}
        }
    }

    /// Start background tokenization for all pipelines that have queued work.
    fn start_all_tokenize(&mut self, max_batch_size: usize, engine: &Arc<Engine>) {
        self.generate.start_tokenize(max_batch_size, engine);
        self.classify.start_tokenize(max_batch_size, engine);
        self.embed.start_tokenize(max_batch_size, engine);
    }

    /// Reap completed tokenization tasks across all pipelines.
    async fn reap_all_tokenize(&mut self) {
        self.generate.collect_tokenized().await;
        self.classify.collect_tokenized().await;
        self.embed.collect_tokenized().await;
    }

    /// Adaptive wait: if any pipeline has queued work below its optimal batch
    /// size, wait for more requests to arrive. Uses the longest recommended
    /// wait budget across all eligible pipelines. During the wait, incoming
    /// messages are dispatched to all pipelines.
    async fn adaptive_wait_if_needed(
        &mut self,
        rx: &mut mpsc::UnboundedReceiver<SchedulerMsg>,
        rx_open: &mut bool,
    ) {
        if !*rx_open { return; }

        // Find the max recommended wait across pipelines with queued work
        let mut wait_budget = Duration::ZERO;
        for (pending, adaptive) in [
            (self.generate.pending_count(), &self.generate.adaptive),
            (self.classify.pending_count(), &self.classify.adaptive),
            (self.embed.pending_count(), &self.embed.adaptive),
        ] {
            if pending > 0 {
                let (target, budget) = adaptive.compute_optimal_batch_and_wait(pending);
                if pending < target && budget > wait_budget {
                    wait_budget = budget;
                }
            }
        }

        if wait_budget.is_zero() { return; }

        // Wait, dispatching incoming messages to all pipelines
        let wait_start = Instant::now();
        while *rx_open {
            let remaining = wait_budget.checked_sub(wait_start.elapsed());
            match remaining {
                Some(rem) if !rem.is_zero() => match tokio::time::timeout(rem, rx.recv()).await {
                    Ok(Some(msg)) => self.dispatch_and_track(msg),
                    Ok(None) => { *rx_open = false; break; }
                    Err(_) => break,
                },
                _ => break,
            }
        }
    }

    fn all_idle(&self) -> bool {
        self.generate.is_idle() && self.classify.is_idle() && self.embed.is_idle()
    }

    fn all_queues_empty(&self) -> bool {
        self.generate.queued.is_empty() && self.classify.queued.is_empty() && self.embed.queued.is_empty()
    }

    fn any_gpu_busy(&self) -> bool {
        self.generate.gpu_busy() || self.classify.gpu_busy() || self.embed.gpu_busy()
    }

    fn any_tokenize_busy(&self) -> bool {
        self.generate.tokenize_busy() || self.classify.tokenize_busy() || self.embed.tokenize_busy()
    }

    async fn collect_gpu_all(&mut self, engine: &Engine) {
        self.generate.collect_gpu(engine).await;
        self.classify.collect_gpu(engine).await;
        self.embed.collect_gpu(engine).await;
    }

    fn submit_all(&mut self, gpu_tx: &GpuQueueTx, max_batch_size: usize) {
        if let Some(sz) = self.generate.submit_ready(gpu_tx, max_batch_size) {
            tracing::info!(batch_size = sz, queue_remaining = self.generate.queued.len(), "gen batch submitted");
        }
        if let Some(sz) = self.classify.submit_ready(gpu_tx, max_batch_size) {
            tracing::debug!(batch_size = sz, queue_remaining = self.classify.queued.len(), "classify batch submitted");
        }
        if let Some(sz) = self.embed.submit_ready(gpu_tx, max_batch_size) {
            tracing::debug!(batch_size = sz, queue_remaining = self.embed.queued.len(), "embed batch submitted");
        }
    }

    async fn shutdown_all(&mut self, engine: &Engine) {
        self.generate.shutdown(engine).await;
        self.classify.shutdown(engine).await;
        self.embed.shutdown(engine).await;
    }
}

// ---------------------------------------------------------------------------
// Main scheduler loop
// ---------------------------------------------------------------------------

/// Batch runtime main loop.
///
/// Double-buffered scheduler loop: prepares batch N+1 while GPU processes batch N.
/// All three task types (gen, classify, embed) use identical `TaskPipeline` state.
///
/// One loop iteration does:
/// 1. If fully idle, block until the first request arrives.
/// 2. Drain any immediately available requests from the channel.
/// 3. Kick off background tokenization for queued work.
/// 4. Optionally wait a little longer to build a better batch.
/// 5. Kick off tokenization for requests that arrived during the wait.
/// 6. Move finished tokenization jobs into `ready_batches`.
/// 7. Collect finished GPU work (block-wait if a ready batch needs the slot).
/// 8. Submit one ready batch per pipeline if its GPU slot is free.
/// 9. If GPU is busy, briefly yield or wait for more arrivals.
pub(crate) async fn batch_runtime_loop(
    engine: Arc<Engine>,
    config: SchedulerConfig,
    mut rx: mpsc::UnboundedReceiver<SchedulerMsg>,
    gpu_tx: GpuQueueTx,
) {
    let max_batch_size = config.max_batch_size.max(1);
    let max_batch_wait = Duration::from_millis(config.max_batch_wait_ms.max(1));
    let mut rx_open = true;

    let adaptive_config = &engine.engine_config().adaptive;
    let new_adaptive = || AdaptiveSchedulerState::new(
        max_batch_size,
        config.max_batch_wait_ms.max(1) as f64,
        adaptive_config,
    );
    let mut q = BatchRuntimeQueues {
        generate: TaskPipeline::new(new_adaptive()),
        classify: TaskPipeline::new(new_adaptive()),
        embed: TaskPipeline::new(new_adaptive()),
    };

    loop {
        // ── Phase A: Wait for first request if nothing pending ──
        if q.all_idle() {
            match rx.recv().await {
                Some(msg) => q.dispatch_and_track(msg),
                None => break,
            }
        }

        // ── Phase B: Drain channel messages (non-blocking) ──
        loop {
            match rx.try_recv() {
                Ok(msg) => q.dispatch_and_track(msg),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => { rx_open = false; break; }
            }
        }

        // ── Phase C: Start tokenization, then adaptive wait (overlapped) ──
        q.start_all_tokenize(max_batch_size, &engine);
        q.adaptive_wait_if_needed(&mut rx, &mut rx_open).await;
        q.start_all_tokenize(max_batch_size, &engine);
        q.reap_all_tokenize().await;

        // ── Phase D: Collect GPU results + submit ready batches ──
        q.collect_gpu_all(&engine).await;
        q.submit_all(&gpu_tx, max_batch_size);

        // ── Phase G: Brief yield if GPU is busy to allow more requests to arrive ──
        if q.any_gpu_busy() && rx_open {
            if q.all_queues_empty() {
                match tokio::time::timeout(max_batch_wait, rx.recv()).await {
                    Ok(Some(msg)) => q.dispatch_and_track(msg),
                    Ok(None) => { rx_open = false; }
                    Err(_) => {}
                }
            } else if q.any_tokenize_busy() {
                tokio::task::yield_now().await;
            }
        }

        // Check exit condition
        if !rx_open && q.all_idle() {
            break;
        }
    }

    // ── Shutdown drain ──
    q.shutdown_all(&engine).await;

    let shutdown_err = || EngineError::Unavailable("scheduler loop stopped".into());
    while let Ok(msg) = rx.try_recv() {
        match msg {
            SchedulerMsg::NewRequest(item) => GenerationRequestState::fail(item, shutdown_err()),
            SchedulerMsg::NewClassifyRequest(item) => InFlightClassifyRequest::fail(item, shutdown_err()),
            SchedulerMsg::NewEmbedRequest(item) => InFlightEmbedRequest::fail(item, shutdown_err()),
            SchedulerMsg::Abort(_) => {}
        }
    }

    tracing::info!("batch runtime loop exited");
}

// ---------------------------------------------------------------------------
// Tokenization helpers
// ---------------------------------------------------------------------------

/// Sequence count for batch sizing. All task types use this uniformly:
/// gen = 1 sequence per request (future: multi-prompt batch completions),
/// classify/embed = number of input texts/token_id arrays.
trait SeqCount {
    fn seq_count(&self) -> usize;
}
impl SeqCount for GenerationRequestState {
    fn seq_count(&self) -> usize { 1 }
}
impl SeqCount for InFlightClassifyRequest {
    fn seq_count(&self) -> usize {
        match &self.request.inputs {
            ClassificationInputs::Texts(v) => v.len(),
            ClassificationInputs::TokenIds(v) => v.len(),
        }
    }
}
impl SeqCount for InFlightEmbedRequest {
    fn seq_count(&self) -> usize {
        match &self.request.inputs {
            ClassificationInputs::Texts(v) => v.len(),
            ClassificationInputs::TokenIds(v) => v.len(),
        }
    }
}

/// Tokenize a batch of requests in parallel using rayon.
///
/// `tokenize_one` maps each `(item, index)` to a tokenized output or error.
/// `fail_one` handles items that fail tokenization.
fn tokenize_batch_sync<T: BatchTask>(
    items: Vec<T>,
    tokenize_one: impl Fn(&T, usize) -> Result<T::Tokenized, EngineError> + Sync,
    fail_one: impl Fn(T, EngineError),
) -> Option<ReadyBatch<T>> {
    let results: Vec<_> = items
        .par_iter()
        .enumerate()
        .map(|(idx, item)| tokenize_one(item, idx))
        .collect();

    let mut requests = Vec::with_capacity(items.len());
    let mut tokenized = Vec::with_capacity(items.len());

    for (item, result) in items.into_iter().zip(results) {
        match result {
            Ok(t) => { requests.push(item); tokenized.push(t); }
            Err(e) => fail_one(item, e),
        }
    }

    if requests.is_empty() { None } else { Some(ReadyBatch { requests, tokenized }) }
}

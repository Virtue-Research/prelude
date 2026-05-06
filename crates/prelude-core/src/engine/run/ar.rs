//! Unified AR (autoregressive) scheduling loop.
//!
//! Handles both one-shot (batch) and multi-token (continuous batching) generation
//! in a single loop. The Scheduler decides what to prefill/decode each step;
//! the Executor handles device-specific execution via submit/collect.
//!
//! This replaces the previous 4 separate loops:
//! - `runtime/gpu_batch.rs` + `runtime/cpu_batch.rs` (one-shot generation)
//! - `runtime/gpu_continuous.rs` + `runtime/cpu_continuous.rs` (streaming decode)
//!
//! # Architecture
//!
//! ```text
//! ar_loop
//!   ├── drain requests from channel → scheduler.add_request()
//!   ├── scheduler.schedule_step() → SchedulerStep { prefill_ids, decode_ids }
//!   ├── executor.submit(step) → ExecutionHandle     ← device-specific
//!   ├── executor.collect(handle) → ModelOutput       ← device-specific
//!   ├── sample tokens (argmax / logits_processor)   ← device-agnostic
//!   ├── check stop conditions (eos, stop tokens/strings, max_tokens)
//!   ├── stream token deltas to callers
//!   └── scheduler.update(finished sequences)
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::cache::deltanet_pool::DeltaNetPrefixState;
use crate::engine::executor::{Executor, ForwardBatch, ModelOutput};
use crate::engine::{
    DecodeMetrics, Engine, EngineError, PreparedGenerateRequest, TokenLogprobInfo, Usage,
};
use crate::scheduler::{
    FinishReason, SamplingParams, Scheduler, SchedulerConfig, SchedulerStep, SeqFinishReason,
    Sequence,
};
use crate::types::{GenerateRequest, GenerateResult, StreamEvent};

// ── Response channel ───────────────────────────────────────────────

/// How generation output is delivered back to the caller.
pub enum ResponseChannel {
    /// Non-streaming: send a complete GenerateResult when done.
    Complete(tokio::sync::oneshot::Sender<Result<GenerateResult, EngineError>>),
    /// Streaming: tokens are sent per-token through this channel.
    Stream(mpsc::UnboundedSender<StreamEvent>),
}

// ── Per-request runtime state ──────────────────────────────────────

/// Per-request state tracked across decode steps.
///
/// Mirrors `RuntimeSequenceState` from the legacy gpu_continuous.rs but
/// is device-agnostic — no GPU queue references.
/// Per-request runtime state for response delivery and sampling.
/// Scheduling-relevant state (block_table, kv_computed_len) lives on Sequence
/// in the Scheduler. This struct only holds response/streaming/sampling concerns.
pub struct ArSequenceState {
    pub request_id: String,
    pub prepared: Option<PreparedGenerateRequest>,
    pub response: ResponseChannel,
    pub gen_start: Instant,
    pub prefill_ms: f32,
    pub started_sent: bool,
    pub sent_text_len: usize,
    pub pending_token: Option<u32>,
    pub output_tokens: Vec<u32>,
    pub token_logprobs: Vec<TokenLogprobInfo>,
    pub prompt_token_logprobs: Option<Vec<TokenLogprobInfo>>,
    /// Cached from PreparedGenerateRequest so it survives prepared.take() during prefill.
    pub max_new_tokens: usize,
}

impl ArSequenceState {
    fn max_new(&self) -> usize {
        self.max_new_tokens
    }

    fn is_greedy(&self) -> bool {
        self.prepared.as_ref().map(|p| p.is_greedy).unwrap_or(true)
    }

    fn ensure_started(&mut self) {
        if self.started_sent {
            return;
        }
        if let ResponseChannel::Stream(tx) = &self.response {
            let _ = tx.send(StreamEvent::Started);
        }
        self.started_sent = true;
    }

    fn emit_text_delta(
        &mut self,
        tokenizer: &fastokens::Tokenizer,
        logprobs: Option<TokenLogprobInfo>,
    ) {
        let ResponseChannel::Stream(tx) = &self.response else {
            return;
        };
        let text = tokenizer
            .decode(&self.output_tokens, true)
            .unwrap_or_default();
        if text.len() > self.sent_text_len {
            let _ = tx.send(StreamEvent::Token {
                text: text[self.sent_text_len..].to_string(),
                logprobs,
            });
            self.sent_text_len = text.len();
        }
    }

    fn current_text(&self, tokenizer: &fastokens::Tokenizer) -> String {
        tokenizer
            .decode(&self.output_tokens, true)
            .unwrap_or_default()
    }
}

// ── Messages ───────────────────────────────────────────────────────

/// Messages from the engine API layer to the AR loop.
pub enum ArMessage {
    /// New generation request. Preparation/tokenization is run by the AR loop's
    /// prepare worker path so initial batch wait can overlap that CPU work.
    RawRequest {
        request: GenerateRequest,
        response: ResponseChannel,
    },
    /// New generation request (pre-tokenized, pre-prepared).
    NewRequest {
        prepared: PreparedGenerateRequest,
        response: ResponseChannel,
    },
    /// Cancel an in-flight request.
    Abort(String),
}

struct PreparedArMessage {
    request_id: String,
    response: ResponseChannel,
    result: Result<PreparedGenerateRequest, EngineError>,
}

// ── Main loop ──────────────────────────────────────────────────────

/// Run the unified AR scheduling loop.
///
/// Handles both batch and continuous batching in a single loop.
/// The Scheduler decides what goes into each step; the Executor
/// handles device-specific execution via submit/collect.
pub async fn ar_loop(
    engine: Arc<Engine>,
    executor: &dyn Executor,
    scheduler_config: SchedulerConfig,
    mut rx: mpsc::UnboundedReceiver<ArMessage>,
) {
    let mut scheduler = Scheduler::new(scheduler_config);
    if let Some(bm_arc) = engine.cache.block_manager_arc() {
        scheduler.set_block_manager(bm_arc);
    }
    let mut states: HashMap<String, ArSequenceState> = HashMap::new();
    let mut rx_open = true;
    let (prepare_tx, mut prepare_rx) = mpsc::unbounded_channel::<PreparedArMessage>();
    let mut pending_prepares = 0usize;
    let mut initial_wait_consumed = false;

    loop {
        let deltanet_pool_ref = engine.cache.deltanet_pool.as_ref();
        // ── Phase 1: Wait for at least one request if idle ─────────
        if !scheduler.has_work() {
            loop {
                if scheduler.has_work() || (!rx_open && pending_prepares == 0) {
                    break;
                }
                tokio::select! {
                    biased;
                    prepared = prepare_rx.recv(), if pending_prepares > 0 => {
                        if let Some(msg) = prepared {
                            handle_prepared_message(
                                &engine,
                                msg,
                                &mut scheduler,
                                &mut states,
                                deltanet_pool_ref,
                                &mut pending_prepares,
                            );
                        }
                    }
                    msg = rx.recv(), if rx_open => {
                        match msg {
                            Some(msg) => handle_message(
                                Some(&engine),
                                msg,
                                &mut scheduler,
                                &mut states,
                                deltanet_pool_ref,
                                Some(&prepare_tx),
                                &mut pending_prepares,
                            ),
                            None => {
                                rx_open = false;
                            }
                        }
                    }
                }
                if scheduler.has_work()
                    || (pending_prepares > 0
                        && !initial_wait_consumed
                        && scheduler.config().max_batch_wait_ms > 0)
                {
                    break;
                }
            }
        }

        // ── Phase 2: Drain all pending messages (non-blocking) ─────
        drain_ready_messages(
            Some(&engine),
            &mut rx,
            &mut rx_open,
            &mut scheduler,
            &mut states,
            deltanet_pool_ref,
            Some(&prepare_tx),
            &mut pending_prepares,
        );
        drain_ready_prepared_messages(
            &engine,
            &mut prepare_rx,
            &mut scheduler,
            &mut states,
            deltanet_pool_ref,
            &mut pending_prepares,
        );

        // When the server is idle, give concurrent HTTP handlers a short
        // window to finish tokenization and enqueue peers. Without this,
        // closed-loop prefill benchmarks often run the first request alone
        // and only batch the trailing requests.
        if !initial_wait_consumed
            && wait_for_initial_prefill_batch(
                &engine,
                &mut rx,
                &mut rx_open,
                &mut prepare_rx,
                &prepare_tx,
                &mut pending_prepares,
                &mut scheduler,
                &mut states,
                deltanet_pool_ref,
            )
            .await
        {
            initial_wait_consumed = true;
        }

        refresh_waiting_prefix_cache(&engine, &mut scheduler);

        // ── Phase 3: Schedule next step ────────────────────────────
        // schedule_step() syncs block availability internally.
        let Some(mut step) = scheduler.schedule_step() else {
            if !rx_open && pending_prepares == 0 && !scheduler.has_work() {
                break;
            }
            tokio::task::yield_now().await;
            continue;
        };
        initial_wait_consumed = false;

        // ── Phase 4: Build batch + forward + process output ──────────
        // Pure decode → ForwardBatch::Decode (CUDA graph eligible).
        // Anything with prefill → ForwardBatch::Mixed (single forward pass).
        // `build_step_batch` may shrink `step` in place when the KV
        // block pool is exhausted (see the `retain`/rollback paths
        // inside) so process_step_output, fail_step, etc. all see a
        // step that exactly matches what the executor actually ran.
        let batch = build_step_batch(&engine, &mut scheduler, &mut states, &mut step);
        let handle = match executor.submit(batch) {
            Ok(h) => h,
            Err(error) => {
                fail_step(&engine, &mut scheduler, &mut states, &step, error);
                continue;
            }
        };
        let output = match handle.recv().await {
            Ok(o) => o,
            Err(error) => {
                fail_step(&engine, &mut scheduler, &mut states, &step, error);
                continue;
            }
        };
        process_step_output(&engine, &mut scheduler, &mut states, &step, &output);

        if !rx_open && pending_prepares == 0 && !scheduler.has_work() {
            break;
        }
    }

    // Shutdown: fail remaining requests
    let deltanet_pool_ref = engine.cache.deltanet_pool.as_ref();
    for (id, state) in states.drain() {
        release_resources(deltanet_pool_ref, &mut scheduler, &id);
        fail_state(state, EngineError::Unavailable("AR loop stopped".into()));
    }
    tracing::info!("AR loop exited");
}

// ── Message handling ───────────────────────────────────────────────

fn handle_message(
    engine: Option<&Arc<Engine>>,
    msg: ArMessage,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    deltanet_pool: Option<&std::sync::Mutex<crate::cache::deltanet_pool::DeltaNetPool>>,
    prepare_tx: Option<&mpsc::UnboundedSender<PreparedArMessage>>,
    pending_prepares: &mut usize,
) {
    match msg {
        ArMessage::RawRequest { request, response } => {
            if let (Some(engine), Some(prepare_tx)) = (engine, prepare_tx) {
                spawn_prepare_request(engine, prepare_tx, request, response);
                *pending_prepares += 1;
            } else {
                let message = "raw AR request cannot be prepared without an engine";
                match response {
                    ResponseChannel::Complete(tx) => {
                        let _ = tx.send(Err(EngineError::Internal(message.into())));
                    }
                    ResponseChannel::Stream(tx) => {
                        let _ = tx.send(StreamEvent::Error {
                            message: message.into(),
                        });
                    }
                }
            }
        }
        ArMessage::NewRequest { prepared, response } => admit_prepared_request(
            engine.map(|e| e.as_ref()),
            prepared,
            response,
            scheduler,
            states,
            deltanet_pool,
        ),
        ArMessage::Abort(request_id) => {
            // Free the DeltaNet pool slot + any allocated KV blocks before
            // dropping the sequence: skipping this leaks a slot per cancelled
            // request, which exhausts the (small) DeltaNet pool under heavy
            // cancel-and-retry load and starves new admissions.
            release_resources(deltanet_pool, scheduler, &request_id);
            let _ = scheduler.abort_request(&request_id);
            states.remove(&request_id);
        }
    }
}

fn spawn_prepare_request(
    engine: &Arc<Engine>,
    prepare_tx: &mpsc::UnboundedSender<PreparedArMessage>,
    request: GenerateRequest,
    response: ResponseChannel,
) {
    let engine = Arc::clone(engine);
    let prepare_tx = prepare_tx.clone();
    let request_id = request.request_id.clone();
    tokio::task::spawn_blocking(move || {
        let result = engine.prepare_generate_request(&request, 0);
        let _ = prepare_tx.send(PreparedArMessage {
            request_id,
            response,
            result,
        });
    });
}

fn handle_prepared_message(
    engine: &Engine,
    msg: PreparedArMessage,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    deltanet_pool: Option<&std::sync::Mutex<crate::cache::deltanet_pool::DeltaNetPool>>,
    pending_prepares: &mut usize,
) {
    *pending_prepares = pending_prepares.saturating_sub(1);
    match msg.result {
        Ok(prepared) => admit_prepared_request(
            Some(engine),
            prepared,
            msg.response,
            scheduler,
            states,
            deltanet_pool,
        ),
        Err(error) => {
            tracing::warn!(request_id = %msg.request_id, error = %error, "generation prepare failed");
            match msg.response {
                ResponseChannel::Complete(tx) => {
                    let _ = tx.send(Err(error));
                }
                ResponseChannel::Stream(tx) => {
                    let _ = tx.send(StreamEvent::Error {
                        message: error.to_string(),
                    });
                }
            }
        }
    }
}

fn admit_prepared_request(
    engine: Option<&Engine>,
    prepared: PreparedGenerateRequest,
    response: ResponseChannel,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    deltanet_pool: Option<&std::sync::Mutex<crate::cache::deltanet_pool::DeltaNetPool>>,
) {
    let request_id = prepared.request.request_id.clone();
    let mut seq = Sequence::new(
        request_id.clone(),
        prepared.prompt_tokens.clone(),
        SamplingParams::default(),
        prepared.max_new.saturating_sub(1) as u32,
        prepared.request.stop.strings.clone(),
        prepared.request.stop.token_ids.clone(),
        None,
    );
    if let Some(engine) = engine {
        seq.prefix_cache_key = prefix_cache_key(engine, &seq.input_ids);
    }
    // Allocate a DeltaNet pool slot only when state has to survive across
    // scheduler steps: multi-token decode, prefix reuse, or chunked prefill.
    // One-shot final prefill can run from zero state without touching the
    // pool, avoiding per-layer slot zeroing that would never be consumed.
    //
    // When a slot is required, fail the request if the pool is exhausted:
    // `build_step_batch`
    // only forwards `deltanet_slots` when *every* sequence in the
    // batch has one, so a single slotless admission flips the whole
    // batch onto the non-pooled linear-attention path that clears
    // recurrent state every step — silently corrupting all
    // co-batched DeltaNet requests, not just the slotless one.
    let needs_deltanet_slot = deltanet_slot_needed(&prepared, &seq, scheduler.config());
    if needs_deltanet_slot && let Some(pool_mutex) = deltanet_pool {
        let allocated = pool_mutex.lock().ok().and_then(|mut pool| {
            let slot = pool.allocate()?;
            Some(slot)
        });
        if let Some(slot) = allocated {
            seq.deltanet_slot = Some(slot);
        } else {
            let max_new_tokens = prepared.max_new;
            let mut state = ArSequenceState {
                request_id: prepared.request.request_id.clone(),
                prepared: Some(prepared),
                response,
                gen_start: Instant::now(),
                prefill_ms: 0.0,
                started_sent: false,
                sent_text_len: 0,
                pending_token: None,
                output_tokens: Vec::new(),
                token_logprobs: Vec::new(),
                prompt_token_logprobs: None,
                max_new_tokens,
            };
            state.ensure_started();
            fail_state(
                state,
                EngineError::Unavailable(
                    "DeltaNet pool exhausted — increase `deltanet_pool_slots` or retry".into(),
                ),
            );
            return;
        }
    } else if deltanet_pool.is_some() {
        seq.prefill_must_be_atomic = true;
    }
    scheduler.add_request(seq);
    let max_new_tokens = prepared.max_new;
    states.insert(
        request_id,
        ArSequenceState {
            request_id: prepared.request.request_id.clone(),
            prepared: Some(prepared),
            response,
            gen_start: Instant::now(),
            prefill_ms: 0.0,
            started_sent: false,
            sent_text_len: 0,
            pending_token: None,
            output_tokens: Vec::new(),
            token_logprobs: Vec::new(),
            prompt_token_logprobs: None,
            max_new_tokens,
        },
    );
}

fn deltanet_slot_needed(
    prepared: &PreparedGenerateRequest,
    seq: &Sequence,
    config: &SchedulerConfig,
) -> bool {
    let prefill_cap = if config.long_prefill_token_threshold > 0 {
        config.long_prefill_token_threshold
    } else {
        config.max_num_batched_tokens
    };
    prepared.max_new > 1
        || seq.kv_computed_len > 0
        || seq.prefix_cache_key.is_some()
        || prepared.prompt_tokens.len() > prefill_cap
}

fn drain_ready_messages(
    engine: Option<&Arc<Engine>>,
    rx: &mut mpsc::UnboundedReceiver<ArMessage>,
    rx_open: &mut bool,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    deltanet_pool: Option<&std::sync::Mutex<crate::cache::deltanet_pool::DeltaNetPool>>,
    prepare_tx: Option<&mpsc::UnboundedSender<PreparedArMessage>>,
    pending_prepares: &mut usize,
) {
    loop {
        match rx.try_recv() {
            Ok(msg) => handle_message(
                engine,
                msg,
                scheduler,
                states,
                deltanet_pool,
                prepare_tx,
                pending_prepares,
            ),
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                *rx_open = false;
                break;
            }
        }
    }
}

fn drain_ready_prepared_messages(
    engine: &Engine,
    prepare_rx: &mut mpsc::UnboundedReceiver<PreparedArMessage>,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    deltanet_pool: Option<&std::sync::Mutex<crate::cache::deltanet_pool::DeltaNetPool>>,
    pending_prepares: &mut usize,
) {
    while let Ok(msg) = prepare_rx.try_recv() {
        handle_prepared_message(
            engine,
            msg,
            scheduler,
            states,
            deltanet_pool,
            pending_prepares,
        );
    }
}

async fn wait_for_initial_prefill_batch(
    engine: &Arc<Engine>,
    rx: &mut mpsc::UnboundedReceiver<ArMessage>,
    rx_open: &mut bool,
    prepare_rx: &mut mpsc::UnboundedReceiver<PreparedArMessage>,
    prepare_tx: &mpsc::UnboundedSender<PreparedArMessage>,
    pending_prepares: &mut usize,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    deltanet_pool: Option<&std::sync::Mutex<crate::cache::deltanet_pool::DeltaNetPool>>,
) -> bool {
    if scheduler.num_running() > 0 {
        return false;
    }

    let max_wait_ms = scheduler.config().max_batch_wait_ms;
    if max_wait_ms == 0
        || (scheduler.num_waiting() == 0 && *pending_prepares == 0)
        || scheduler.num_waiting() >= scheduler.config().max_batch_size
    {
        return false;
    }

    let wait = tokio::time::sleep(Duration::from_millis(max_wait_ms));
    tokio::pin!(wait);
    loop {
        tokio::select! {
            biased;
            msg = rx.recv(), if *rx_open => {
                match msg {
                    Some(msg) => handle_message(
                        Some(engine),
                        msg,
                        scheduler,
                        states,
                        deltanet_pool,
                        Some(prepare_tx),
                        pending_prepares,
                    ),
                    None => {
                        *rx_open = false;
                        break;
                    }
                }
                drain_ready_messages(
                    Some(engine),
                    rx,
                    rx_open,
                    scheduler,
                    states,
                    deltanet_pool,
                    Some(prepare_tx),
                    pending_prepares,
                );
                drain_ready_prepared_messages(
                    engine,
                    prepare_rx,
                    scheduler,
                    states,
                    deltanet_pool,
                    pending_prepares,
                );
                if !*rx_open || scheduler.num_waiting() >= scheduler.config().max_batch_size {
                    break;
                }
            }
            prepared = prepare_rx.recv(), if *pending_prepares > 0 => {
                if let Some(msg) = prepared {
                    handle_prepared_message(
                        engine,
                        msg,
                        scheduler,
                        states,
                        deltanet_pool,
                        pending_prepares,
                    );
                }
                drain_ready_prepared_messages(
                    engine,
                    prepare_rx,
                    scheduler,
                    states,
                    deltanet_pool,
                    pending_prepares,
                );
                if scheduler.num_waiting() >= scheduler.config().max_batch_size {
                    break;
                }
            }
            _ = &mut wait => break,
        }
    }
    // The wait window is the latency budget. Once it expires, use whatever
    // prepare work has completed and let slower tokenization join a later
    // scheduler step instead of blocking ready requests indefinitely.
    drain_ready_prepared_messages(
        engine,
        prepare_rx,
        scheduler,
        states,
        deltanet_pool,
        pending_prepares,
    );
    true
}

struct PrefixAttach {
    deltanet_state: Option<DeltaNetPrefixState>,
    replaced_blocks: Vec<u32>,
}

fn attach_prefix_cache_reuse(engine: &Engine, seq: &mut Sequence) -> Option<PrefixAttach> {
    if engine.cache.prefix_cache.is_none() || seq.input_ids.len() <= 1 {
        return None;
    }
    let Some(pool) = engine.cache.paged_pool.as_ref() else {
        return None;
    };

    let (mut cached_len, mut cached_blocks, deltanet_state) = match if engine
        .cache
        .deltanet_pool
        .is_some()
    {
        engine
            .cache
            .try_prefix_cache_match_paged_with_deltanet_state(&seq.input_ids)
    } else {
        engine
            .cache
            .try_prefix_cache_match_paged_only(&seq.input_ids)
            .map(|(len, blocks)| (len, blocks, None))
    } {
        Ok(hit) => hit,
        Err(error) => {
            tracing::warn!(request_id = %seq.request_id, error = %error, "prefix cache match failed");
            return None;
        }
    };

    // Reuse only full paged-KV blocks. Prefix-cache entries can be smaller
    // than paged blocks; sharing a partial page would let suffix writes mutate
    // cached KV that another prompt may later reuse.
    cached_len -= cached_len % pool.block_size;
    cached_blocks.truncate(cached_len / pool.block_size);

    if cached_len == 0
        || cached_len <= seq.kv_computed_len
        || cached_len >= seq.input_ids.len()
        || cached_blocks.is_empty()
    {
        return None;
    }

    if let Err(error) = engine.cache.retain_paged_blocks(&cached_blocks) {
        tracing::warn!(request_id = %seq.request_id, error = %error, "prefix cache block retain failed");
        return None;
    }

    seq.kv_computed_len = cached_len;
    let replaced_blocks = std::mem::replace(&mut seq.block_table, cached_blocks);
    tracing::debug!(
        request_id = %seq.request_id,
        cached_len,
        cached_blocks = seq.block_table.len(),
        suffix_len = seq.input_ids.len() - cached_len,
        "attached AR prefix-cache reuse"
    );
    Some(PrefixAttach {
        deltanet_state,
        replaced_blocks,
    })
}

fn prefix_cache_key(engine: &Engine, tokens: &[u32]) -> Option<u64> {
    if engine.cache.prefix_cache.is_none() {
        return None;
    }
    let block_size = engine.cache.paged_pool.as_ref()?.block_size;
    // Group by the first several full pages. One page is too coarse for
    // chat templates shared across unrelated policies; several pages capture
    // the policy/system prompt while still avoiding full-prompt hashing.
    let key_tokens = (block_size * 8).min(tokens.len());
    if key_tokens < block_size {
        return None;
    }
    let key_tokens = key_tokens - (key_tokens % block_size);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    block_size.hash(&mut hasher);
    tokens[..key_tokens].hash(&mut hasher);
    Some(hasher.finish())
}

fn refresh_waiting_prefix_cache(engine: &Engine, scheduler: &mut Scheduler) {
    if engine.cache.prefix_cache.is_none() {
        return;
    }

    let planned = scheduler.plan_waiting_shared_prefixes();
    if planned > 0 {
        tracing::debug!(planned, "planned shared prefix-cache boundaries");
    }

    let mut refreshed = 0usize;
    let mut blocks_to_release: Vec<Vec<u32>> = Vec::new();
    scheduler.for_each_waiting_mut(|seq| {
        if seq.kv_computed_len > 0
            && seq
                .prefix_cache_target_len
                .is_none_or(|target_len| seq.kv_computed_len >= target_len)
        {
            return;
        }
        let Some(attach) = attach_prefix_cache_reuse(engine, seq) else {
            return;
        };
        if !attach.replaced_blocks.is_empty() {
            blocks_to_release.push(attach.replaced_blocks);
        }
        if let Some(state) = attach.deltanet_state {
            let restored_slot = if let Some(slot) = seq.deltanet_slot {
                engine.cache.deltanet_pool.as_ref().and_then(|pool_mutex| {
                    pool_mutex
                        .lock()
                        .ok()
                        .and_then(|pool| match pool.restore_slot(slot, &state) {
                            Ok(()) => Some(slot),
                            Err(error) => {
                                tracing::warn!(request_id = %seq.request_id, slot, error = %error, "DeltaNet prefix state restore failed");
                                None
                            }
                        })
                })
            } else {
                engine.cache.deltanet_pool.as_ref().and_then(|pool_mutex| {
                    pool_mutex.lock().ok().and_then(|mut pool| {
                        let slot = pool.allocate()?;
                        if let Err(error) = pool.restore_slot(slot, &state) {
                            tracing::warn!(request_id = %seq.request_id, slot, error = %error, "DeltaNet prefix state restore failed");
                            pool.free(slot);
                            return None;
                        }
                        Some(slot)
                    })
                })
            };
            if let Some(slot) = restored_slot {
                seq.deltanet_slot = Some(slot);
            } else {
                seq.kv_computed_len = 0;
                blocks_to_release.push(std::mem::take(&mut seq.block_table));
                return;
            }
        }
        if seq.kv_computed_len > 0 {
            refreshed += 1;
        }
    });
    for blocks in blocks_to_release {
        scheduler.free_blocks(&blocks);
    }

    if refreshed > 0 {
        tracing::debug!(refreshed, "refreshed waiting prefix-cache reuse");
    }
}

// ── Batch construction ─────────────────────────────────────────────

/// Build a prefill ForwardBatch, handling both full and chunked prefill.
///
/// Final chunk: takes `prepared` from state (consumed).
/// Partial chunk: builds a minimal item from cached prompt tokens; `prepared`
/// stays in state for the final chunk (which needs logits_processor).
/// Build a ForwardBatch for the current step.
///
/// Pure decode → ForwardBatch::Decode (CUDA graph eligible).
/// Any prefill → ForwardBatch::Mixed (single forward pass with variable Q per request).
fn build_step_batch(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    step: &mut SchedulerStep,
) -> ForwardBatch {
    use crate::engine::executor::StepRequest;

    // Pure decode → use Decode variant for CUDA graph eligibility
    if step.prefill_request_ids.is_empty() {
        // Allocate new KV cache blocks for decode where needed.
        // total_len() = input_ids + output_ids (output_ids already includes the
        // token being decoded). The decode token sits at position total_len()-1
        // and context_len = total_len(). A new block is needed when position
        // total_len()-1 starts a fresh block, i.e. (total_len()-1) % block_size == 0.
        let block_size = engine
            .cache
            .paged_pool
            .as_ref()
            .map(|p| p.block_size)
            .unwrap_or(16);
        for id in &step.decode_request_ids {
            if let Some(seq) = scheduler.get_sequence(id) {
                let seq_len = seq.total_len();
                let position = seq_len - 1; // 0-indexed position of the decode token
                if position % block_size == 0 {
                    if let Some(new_block) = scheduler.allocate_block() {
                        if let Some(seq_mut) = scheduler.get_sequence_mut(id) {
                            seq_mut.block_table.push(new_block);
                        }
                    }
                }
            }
        }
        // Filter out requests whose block_table doesn't cover the current
        // decode position (allocate_block above returned None — KV pool
        // exhausted). We MUST drop these IDs from `step` itself: the
        // outer loop in `process_step_output` indexes the forward output
        // by position in `step.decode_request_ids`, so a length mismatch
        // between the batch we forward and the IDs we iterate would
        // misalign every logits row after the first skip and silently
        // mis-sample tokens for unrelated requests. The skipped requests
        // stay in the scheduler so the next step retries once blocks free.
        let mut deferred: Vec<String> = Vec::new();
        step.decode_request_ids.retain(|id| {
            let Some(seq) = scheduler.get_sequence(id) else {
                deferred.push(id.clone());
                return false;
            };
            let position = seq.total_len() - 1;
            if seq.block_table.len() * block_size <= position {
                tracing::warn!(
                    request_id = %id,
                    position,
                    blocks = seq.block_table.len(),
                    "KV block pool exhausted for decode — deferring this request"
                );
                deferred.push(id.clone());
                return false;
            }
            true
        });

        let cap = step.decode_request_ids.len();
        let mut tokens = Vec::with_capacity(cap);
        let mut positions = Vec::with_capacity(cap);
        let mut block_tables = Vec::with_capacity(cap);
        let mut dn_slots: Vec<u32> = Vec::new();
        let mut has_dn = false;
        for id in &step.decode_request_ids {
            if let (Some(state), Some(seq)) = (states.get(id), scheduler.get_sequence(id)) {
                let position = seq.total_len() - 1;
                tokens.push(state.pending_token.unwrap_or(0));
                // position = total_len() - 1: the decode token's 0-indexed position
                // (output_ids already contains this token from on_token_generated)
                positions.push(position);
                block_tables.push(seq.block_table.clone());
                if let Some(slot) = seq.deltanet_slot {
                    dn_slots.push(slot);
                    has_dn = true;
                }
            }
        }
        let deltanet_slots = if has_dn && dn_slots.len() == tokens.len() {
            Some(dn_slots)
        } else {
            None
        };
        let sample_greedy = decode_requests_allow_executor_greedy(states, &step.decode_request_ids);
        return ForwardBatch::Decode {
            tokens,
            positions,
            block_tables,
            deltanet_slots,
            sample_greedy,
        };
    }

    // Mixed or prefill-only → build unified StepRequests
    let mut requests: Vec<StepRequest> = Vec::new();
    let block_size = engine
        .cache
        .paged_pool
        .as_ref()
        .map(|p| p.block_size)
        .unwrap_or(16);
    let batch_needs_paged_prefill = prefill_batch_needs_kv(scheduler, step);

    // ── Prefill: try to allocate blocks; if pool is exhausted, drop the
    // request from `step` AND roll back the kv_computed_len bump that
    // `Scheduler::schedule_step` performed when planning this step. The
    // request stays in the scheduler so the same chunk can be retried
    // next step; failing to roll back would make the scheduler think
    // those tokens were already cached, and the next forward would
    // start from a pointer past the (still-uncomputed) tokens — wrong
    // attention output forever.
    let mut has_executor_sample_rows = false;
    let mut can_executor_sample_rows = true;
    let mut prefill_keep: Vec<bool> = Vec::with_capacity(step.prefill_request_ids.len());
    for (idx, id) in step.prefill_request_ids.iter().enumerate() {
        let chunk_len = step.prefill_chunk_lens.get(idx).copied().unwrap_or(0);
        let Some(seq) = scheduler.get_sequence(id) else {
            prefill_keep.push(false);
            continue;
        };
        let computed = seq.kv_computed_len.saturating_sub(chunk_len);
        let full_prompt_len = seq.input_ids.len();
        let is_final = computed + chunk_len >= full_prompt_len;
        let end = if is_final {
            full_prompt_len
        } else {
            computed + chunk_len
        };

        let total_blocks_needed = end.div_ceil(block_size);
        let current_blocks = seq.block_table.len();
        let deltanet_slot = seq.deltanet_slot;
        let needs_kv_cache = batch_needs_paged_prefill;
        if needs_kv_cache && current_blocks < total_blocks_needed {
            for _ in current_blocks..total_blocks_needed {
                match scheduler.allocate_block() {
                    Some(block) => {
                        if let Some(seq_mut) = scheduler.get_sequence_mut(id) {
                            seq_mut.block_table.push(block);
                        }
                    }
                    None => break,
                }
            }
            let actual = scheduler
                .get_sequence(id)
                .map(|s| s.block_table.len())
                .unwrap_or(0);
            if actual < total_blocks_needed {
                tracing::warn!(
                    request_id = %id,
                    needed = total_blocks_needed,
                    got = actual,
                    "KV block pool exhausted during prefill chunk allocation — deferring"
                );
                if let Some(seq_mut) = scheduler.get_sequence_mut(id) {
                    seq_mut.kv_computed_len = seq_mut.kv_computed_len.saturating_sub(chunk_len);
                    if seq_mut.status == crate::scheduler::SequenceStatus::Decoding {
                        seq_mut.status = crate::scheduler::SequenceStatus::Prefilling;
                    }
                }
                prefill_keep.push(false);
                continue;
            }
        }

        let Some(seq) = scheduler.get_sequence(id) else {
            prefill_keep.push(false);
            continue;
        };
        let chunk_tokens = seq.input_ids[computed..end].to_vec();
        if is_final {
            has_executor_sample_rows = true;
            can_executor_sample_rows &= states.get(id).map_or(false, state_allows_executor_greedy);
        }
        let prompt_logprobs = states
            .get(id)
            .and_then(|s| s.prepared.as_ref())
            .and_then(|p| p.request.prompt_logprobs);
        requests.push(StepRequest {
            tokens: chunk_tokens,
            context_len: end,
            position_start: computed,
            block_table: seq.block_table.clone(),
            is_prefill_final: is_final,
            is_prefill_partial: !is_final,
            deltanet_slot,
            prompt_logprobs,
            needs_kv_cache,
        });
        prefill_keep.push(true);
    }
    // Compact step.prefill_request_ids / step.prefill_chunk_lens by
    // dropping the indices we marked false. process_step_output walks
    // both vectors index-parallel with the forward output, so they MUST
    // shrink together.
    {
        let mut keep_iter = prefill_keep.iter().copied();
        step.prefill_request_ids
            .retain(|_| keep_iter.next().unwrap_or(false));
        let mut keep_iter = prefill_keep.iter().copied();
        step.prefill_chunk_lens
            .retain(|_| keep_iter.next().unwrap_or(false));
    }

    // ── Decode: same pattern as the pure-decode branch above. Try to
    // grow block_table; drop deferred requests from step.decode_request_ids
    // so the index-parallel logits walk in process_step_output stays
    // aligned with what we actually forward.
    for id in &step.decode_request_ids {
        if let Some(seq) = scheduler.get_sequence(id) {
            let position = seq.total_len() - 1;
            if position % block_size == 0 {
                if let Some(new_block) = scheduler.allocate_block() {
                    if let Some(seq_mut) = scheduler.get_sequence_mut(id) {
                        seq_mut.block_table.push(new_block);
                    }
                }
            }
        }
    }
    step.decode_request_ids.retain(|id| {
        let Some(seq) = scheduler.get_sequence(id) else {
            return false;
        };
        let position = seq.total_len() - 1;
        if seq.block_table.len() * block_size <= position {
            tracing::warn!(
                request_id = %id,
                position,
                blocks = seq.block_table.len(),
                "KV block pool exhausted for decode in mixed batch — deferring"
            );
            return false;
        }
        true
    });

    for id in &step.decode_request_ids {
        has_executor_sample_rows = true;
        can_executor_sample_rows &= states.get(id).map_or(false, state_allows_executor_greedy);
        if let (Some(state), Some(seq)) = (states.get(id), scheduler.get_sequence(id)) {
            let seq_len = seq.total_len();
            let position = seq_len - 1;
            requests.push(StepRequest {
                tokens: vec![state.pending_token.unwrap_or(0)],
                context_len: seq_len,
                position_start: position,
                block_table: seq.block_table.clone(),
                is_prefill_final: false,
                is_prefill_partial: false,
                deltanet_slot: seq.deltanet_slot,
                prompt_logprobs: None,
                needs_kv_cache: true,
            });
        }
    }

    ForwardBatch::Mixed {
        requests,
        sample_greedy: has_executor_sample_rows && can_executor_sample_rows,
    }
}

fn prefill_batch_needs_kv(scheduler: &Scheduler, step: &SchedulerStep) -> bool {
    if !step.decode_request_ids.is_empty() {
        return true;
    }

    step.prefill_request_ids
        .iter()
        .enumerate()
        .any(|(idx, id)| {
            let chunk_len = step.prefill_chunk_lens.get(idx).copied().unwrap_or(0);
            scheduler
                .get_sequence(id)
                .map(|seq| prefill_request_needs_kv(seq, chunk_len))
                .unwrap_or(false)
        })
}

fn prefill_request_needs_kv(seq: &Sequence, chunk_len: usize) -> bool {
    let computed_before = seq.kv_computed_len.saturating_sub(chunk_len);
    let computed_after = computed_before + chunk_len;
    let is_final_prompt_chunk = computed_after >= seq.input_ids.len();

    // A one-shot final prefill can emit the first generated token directly
    // from prompt logits. KV cache is only needed when this request has prior
    // cached chunks, will continue prompt prefill, or will decode after that
    // first token. In AR mode Sequence::max_new_tokens is initialized to
    // request.max_new - 1, so any positive remaining token budget means there
    // will be a decode step that needs prompt KV.
    computed_before > 0 || !is_final_prompt_chunk || seq.remaining_tokens() > 0
}

fn state_allows_executor_greedy(state: &ArSequenceState) -> bool {
    state.is_greedy()
        && state
            .prepared
            .as_ref()
            .and_then(|p| p.request.logprobs)
            .is_none()
}

fn decode_requests_allow_executor_greedy(
    states: &HashMap<String, ArSequenceState>,
    decode_request_ids: &[String],
) -> bool {
    decode_request_ids
        .iter()
        .all(|id| states.get(id).map_or(false, state_allows_executor_greedy))
}

// ── Output processing (sampling + stop conditions + delivery) ──────

/// Process output from a single forward pass. Handles both prefill results
/// and decode logits from the unified batch.
fn process_step_output(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    step: &SchedulerStep,
    output: &ModelOutput,
) {
    let mut completed: Vec<(String, FinishReason)> = Vec::new();
    let mut logit_row = 0usize; // tracks position in output.logits
    let prefill_argmax_tokens = batched_prefill_argmax_tokens(scheduler, states, step, output);

    // ── Prefill results ───────────────────────────────────────────
    let mut prefill_result_idx = 0usize;
    for (i, request_id) in step.prefill_request_ids.iter().enumerate() {
        let Some(state) = states.get_mut(request_id) else {
            logit_row += 1;
            continue;
        };

        let Some(seq) = scheduler.get_sequence(request_id) else {
            logit_row += 1;
            continue;
        };

        let _ = step.prefill_chunk_lens.get(i); // chunk_len unused here; see kv_computed_len below
        let full_prompt_len = seq.input_ids.len();
        // scheduler::schedule_step already incremented kv_computed_len to reflect
        // this chunk's tokens. Don't add chunk_len again — doing so double-counts
        // and flags the second-to-last chunk as final on prompts where the last
        // chunk is shorter than the chunk budget.
        let computed = seq.kv_computed_len;
        let is_final = computed >= full_prompt_len;

        // Update scheduler sequence block table and state from prefill result.
        // kv_computed_len and status were already updated by the scheduler
        // during schedule_step (eagerly, before forward).
        //
        // Hold onto `result` past the increment so the prefix-cache write
        // below can read the SAME entry. Earlier this code recovered it
        // via `prefill_results.get(prefill_result_idx.saturating_sub(1))`,
        // which silently read the previous request's block_table whenever
        // this request's `prefill_results.get(...)` was None — so the
        // prefix cache could be populated against tokens that belonged to
        // a different sequence.
        let prefill_result = output.prefill_results.get(prefill_result_idx);
        if let Some(result) = prefill_result {
            if let Some(seq_mut) = scheduler.get_sequence_mut(request_id) {
                seq_mut.block_table = result.block_table.clone();
            }
            state.prefill_ms += result.prefill_ms;
            state.prompt_token_logprobs = result.prompt_token_logprobs.clone();
            prefill_result_idx += 1;
        }

        // Populate prefix cache once a prefill reaches a reusable boundary.
        // Non-hybrid models only need KV blocks and can use the completed
        // prompt. Hybrid DeltaNet models additionally need a state snapshot
        // at the exact block-aligned boundary being cached.
        if engine.cache.prefix_cache.is_some() {
            if let (Some(seq), Some(pool), Some(result)) = (
                scheduler.get_sequence(request_id),
                engine.cache.paged_pool.as_ref(),
                prefill_result,
            ) {
                if !result.block_table.is_empty() && !seq.input_ids.is_empty() {
                    if let Some(dn_pool_mutex) = engine.cache.deltanet_pool.as_ref() {
                        if computed > 0
                            && computed <= seq.input_ids.len()
                            && computed % pool.block_size == 0
                            && let Some(slot) = seq.deltanet_slot
                        {
                            match dn_pool_mutex
                                .lock()
                                .map_err(|e| {
                                    EngineError::Internal(format!("deltanet pool lock: {e}"))
                                })
                                .and_then(|pool| {
                                    pool.snapshot_slot(slot).map_err(|e| {
                                        EngineError::Internal(format!(
                                            "DeltaNet prefix state snapshot failed: {e}"
                                        ))
                                    })
                                }) {
                                Ok(deltanet_state) => {
                                    if let Err(e) = engine
                                        .cache
                                        .try_prefix_cache_insert_paged_with_deltanet_state(
                                            &seq.input_ids[..computed],
                                            &result.block_table,
                                            pool.block_size,
                                            deltanet_state,
                                        )
                                    {
                                        tracing::warn!("hybrid prefix cache insert failed: {e}");
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(error = %e, "hybrid prefix state snapshot failed");
                                }
                            }
                        }
                    } else if is_final
                        && let Err(e) = engine.cache.try_prefix_cache_insert_paged_only(
                            &seq.input_ids,
                            &result.block_table,
                            pool.block_size,
                        )
                    {
                        tracing::warn!("prefix cache insert (ar_loop) failed: {e}");
                    }
                }
            }
        }

        if is_final {
            // Final chunk: sample first token from logits. Mirror the decode
            // path's sampling: greedy → argmax, otherwise route through the
            // request's own `LogitsProcessor` so temperature/top-p/penalties
            // still apply to this first token.
            let needs_row = !state.is_greedy()
                || state
                    .prepared
                    .as_ref()
                    .and_then(|p| p.request.logprobs)
                    .is_some()
                || prefill_argmax_tokens.is_none();
            let row = if needs_row {
                output.logits.get(logit_row).ok()
            } else {
                None
            };
            let token = if state.is_greedy() {
                if let Some(token) = prefill_argmax_tokens
                    .as_ref()
                    .and_then(|tokens| tokens.get(i).copied())
                {
                    token
                } else if let Some(ref row) = row {
                    sample_token(row, state)
                } else {
                    0
                }
            } else if let Some(ref row) = row {
                sample_token(row, state)
            } else {
                0
            };

            if prefill_argmax_tokens.is_some() || row.is_some() {
                let lp = row
                    .as_ref()
                    .and_then(|r| extract_token_logprobs(engine, r, token, state));
                process_single_token(
                    engine,
                    scheduler,
                    state,
                    request_id,
                    token,
                    lp,
                    &mut completed,
                );
            }
        }
        // Partial chunk: nothing to do (progress already recorded)
        logit_row += 1;
    }

    // ── Decode results (batched GPU sampling) ──────────────────────
    let num_decode = step.decode_request_ids.len();
    if num_decode > 0 {
        let decode_start_row = logit_row;

        // Check if all decode sequences are greedy
        let all_greedy = step
            .decode_request_ids
            .iter()
            .all(|id| states.get(id).map(|s| s.is_greedy()).unwrap_or(true));
        let can_use_executor_tokens =
            decode_requests_allow_executor_greedy(states, &step.decode_request_ids);

        // Greedy decode only needs argmax. Avoid the FlashInfer sampler path
        // here because it converts the full [batch, vocab] logits tensor to
        // F32 before sampling; BF16 argmax gives the same token and saves a
        // large per-step copy/conversion.
        let batched_tokens: Option<Vec<u32>> = if all_greedy {
            let executor_tokens = can_use_executor_tokens
                .then(|| sampled_tokens_for_rows(output, decode_start_row, num_decode))
                .flatten();
            executor_tokens.or_else(|| decode_argmax_tokens(output, decode_start_row, num_decode))
        } else {
            None
        };

        for (i, request_id) in step.decode_request_ids.iter().enumerate() {
            let Some(state) = states.get_mut(request_id) else {
                logit_row += 1;
                continue;
            };

            let wants_logprobs = state
                .prepared
                .as_ref()
                .and_then(|p| p.request.logprobs)
                .is_some();
            let needs_row = batched_tokens.is_none() || wants_logprobs;
            let row = if needs_row {
                output.logits.get(logit_row).ok()
            } else {
                None
            };

            let token = if let Some(ref tokens) = batched_tokens {
                tokens[i]
            } else if let Some(ref row) = row {
                sample_token(row, state)
            } else {
                0
            };

            // Per-token top-k logprobs when requested. Without this the
            // /v1/completions response leaves `logprobs.tokens = [first_only]`
            // and downstream tools count tokens via `len(logprobs.tokens)`.
            let lp = row
                .as_ref()
                .and_then(|r| extract_token_logprobs(engine, r, token, state));

            process_single_token(
                engine,
                scheduler,
                state,
                request_id,
                token,
                lp,
                &mut completed,
            );
            logit_row += 1;
        }
    }

    for (request_id, finish_reason) in completed {
        // Get prompt_len from scheduler sequence before finishing
        let prompt_len = scheduler
            .get_sequence(&request_id)
            .map(|seq| seq.input_ids.len())
            .unwrap_or(0);
        release_resources(engine.cache.deltanet_pool.as_ref(), scheduler, &request_id);
        scheduler.finish_request(&request_id, seq_finish_reason(&finish_reason));
        if let Some(state) = states.remove(&request_id) {
            finish_state(engine, state, finish_reason, prompt_len);
        }
    }
}

fn batched_prefill_argmax_tokens(
    scheduler: &Scheduler,
    states: &HashMap<String, ArSequenceState>,
    step: &SchedulerStep,
    output: &ModelOutput,
) -> Option<Vec<u32>> {
    let mut final_greedy_count = 0usize;
    for (i, request_id) in step.prefill_request_ids.iter().enumerate() {
        let state = states.get(request_id)?;
        let seq = scheduler.get_sequence(request_id)?;
        if seq.kv_computed_len >= seq.input_ids.len() {
            if !state.is_greedy() {
                return None;
            }
            final_greedy_count = i + 1;
        }
    }

    if final_greedy_count == 0 {
        return None;
    }

    if let Some(tokens) = sampled_tokens_for_rows(output, 0, final_greedy_count) {
        return Some(tokens);
    }

    output
        .logits
        .narrow(0, 0, final_greedy_count)
        .ok()
        .and_then(|logits| {
            logits
                .argmax(crate::tensor::D::Minus1)
                .and_then(|t| t.to_vec1::<u32>())
                .ok()
        })
}

fn sampled_tokens_for_rows(
    output: &ModelOutput,
    start_row: usize,
    row_count: usize,
) -> Option<Vec<u32>> {
    let tokens = output.sampled_tokens.as_ref()?;
    if tokens.len() == row_count && start_row == 0 {
        return Some(tokens.clone());
    }
    let end = start_row.checked_add(row_count)?;
    if tokens.len() < end {
        return None;
    }
    Some(tokens[start_row..end].to_vec())
}

fn decode_argmax_tokens(
    output: &ModelOutput,
    decode_start_row: usize,
    num_decode: usize,
) -> Option<Vec<u32>> {
    output
        .logits
        .narrow(0, decode_start_row, num_decode)
        .and_then(|decode_logits| decode_logits.argmax(crate::tensor::D::Minus1))
        .and_then(|tokens| tokens.to_vec1::<u32>())
        .ok()
}

/// Sample a token from a logits row using the sequence's sampling config.
///
/// Greedy → argmax. Otherwise route through the request's `LogitsProcessor`
/// (temperature/top-p/etc.). On any failure, fall back to argmax to avoid
/// emitting a degenerate `0` token.
fn sample_token(row: &crate::tensor::Tensor, state: &mut ArSequenceState) -> u32 {
    let argmax_fallback = || {
        row.argmax(crate::tensor::D::Minus1)
            .and_then(|t| t.to_scalar::<u32>())
            .unwrap_or(0)
    };
    if state.is_greedy() {
        return argmax_fallback();
    }
    let Some(prepared) = state.prepared.as_mut() else {
        return argmax_fallback();
    };
    let Ok(row_f32) = row.to_dtype(crate::tensor::DType::F32) else {
        return argmax_fallback();
    };
    prepared
        .logits_processor
        .sample(&row_f32)
        .unwrap_or_else(|_| argmax_fallback())
}

/// Compute top-k logprobs for a sampled token if the request asked for them.
fn extract_token_logprobs(
    engine: &Engine,
    row: &crate::tensor::Tensor,
    token: u32,
    state: &ArSequenceState,
) -> Option<TokenLogprobInfo> {
    let k = state.prepared.as_ref()?.request.logprobs?;
    Engine::extract_top_logprobs(row, token, k, &engine.tokenizer).ok()
}

/// Process a single sampled token: check stop conditions, stream, check length.
fn process_single_token(
    engine: &Engine,
    scheduler: &mut Scheduler,
    state: &mut ArSequenceState,
    request_id: &str,
    next_token: u32,
    token_logprobs: Option<TokenLogprobInfo>,
    completed: &mut Vec<(String, FinishReason)>,
) {
    scheduler.on_token_generated(request_id, next_token);
    state.ensure_started();

    // Record the sampled token before stop checks so usage and logprobs count
    // EOS/stop-token generations consistently with HF/OpenAI semantics. Special
    // tokens are skipped by decode(..., true), so they do not appear in text.
    state.pending_token = Some(next_token);
    state.output_tokens.push(next_token);
    if let Some(ref lp) = token_logprobs {
        state.token_logprobs.push(lp.clone());
    }

    // Check stop conditions before streaming visible text.
    if engine.is_eos(next_token) {
        completed.push((request_id.to_string(), FinishReason::Eos));
        return;
    }
    if let Some(ref prepared) = state.prepared {
        if engine.check_stop_tokens(next_token, &prepared.request.stop.token_ids) {
            completed.push((request_id.to_string(), FinishReason::Stop));
            return;
        }
    }

    state.emit_text_delta(&engine.tokenizer, token_logprobs);

    // Check max length before stop strings. Most production requests have no
    // stop strings, so avoid decoding the whole generated text every token.
    if state.output_tokens.len() >= state.max_new() {
        completed.push((request_id.to_string(), FinishReason::Length));
    } else if let Some(ref prepared) = state.prepared {
        if !prepared.request.stop.strings.is_empty() {
            let text = state.current_text(&engine.tokenizer);
            if prepared
                .request
                .stop
                .strings
                .iter()
                .any(|s| text.contains(s))
            {
                completed.push((request_id.to_string(), FinishReason::Stop));
            }
        }
    }
}

// ── State finalization ─────────────────────────────────────────────

fn finish_state(
    engine: &Engine,
    mut state: ArSequenceState,
    finish_reason: FinishReason,
    prompt_len: usize,
) {
    state.ensure_started();

    let output_text = state.current_text(&engine.tokenizer);
    let completion_tokens = state.output_tokens.len() as u32;
    let total_ms = state.gen_start.elapsed().as_secs_f32() * 1000.0;
    let usage = Usage {
        prompt_tokens: prompt_len as u32,
        completion_tokens,
        total_tokens: prompt_len as u32 + completion_tokens,
    };
    let metrics = DecodeMetrics {
        ttft_ms: state.prefill_ms,
        prefill_ms: state.prefill_ms,
        decode_ms: (total_ms - state.prefill_ms).max(0.0),
        total_ms,
    };
    let result = GenerateResult {
        model: state
            .prepared
            .as_ref()
            .map(|p| p.request.model.clone())
            .unwrap_or_default(),
        output_token_ids: state.output_tokens,
        output_text,
        finish_reason: finish_reason.clone(),
        usage: usage.clone(),
        metrics: metrics.clone(),
        token_logprobs: if state.token_logprobs.is_empty() {
            None
        } else {
            Some(state.token_logprobs)
        },
        prompt_token_logprobs: state.prompt_token_logprobs,
    };

    match state.response {
        ResponseChannel::Complete(tx) => {
            let _ = tx.send(Ok(result));
        }
        ResponseChannel::Stream(tx) => {
            let _ = tx.send(StreamEvent::Finished {
                finish_reason,
                usage,
                metrics,
            });
        }
    }
}

fn fail_state(mut state: ArSequenceState, error: EngineError) {
    state.ensure_started();
    match state.response {
        ResponseChannel::Complete(tx) => {
            let _ = tx.send(Err(error));
        }
        ResponseChannel::Stream(tx) => {
            let _ = tx.send(StreamEvent::Error {
                message: error.to_string(),
            });
        }
    }
}

fn fail_step(
    engine: &Engine,
    scheduler: &mut Scheduler,
    states: &mut HashMap<String, ArSequenceState>,
    step: &SchedulerStep,
    error: EngineError,
) {
    for id in step
        .prefill_request_ids
        .iter()
        .chain(step.decode_request_ids.iter())
    {
        release_resources(engine.cache.deltanet_pool.as_ref(), scheduler, id);
        let _ = scheduler.abort_request(id);
        if let Some(state) = states.remove(id) {
            fail_state(state, error.clone());
        }
    }
}

fn release_resources(
    deltanet_pool: Option<&std::sync::Mutex<crate::cache::deltanet_pool::DeltaNetPool>>,
    scheduler: &mut Scheduler,
    request_id: &str,
) {
    if let Some(seq) = scheduler.get_sequence(request_id) {
        if !seq.block_table.is_empty() {
            let blocks = seq.block_table.clone();
            scheduler.free_blocks(&blocks);
        }
        if let Some(slot) = seq.deltanet_slot {
            if let Some(pool_mutex) = deltanet_pool {
                if let Ok(mut pool) = pool_mutex.lock() {
                    pool.free(slot);
                }
            }
        }
        // Clear on the sequence
        if let Some(seq_mut) = scheduler.get_sequence_mut(request_id) {
            seq_mut.block_table.clear();
            seq_mut.deltanet_slot = None;
        }
    }
}

fn seq_finish_reason(reason: &FinishReason) -> SeqFinishReason {
    match reason {
        FinishReason::Stop => SeqFinishReason::Stop,
        FinishReason::Length => SeqFinishReason::Length,
        FinishReason::Eos => SeqFinishReason::Eos,
        FinishReason::Cancelled => SeqFinishReason::Abort("cancelled".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prepared(request_id: &str, max_new: usize) -> PreparedGenerateRequest {
        use crate::engine::sampling::{LogitsProcessor, Sampling};
        PreparedGenerateRequest {
            request_idx: 0,
            request: crate::types::GenerateRequest {
                request_id: request_id.to_string(),
                model: String::new(),
                input: crate::types::PromptInput::Text(String::new()),
                sampling: SamplingParams::default(),
                max_new_tokens: max_new as u32,
                stop: Default::default(),
                seed: None,
                deadline_ms: None,
                logprobs: None,
                prompt_logprobs: None,
            },
            prompt_tokens: vec![1, 2, 3],
            max_new,
            is_greedy: true,
            logits_processor: LogitsProcessor::from_sampling(42, Sampling::ArgMax),
        }
    }

    // ── handle_message tests ───────────────────────────────────────

    #[test]
    fn handle_new_request_adds_to_scheduler() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let mut states = HashMap::new();
        let mut pending_prepares = 0;

        let prepared = make_prepared("r1", 10);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        handle_message(
            None,
            ArMessage::NewRequest {
                prepared,
                response: ResponseChannel::Complete(tx),
            },
            &mut scheduler,
            &mut states,
            None,
            None,
            &mut pending_prepares,
        );

        assert_eq!(scheduler.num_waiting(), 1);
        assert!(states.contains_key("r1"));
    }

    #[test]
    fn handle_abort_removes_from_scheduler() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let mut states = HashMap::new();
        let mut pending_prepares = 0;

        let prepared = make_prepared("r1", 10);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        handle_message(
            None,
            ArMessage::NewRequest {
                prepared,
                response: ResponseChannel::Complete(tx),
            },
            &mut scheduler,
            &mut states,
            None,
            None,
            &mut pending_prepares,
        );
        assert_eq!(scheduler.num_waiting(), 1);

        handle_message(
            None,
            ArMessage::Abort("r1".into()),
            &mut scheduler,
            &mut states,
            None,
            None,
            &mut pending_prepares,
        );
        assert!(states.is_empty());
    }

    // ── sample_batch tests ─────────────────────────────────────────

    // ── fail_step tests ────────────────────────────────────────────

    #[test]
    fn fail_step_aborts_all_ids() {
        let mut scheduler = Scheduler::new(SchedulerConfig::default());

        // Add two requests
        for id in &["r1", "r2"] {
            let seq = Sequence::new(
                id.to_string(),
                vec![1, 2, 3],
                SamplingParams::default(),
                10,
                vec![],
                vec![],
                None,
            );
            scheduler.add_request(seq);
        }
        assert_eq!(scheduler.num_waiting(), 2);

        let step = SchedulerStep::prefill(vec!["r1".into(), "r2".into()], vec![3, 3]);

        // We can't call fail_step without Engine, but we can test the scheduler abort part
        for id in step
            .prefill_request_ids
            .iter()
            .chain(step.decode_request_ids.iter())
        {
            let _ = scheduler.abort_request(id);
        }
        assert_eq!(scheduler.num_waiting(), 0);
    }

    // ── seq_finish_reason mapping ──────────────────────────────────

    #[test]
    fn finish_reason_mapping() {
        assert!(matches!(
            seq_finish_reason(&FinishReason::Eos),
            SeqFinishReason::Eos
        ));
        assert!(matches!(
            seq_finish_reason(&FinishReason::Stop),
            SeqFinishReason::Stop
        ));
        assert!(matches!(
            seq_finish_reason(&FinishReason::Length),
            SeqFinishReason::Length
        ));
        assert!(matches!(
            seq_finish_reason(&FinishReason::Cancelled),
            SeqFinishReason::Abort(_)
        ));
    }

    // ── build_forward_batch tests ─────────────────────────────────

    fn make_decode_state(id: &str, pending_token: u32, _position: usize) -> ArSequenceState {
        let max_new = 10;
        ArSequenceState {
            request_id: id.to_string(),
            prepared: Some(make_prepared(id, max_new)),
            response: ResponseChannel::Complete(tokio::sync::oneshot::channel().0),
            gen_start: Instant::now(),
            prefill_ms: 0.0,
            started_sent: true,
            sent_text_len: 0,
            pending_token: Some(pending_token),
            output_tokens: vec![pending_token],
            token_logprobs: vec![],
            prompt_token_logprobs: None,
            max_new_tokens: max_new,
        }
    }

    #[test]
    fn decode_state_has_correct_fields() {
        let state = make_decode_state("s1", 42, 5);
        assert_eq!(state.pending_token, Some(42));
        assert_eq!(state.output_tokens, vec![42]);
    }

    #[test]
    fn prefill_kv_needed_only_when_followup_work_needs_cache() {
        let make_seq = |id: &str, max_new_after_prefill: u32, kv_computed_len: usize| {
            let mut seq = Sequence::new(
                id.to_string(),
                vec![1, 2, 3],
                SamplingParams::default(),
                max_new_after_prefill,
                vec![],
                vec![],
                None,
            );
            seq.kv_computed_len = kv_computed_len;
            seq
        };

        // max_tokens=1: final one-shot prefill can sample from prompt logits
        // and never needs paged KV.
        assert!(!prefill_request_needs_kv(&make_seq("m1", 0, 3), 3));

        // max_tokens=2: the first decode step after prefill needs prompt KV.
        assert!(prefill_request_needs_kv(&make_seq("m2", 1, 3), 3));

        // Chunked prompt work needs KV even if it will finish with one token.
        assert!(prefill_request_needs_kv(&make_seq("partial", 0, 2), 2));
        assert!(prefill_request_needs_kv(&make_seq("final_chunk", 0, 3), 1));
    }

    #[test]
    fn deltanet_slot_needed_only_for_cross_step_state() {
        let config = SchedulerConfig {
            max_num_batched_tokens: 8,
            ..SchedulerConfig::default()
        };
        let mut prepared = make_prepared("oneshot", 1);
        prepared.prompt_tokens = vec![1, 2, 3];
        let seq = Sequence::new(
            "oneshot".to_string(),
            prepared.prompt_tokens.clone(),
            SamplingParams::default(),
            0,
            vec![],
            vec![],
            None,
        );
        assert!(!deltanet_slot_needed(&prepared, &seq, &config));

        let decode_prepared = make_prepared("decode", 4);
        let decode_seq = Sequence::new(
            "decode".to_string(),
            decode_prepared.prompt_tokens.clone(),
            SamplingParams::default(),
            3,
            vec![],
            vec![],
            None,
        );
        assert!(deltanet_slot_needed(&decode_prepared, &decode_seq, &config));

        let mut long_prepared = make_prepared("long", 1);
        long_prepared.prompt_tokens = vec![1; 16];
        let long_seq = Sequence::new(
            "long".to_string(),
            long_prepared.prompt_tokens.clone(),
            SamplingParams::default(),
            0,
            vec![],
            vec![],
            None,
        );
        assert!(deltanet_slot_needed(&long_prepared, &long_seq, &config));

        let mut prefix_seq = seq;
        prefix_seq.kv_computed_len = 1;
        assert!(deltanet_slot_needed(&prepared, &prefix_seq, &config));
    }

    #[test]
    fn prefill_state_prepared_can_be_taken() {
        let mut states = HashMap::new();
        let mut pending_prepares = 0;
        let prepared = make_prepared("r1", 5);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        handle_message(
            None,
            ArMessage::NewRequest {
                prepared,
                response: ResponseChannel::Complete(tx),
            },
            &mut Scheduler::new(SchedulerConfig::default()),
            &mut states,
            None,
            None,
            &mut pending_prepares,
        );

        assert!(states.get("r1").unwrap().prepared.is_some());

        // Simulate what build_forward_batch does for prefill: take prepared
        let taken = states.get_mut("r1").and_then(|s| s.prepared.take());
        assert!(taken.is_some());
        assert_eq!(taken.unwrap().prompt_tokens, vec![1, 2, 3]);
        assert!(states.get("r1").unwrap().prepared.is_none());
    }
}

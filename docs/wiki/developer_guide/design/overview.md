# Design Overview

# Prelude Inference Engine -- Architecture Overview

## 1. Engine Hierarchy

All inference goes through the `InferenceEngine` trait (`engine/mod.rs`).
`PseudoEngine` is a mock for testing. `Engine` owns the model weights, tokenizer,
and KV cache and executes a single forward pass. `ScheduledEngine` is the
production path: it wraps `Engine` via an async `ar_loop` that drives continuous
batching, scheduling, and device dispatch.

```
InferenceEngine  (trait: engine/mod.rs)
  ├── PseudoEngine                    mock, no model
  ├── Engine                          owns weights + KV cache; executes forward pass
  │     ├── Tokenizer                 text ↔ token IDs
  │     ├── ModelExecutor
  │     │     ├── model: Box<dyn ModelForward>    models/{qwen3,gemma3,llama,...}.rs
  │     │     │   device-agnostic; calls ops.xxx() for all compute, no kernel refs
  │     │     └── ops: &'static dyn Ops            ops/traits/ops.rs
  │     │           flat API: ops.rms_norm, ops.varlen_attention, ...
  │     │           fused ops: try device kernel → auto-fallback to composed
  │     └── CacheManager
  │           ├── PagedKvPool         KV tensors on device
  │           ├── BlockManager        alloc/free, ref-count  cache/block_manager.rs
  │           └── PrefixKvCache       LPM + LRU eviction     cache/prefix_cache.rs
  └── ScheduledEngine                 Engine + dynamic batching + GPU queue
        │  public handle; spawns ar_loop on creation
        │  sends GenerateRequests via mpsc; streams tokens back via response channel
        │
        └── ar_loop  (engine/run/ar.rs)
              single async task; continuous-batching loop until all seqs finish
              │
              ├── ArScheduler  (scheduler/ar.rs)
              │     pure CPU — no device knowledge
              │     waiting queue / running list / finished list
              │     schedule_step() → selects seqs to prefill / decode
              │     budgets: max_running_requests, max_prefill_tokens, max_total_kv_tokens
              │     state machine: Waiting→Prefilling→Decoding→Finished (preemption)
              │     RequestQueue: FCFS / priority / cache-aware  components/request_queue.rs
              │
              └── Executor  (trait: engine/executor.rs)
                    submit(ForwardBatch) → ExecutionHandle  (non-blocking)
                    handle.await         → ModelOutput
                    enables double-buffering: prepare batch N+1 while device runs N
                    ├── CudaExecutor  prelude-cuda/src/executor.rs
                    │     single-threaded GPU queue draining; CUDA graph capture+replay for decode
                    ├── RocmExecutor  prelude-rocm/src/executor.rs
                    │     HIP command queue; HIP graph capture+replay
                    └── CpuExecutor   prelude-cpu/src/executor.rs
                          block_in_place; sequential, no graph
```

## 2. Request Flow

HTTP requests enter `prelude-server` (axum router, auth, SSE), which is
backend-agnostic and only knows `InferenceEngine`. `ScheduledEngine` tokenizes
the prompt, sends it through an mpsc channel to `ar_loop`, and streams tokens
back to the caller.

```
HTTP Request
    │
    ▼
prelude-server  (axum router, auth, SSE)
    │
    ▼
ScheduledEngine.generate(GenerateRequest)
    │  tokenize + build sampling params
    │
    ├──channel──▶  ar_loop  (loop until all seqs done)
    │                  │
    │          ┌───────▼──────────────────────────────────────┐
    │          │  1. Drain new requests from channel           │
    │          │     ArScheduler::add_request(seq)             │
    │          └───────┬──────────────────────────────────────┘
    │                  │
    │          ┌───────▼──────────────────────────────────────┐
    │          │  2. ArScheduler::schedule_step()              │
    │          │     select sequences to prefill / decode      │
    │          └───────┬──────────────────────────────────────┘
    │                  │
    │          ┌───────▼──────────────────────────────────────┐
    │          │  3. Engine: build ForwardBatch                │
    │          │                                              │
    │          │  Prefill  (varlen attention)                  │
    │          │  • find common prefix → reuse cached KV blocks│
    │          │  • allocate remaining KV blocks               │
    │          │  • model.forward(packed_tokens)               │
    │          │  • sample first output token                  │
    │          │                                              │
    │          │  Decode   (paged attention)                   │
    │          │  • one token per running sequence             │
    │          │  • model.forward(single tokens)               │
    │          │  • read KV from paged block tables            │
    │          └───────┬──────────────────────────────────────┘
    │                  │
    │          ┌───────▼──────────────────────────────────────┐
    │          │  4. Executor::submit(batch)  (non-blocking)   │
    │          │     handle.await → ModelOutput                │
    │          │     double-buffering: prep N+1 while N runs   │
    │          └───────┬──────────────────────────────────────┘
    │                  │
    │          ┌───────▼──────────────────────────────────────┐
    │          │  5. Sample + stop-condition check             │
    │          │     LogitsProcessor (temp, top-p/k)           │
    │          │     EOS / stop string / max tokens            │
    │          └───────┬──────────────────────────────────────┘
    │                  │
    │          ┌───────▼──────────────────────────────────────┐
    │          │  6. Stream token / deliver result             │
    │          │     → StreamEvent::Token                      │
    │          │     → GenerateResult on finish                │
    │          └───────┬──────────────────────────────────────┘
    │                  │  (loop back to step 1)
    │                  └──────────────────────────────────────▶ loop
    │
    ▼
HTTP Response / SSE stream
```

Kernel dispatch via `ops.xxx()` (called from `model.forward()` for every transformer layer):

```
  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  Linear::forward()  →  ops.matmul()  →  CudaOps::matmul()                          │
  │  (models/layers/linear.rs)              (prelude-cuda/src/ops/gemm.rs)              │
  │                                                                                     │
  │  Covers: Q/K/V projections, output projection, MLP gate/up/down, lm_head (~7×/layer)│
  │                                                                                     │
  │  ├─ Try DeepGEMM      prelude-cuda/deepgemm/       SM90+ only, BF16, non-batched   │
  │  └─ Fallback CUTLASS  prelude-cuda/cutlass-gemm/   SM80+, BF16/FP16/F32            │
  └─────────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  ops.varlen_attention()  →  CudaOps::varlen_attention()   [prefill: Q > 1]          │
  │  ops.paged_attention()   →  CudaOps::paged_attention()    [decode:  Q = 1]          │
  │  (ops/traits/attention.rs)  (prelude-cuda/src/cuda_ops.rs)                          │
  │                                                                                     │
  │  ├─ Try FA4         prelude-cuda/src/attn/flash_v4.rs     SM80+, BF16 best-effort  │
  │  └─ Fallback        prelude-cuda/src/attn/flashinfer.rs   SM80+ FA2 / SM90+ FA3    │
  │       FlashInfer plan() runs once per forward, cached across all layers.            │
  └─────────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │  Fused ops (all return Option — None means unsupported, auto-falls back):           │
  │  ops.fused_add_rmsnorm()       →  residual + norm in one kernel                    │
  │  ops.fused_silu_mul()          →  SiLU gate × up-projection in one kernel          │
  │  ops.fused_qknorm_rope()       →  Q/K norm + RoPE in one kernel                   │
  │  ops.fused_knorm_rope_cache_write()  →  K norm + RoPE + KV cache scatter           │
  │  Implementations: prelude-cuda/src/ops/{rmsnorm,rope,kv_cache,elementwise}.rs      │
  └─────────────────────────────────────────────────────────────────────────────────────┘
```


## 3. File Structure

```
prelude-core/src/
├── engine/
│   ├── mod.rs                  exports; InferenceEngine trait
│   ├── engine.rs               Engine struct — model loading, tokenization, forward dispatch
│   ├── scheduled.rs            ScheduledEngine — public handle, spawns ar_loop
│   ├── executor.rs             Executor trait (submit → ExecutionHandle)
│   ├── planner.rs              cache allocation planner for paged attention
│   ├── config.rs               model config loading and resolution
│   ├── device.rs               device initialization and selection
│   ├── loading.rs              weight fetching (HuggingFace)
│   ├── weights.rs              weight loading via safetensors
│   ├── tokenizer.rs            tokenization with fallback handling
│   ├── types.rs                shared types and enums (TaskKind, EngineError)
│   ├── pseudo.rs               PseudoEngine — mock, no model
│   │
│   ├── model_runner/           forward execution paths
│   │   ├── generate.rs         autoregressive token generation forward
│   │   ├── classify.rs         text classification forward and postprocessing
│   │   ├── embed.rs            embedding forward and postprocessing
│   │   ├── prefill.rs          non-paged prefill forward pass
│   │   ├── paged_prefill.rs    paged attention prefill forward
│   │   └── paged_decode.rs     paged attention decode forward (Q=1 per seq)
│   │
│   ├── run/                    scheduling-paradigm loops
│   │   └── ar.rs               ar_loop — continuous batching (prefill + decode + sample)
│   │
│   ├── sampling/
│   │   ├── logits_processor.rs LogitsProcessor — temperature, top-p/k, penalties
│   │   └── grammar.rs          constrained decoding (grammar backends)
│   │
│   └── speculative/            speculative decoding
│       ├── proposer.rs         proposer strategies (EAGLE, draft model, ngram)
│       ├── rejection.rs        token acceptance logic
│       └── tree.rs             tree decoding structure
└──
```


## 4. Engine Code Walkthrough

This section follows the request lifecycle through the engine codebase, from the
top-level data structures down to prefill, decode, and sampling.

### 4.1 `Engine` and `ModelExecutor`

`Engine` is the stateful core — it owns the model weights, KV cache, and tokenizer.
`ModelExecutor` holds the model and its kernel dispatch table (`&'static dyn Ops`).

```rust
// engine/engine.rs

pub struct ModelExecutor {
    pub model: Mutex<ModelVariant>,      // Box<dyn ModelForward>
    pub device: Device,
    pub dtype: DType,
    pub config: CommonModelConfig,
    pub ops: &'static dyn Ops,          // kernel dispatch table (rms_norm, matmul, attention, ...)
}

pub struct Engine {
    pub executor: ModelExecutor,
    pub cache: CacheManager,
    pub tokenizer: Tokenizer,
    pub model_id: String,
    pub eos_token_ids: Vec<u32>,
    pub engine_config: EngineConfig,
}
```

`Engine::forward_batch` is the single dispatch point for all inference work:

```rust
// engine/engine.rs

pub fn forward_batch(&self, batch: ForwardBatch) -> Result<ModelOutput, EngineError> {
    match batch {
        ForwardBatch::Prefill { items } =>
            self.forward_prefill(items),          // allocate KV blocks + varlen prefill + first token
        ForwardBatch::Decode { tokens, positions, block_tables, deltanet_slots } =>
            self.forward_decode(tokens, positions, block_tables, deltanet_slots),
        ForwardBatch::OneShot { token_groups, task } =>
            self.forward_oneshot(token_groups, task),  // classify / embed
    }
}
```

### 4.2 Tokenization (`engine/tokenizer.rs`)

Before a request enters the scheduler it is tokenized and validated.
`tokenize_prompt_input` dispatches on input type — raw text is encoded with special
tokens; pre-tokenized IDs are passed through directly:

```rust
// engine/tokenizer.rs

fn tokenize_prompt_input(tokenizer: &Tokenizer, input: &PromptInput) -> Result<Vec<u32>> {
    match input {
        PromptInput::Text(text) => tokenizer.encode_with_special_tokens(text),
        PromptInput::TokenIds(ids) => Ok(ids.clone()),
    }
}
```

For batch classification / embedding inputs the same function is called per text and
token counts are accumulated:

```rust
fn tokenize_batch_inputs(tokenizer, inputs) -> Result<(Vec<Vec<u32>>, u32)> {
    match inputs {
        ClassificationInputs::Texts(texts) =>
            texts.iter().map(|t| tokenize_text(tokenizer, t)).collect(),
        ClassificationInputs::TokenIds(ids) =>
            Ok((ids.clone(), ids.iter().map(|v| v.len() as u32).sum())),
    }
}
```

`prepare_generate_request` (called by `ScheduledEngine` before enqueue) tokenizes,
validates against max context length, and builds the `LogitsProcessor` for the request:

```rust
// engine/model_runner/generate.rs

pub fn prepare_generate_request(engine, request) -> Result<PreparedGenerateRequest> {
    let tokens = tokenize_prompt_input(&engine.tokenizer, &request.prompt)?;
    validate_prompt_len(&tokens, engine.engine_config.max_model_len)?;
    let sampling = build_sampling(&request);     // ArgMax | TopK | TopP | TopKThenTopP
    let logits_processor = LogitsProcessor::new(sampling, request.seed);
    Ok(PreparedGenerateRequest { tokens, logits_processor, .. })
}
```

### 4.3 Scheduling and Dispatch (`engine/scheduled.rs`, `engine/run/ar.rs`)

`ScheduledEngine` is a thin async handle. It holds only the channel sender and the
loop's `JoinHandle`:

```rust
// engine/scheduled.rs

pub struct ScheduledEngine {
    ar_tx: mpsc::UnboundedSender<ArMessage>,
    executor: Arc<dyn Executor>,
    engine: Arc<Engine>,
    _ar_loop_handle: tokio::task::JoinHandle<()>,
}

impl ScheduledEngine {
    pub fn new(engine: Engine, config: SchedulerConfig) -> Self {
        let engine = Arc::new(engine);
        let executor = create_executor(engine.clone());
        let (tx, rx) = mpsc::unbounded_channel();
        let handle = tokio::spawn(ar_loop(engine.clone(), executor.clone(), config, rx));
        Self { ar_tx: tx, executor, engine, _ar_loop_handle: handle }
    }
}
```

`ar_loop` is a single `async fn` that runs for the lifetime of the engine.
Its body is a six-phase loop:

```rust
// engine/run/ar.rs

async fn ar_loop(engine, executor, config, mut rx) {
    let mut scheduler = ArScheduler::new(config);
    let mut states: HashMap<String, ArSequenceState> = HashMap::new();

    loop {
        // 1. Wait — suspend until at least one request arrives (no busy-polling when idle)
        if !scheduler.has_work() {
            match rx.recv().await {
                Some(msg) => handle_message(msg, &mut scheduler, &mut states),
                None => break,   // all senders dropped → shutdown
            }
        }

        // 2. Drain — collect any other pending messages without blocking
        while let Ok(msg) = rx.try_recv() {
            handle_message(msg, &mut scheduler, &mut states);
        }

        // 3. Schedule — ArScheduler picks which seqs to prefill / decode this step
        let step = scheduler.schedule_step();

        // 4. Build & Submit — assemble ForwardBatch, send to device worker (non-blocking)
        let batch = build_forward_batch(&mut states, &step);
        let handle = executor.submit(batch)?;

        // 5. Collect — await device completion (tokio yields here; device runs concurrently)
        let output = handle.recv().await?;

        // 6. Process — sample, check stop conditions, stream deltas, free finished seqs
        process_output(&engine, &mut scheduler, &mut states, &step, output);

        if !rx_open && !scheduler.has_work() { break; }
    }
}
```

After `process_output`, `process_single_token` runs for every sequence that produced
a token — it evaluates all stop conditions in order and streams the incremental text:

```rust
fn process_single_token(engine, scheduler, state, token, ..) {
    if engine.eos_token_ids.contains(&token)    { completed = true; }
    if state.stop_token_ids.contains(&token)    { completed = true; }

    state.output_tokens.push(token);
    state.emit_text_delta();                     // decode incremental text → StreamEvent::Token
    state.next_decode_position += 1;

    if state.output_tokens.len() >= state.max_new_tokens { completed = true; }
    for stop in &state.stop_strings {
        if state.current_text().contains(stop)  { completed = true; }
    }

    if completed { finish_state(engine, state, reason); }
}
```

### 4.4 Prefill Planning (`engine/planner.rs`)

Before any prefill forward pass the engine builds an allocation plan that resolves
prefix cache hits and computes exactly how many new KV blocks each request needs.

**Prefix reuse** is computed across the whole batch first:

```rust
// engine/planner.rs

fn build_prefix_reuse_candidate(engine, items) -> PrefixReuseCandidate {
    // Find the longest common prefix across all requests in this batch
    let common_tokens = find_common_prefix(items.iter().map(|i| &i.prompt_tokens));
    // Validate alignment: cached_len must be block_size-aligned and < min_prompt_len
    engine.resolve_paged_prefix_reuse(common_tokens, min_prompt_len)
}
```

**Block allocation** turns the prefix reuse candidate into a concrete per-request
plan of `(total_blocks_needed, new_blocks_needed, cached_len)`:

```rust
fn build_cache_allocation_entries(items, candidate, block_size) -> Vec<AllocationEntry> {
    items.iter().map(|item| {
        let cached_len     = candidate.cached_len;
        let total_blocks   = item.prompt_tokens.len().div_ceil(block_size);
        let new_blocks     = total_blocks - cached_len / block_size;  // only allocate the suffix
        AllocationEntry { total_blocks, new_blocks, cached_len }
    }).collect()
}
```

`allocate_block_tables_from_plan` locks `BlockManager`, increments ref-counts on
shared prefix blocks, and allocates new blocks for each request's suffix:

```rust
fn allocate_block_tables_from_plan(engine, items, plan) -> Vec<Vec<u32>> {
    let mut bm = engine.cache.block_manager.lock();
    items.iter().zip(&plan).map(|(item, entry)| {
        let mut table = shared_prefix_blocks.clone();   // reuse; ref-count incremented
        for _ in 0..entry.new_blocks {
            table.push(bm.allocate());                  // fresh block for the suffix
        }
        table
    }).collect()
}
```

### 4.5 Paged Prefill (`engine/model_runner/paged_prefill.rs`)

`batch_prefill_paged` takes a batch of prepared requests and returns one
`BatchPrefillResult` per request containing the first sampled token and the
allocated block table. It does five things in sequence:

**1. Resolve prefix reuse and allocate block tables** (via planner):

```rust
// engine/model_runner/paged_prefill.rs

let candidate   = engine.build_prefix_reuse_candidate(&items);
let plan        = build_cache_allocation_plan(&items, &candidate, block_size);
let block_tables = engine.allocate_block_tables_from_plan(&items, &plan);
```

**2. Build packed varlen input** — only the uncached suffix tokens are forwarded;
`cu_seqlens_q` covers suffix lengths while `cu_seqlens_k` covers full context lengths
(so attention can see the cached prefix via paged KV):

```rust
let packed_tokens: Vec<u32> = items.iter().map(|item|
    item.prompt_tokens[cached_len..]   // skip the prefix cache hit
).flatten().collect();

// cu_seqlens_q  =  cumulative suffix lengths  (what the model computes Q for)
// cu_seqlens_k  =  cumulative full lengths    (what the model attends over)
```

**3. Compute slot_mapping** — maps each suffix token to its physical KV cache slot:

```rust
// For token at position `pos` in request `i`:
let slot = block_manager.slot(&block_tables[i], cached_len + pos, block_size);
// slot_mapping[flat_index] = slot
```

**4. Run the forward pass** with paged KV context:

```rust
let ctx = BatchAttnContext {
    cu_seqlens_q, cu_seqlens_k,
    position_ids,
    block_tables,   // (batch, max_blocks)
    slot_mapping,   // (total_suffix_tokens,)
    deltanet_slots, // one slot per hybrid-model request, if any
};
let logits = model.lock().forward(packed_tokens, &ctx)?;
// last_token_select() picks the last token's logit row per request
```

**5. Sample the first output token** — greedy fast path uses a single GPU argmax
over the whole batch; stochastic requests fall back to per-request CPU sampling:

```rust
if plan.all_greedy {
    logits.argmax(-1)   // single GPU kernel for the whole batch
} else {
    items.iter().enumerate().map(|(i, item)| {
        item.logits_processor.sample(logits.get(i).to_f32())
    }).collect()
}
```

### 4.6 Paged Decode (`engine/model_runner/paged_decode.rs`)

There are two decode entry points depending on context:

**`batch_decode_paged`** — called by the ar_loop for each decode step.
Every active sequence contributes exactly one query token (Q = 1):

```rust
// engine/model_runner/paged_decode.rs

fn batch_decode_paged(seqs: &[BatchDecodeSeq]) -> Result<Tensor> {
    // flat_tokens:   one token per sequence (their last sampled token)
    // cu_seqlens_q:  [0, 1, 2, ..., N]       each seq contributes Q=1
    // cu_seqlens_k:  cumulative context lens  variable per seq
    // position_ids:  [pos_0, pos_1, ..., pos_N-1]
    // block_tables:  (N, max_blocks) padded to longest sequence

    let logits = model.lock().forward(flat_tokens, &ctx)?;
    Ok(logits.squeeze(1))   // (N, vocab_size)
}
```

**`batched_stream_decode`** — used when the executor runs prefill + full decode
in one call (multi-token path in `generate.rs`). It manages per-request state
across decode iterations and streams tokens back to callers:

```rust
fn batched_stream_decode(engine, executor, prefill_results, requests) {
    // per-request state: block_table, output_tokens, position, finished, …
    let mut seq_states: Vec<DecodeSeqState> = init_from_prefill(prefill_results);

    loop {
        let active: Vec<_> = seq_states.iter().filter(|s| !s.finished).collect();
        if active.is_empty() { break; }

        // allocate a new block for any sequence crossing a block boundary
        for s in &mut active { maybe_allocate_block(s, &engine.cache); }

        // one GPU forward for all active sequences together
        let logits = batch_decode_paged(&active)?;

        // sample: batch argmax if all greedy, else per-request logits_processor
        let next_tokens = sample_batch(&seq_states, &active_ids, logits);

        for (s, token) in active.iter_mut().zip(next_tokens) {
            // EOS / stop / max_tokens checks → mark finished
            // stream incremental text via response channel
            process_single_token(engine, s, token);
        }
    }

    // free all KV blocks and deltanet slots before returning
    for s in &seq_states { release_resources(engine, s); }
}
```

### 4.7 Sampling (`engine/sampling/logits_processor.rs`)

`LogitsProcessor` wraps a seeded RNG and a `Sampling` strategy. The strategy is
chosen once per request during `prepare_generate_request`:

```rust
// engine/sampling/logits_processor.rs

pub enum Sampling {
    ArgMax,                                         // greedy — no RNG needed
    All      { temperature: f64 },                  // sample over full vocabulary
    TopK     { k: usize, temperature: f64 },
    TopP     { p: f64,   temperature: f64 },        // nucleus sampling
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,   // seeded → deterministic given same seed
    sampling: Sampling,
}
```

`sample` converts logits to a probability distribution with temperature scaling
(numerically stable via max subtraction) and then dispatches:

```rust
pub fn sample(&mut self, logits: &[f32]) -> u32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter()
        .map(|&l| ((l - max) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = probs.iter().map(|p| p / sum).collect();

    match &self.sampling {
        Sampling::ArgMax       => argmax(&probs),
        Sampling::All { .. }   => sample_multinomial(&mut self.rng, &probs),
        Sampling::TopK { k, .. }  => sample_topk(&mut self.rng, &probs, *k),
        Sampling::TopP { p, .. }  => sample_topp(&mut self.rng, &probs, *p),
        Sampling::TopKThenTopP { k, p, .. } => sample_topk_topp(&mut self.rng, &probs, *k, *p),
    }
}
```

Top-K uses `select_nth_unstable_by` (O(n) partition, not a full sort) to find the
k-th largest probability and then samples from indices above that threshold.
Top-P (nucleus) sorts descending, accumulates until the cumulative probability
reaches `p`, and zeroes out the tail before sampling.

### 4.8 `Executor` trait and device registration

`Executor` is the single boundary between device-agnostic scheduling and
device-specific kernel dispatch. The trait is minimal by design:

```rust
// engine/executor.rs

pub trait Executor: Send + Sync + 'static {
    fn submit(&self, batch: ForwardBatch) -> Result<ExecutionHandle, EngineError>;
}

pub struct ExecutionHandle {
    rx: oneshot::Receiver<Result<ModelOutput, EngineError>>,
}

impl ExecutionHandle {
    // Yields the tokio task while the device worker runs — enables double-buffering
    pub async fn recv(self) -> Result<ModelOutput, EngineError> {
        self.rx.await.unwrap_or_else(|_| Err(EngineError::Internal("worker dropped".into())))
    }
}
```

Device crates self-register at startup through a static registry. `create_executor`
picks the highest-priority backend whose `probe` and `supports` predicates both pass:

```rust
pub struct ExecutorBackend {
    pub name: &'static str,
    pub priority: u32,
    pub probe: fn() -> bool,           // runtime check (e.g. CUDA driver present)
    pub supports: fn(&Device) -> bool, // device type match
    pub create: ExecutorFactory,       // Arc<Engine> → Box<dyn Executor>
}

fn create_executor(engine: Arc<Engine>) -> Box<dyn Executor> {
    EXECUTOR_REGISTRY.lock()
        .iter()
        .filter(|b| (b.probe)() && (b.supports)(&engine.executor.device))
        .max_by_key(|b| b.priority)
        .map(|b| (b.create)(engine.clone()))
        .unwrap_or_else(|| Box::new(StubExecutor))   // StubExecutor returns Unavailable
}
```

## 5. GPU Queue

All GPU-bound work is serialized through a single FIFO queue (`runtime/gpu_queue.rs`)
consumed by one dedicated OS thread (not a tokio task -- avoids `spawn_blocking`
overhead, critical when per-token decode is only a few milliseconds).

**GpuPacket variants:**

| Variant          | Description                                  | Producer           |
|------------------|----------------------------------------------|--------------------|
| `GenerateBatch`  | Prefill-only generation (max_new=1)          | Batch runtime      |
| `PrefillPaged`   | Varlen prefill + paged KV write              | Continuous runtime  |
| `DecodePaged`    | Batch decode with paged KV (Q=1 per seq)     | Continuous runtime  |
| `ClassifyBatch`  | Classification forward pass                  | Batch runtime      |
| `EmbedBatch`     | Embedding forward pass                       | Batch runtime      |

## 6. Attention Backends

Modular dispatch lives exclusively in `models/common/attn/mod.rs`. Model code
has zero `#[cfg]` gates for attention -- adding a backend means one file in
`attn/` and one dispatch branch in `mod.rs`.

| Backend | Feature flag | GPU requirement | Key capabilities |
|---|---|---|---|
| FA4 | `flash-attn-v4` | SM80+ | Prefill + paged decode, AOT CuTeDSL |
| FlashInfer | `flashinfer` | SM80+ (FA2) / SM90+ (FA3) | All attention paths, CUDA graph (32 graphs), plan caching |
| FA3 | `flash-attn-v3` | SM90 (Hopper) | Legacy, replaced by FlashInfer |
| FA2 | `flash-attn` | SM80+ | Legacy, replaced by FlashInfer |
| CPU | (always available) | None | Tiled BF16 (AVX-512) + F32 matmul SDPA |

**Dispatch priority:** FA4 -> FlashInfer -> FA3 -> FA2 -> CPU.
Recommended GPU build: `flashinfer-v4,onednn,deepgemm` (~98MB binary).

## 7. GEMM Backends

| Backend   | Feature flag | Target   | Notes                                         |
|-----------|-------------|----------|-----------------------------------------------|
| cuBLAS    | `cuda`      | GPU      | Default GPU GEMM                              |
| DeepGEMM  | `deepgemm`  | SM90+    | BF16, replaces cuBLAS. Decode 17%-2x faster   |
| oneDNN    | `onednn`    | CPU      | BF16 + F32 GEMM, packed weights, static link  |
| Built-in  | (default)   | CPU      | Fallback F32 GEMM when oneDNN is absent       |

## 8. KV Cache

**Paged KV cache** with `BlockManager` (vLLM-style). Block size is auto-tuned
for the active attention backend (128 with FlashInfer/FA3, 16 otherwise) and
overridable via `PRELUDE_PAGED_BLOCK_SIZE`.

**KV cache write**: custom vectorized PTX kernel (`scatter_kv_cache_flash` in
`ops/gpu/kv_cache.rs`), 128-bit float4 loads/stores.

**Prefix cache** (`prefix_cache.rs`): hash-trie structure that matches incoming
prompts against cached token blocks using hash chains. LRU eviction of leaf
blocks. Ref-counted integration with `BlockManager` so cached blocks survive
across requests.

**CUDA graph decode** (`cuda_graph.rs`): Optional graph capture for decode steps.
FlashInfer mode: 32 graphs (no seqlen bucketing), ~750ms warmup. Pre-allocated
metadata buffers for address stability across capture/replay.

## 9. CPU Optimization

Two layers, both BF16-focused:

**Pure Rust kernels (`cpu_ops/`)** -- zero external dependencies:
- RMSNorm, fused Add+RMSNorm, SiLU*Mul, RoPE
- FlashAttention-style tiled prefill + decode (online softmax)
- Runtime AVX-512 detection with scalar fallback
- NUMA-aware rayon pool with physical core binding

**oneDNN (feature `onednn`)** -- BF16/F32 GEMM only:
- Pre-packed weights (`PackedWeight::pack`) for amortized matmul
- Primitive cache keyed by (M,K,N) avoids JIT recompilation
- Static linking adds ~33MB to binary, auto-built via CMake

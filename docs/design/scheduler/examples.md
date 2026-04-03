# Scheduler Examples

Concrete scenarios showing how the scheduler design handles real workloads.
See [main design doc](README.md) for architecture and implementation details.

## Example 1: AR Serving — Qwen3-32B with Prefix Caching

Multi-turn chat serving. System prompt shared across all requests. Shows prefix cache hit flow.

```
Request 1: [system_prompt (2000 tokens)] + "What is Rust?"
Request 2: [system_prompt (2000 tokens)] + "Explain lifetimes"
Request 3: [system_prompt (2000 tokens)] + "What is ownership?"
```

```
Step 1: Request 1 arrives
  prefix_cache.match_prefix([system_prompt, "What is Rust?"]) → ([], 0)
  No cache hit. Allocate 128 blocks (2000 + prompt tokens / 16).
  Schedule full prefill: num_new_tokens = 2006.

Step 2: Request 1 finishes prefill, starts decode
  Insert [system_prompt, "What is Rust?", ...output...] into prefix cache.
  Prefix cache tree: root → [system_prompt (125 blocks)] → ["What is Rust?" (1 block)]

Step 3: Request 2 arrives
  prefix_cache.match_prefix([system_prompt, "Explain lifetimes"]) → (125 blocks, 2000 tokens)
  Cache hit! 2000 tokens already computed. Share 125 blocks (ref_count++).
  Only need to prefill "Explain lifetimes" (2 tokens). num_new_tokens = 2.
  Radix tree splits: root → [system_prompt] → ["What is Rust?", ...]
                                             → ["Explain lifetimes"]

Step 4: Request 3 arrives
  prefix_cache.match_prefix([system_prompt, "What is ownership?"]) → (125 blocks, 2000 tokens)
  Same cache hit. 2000 tokens skipped. Prefill only "What is ownership?" (3 tokens).
```

Savings: requests 2 and 3 skip 99.8% of prefill compute.
Without radix tree (vLLM's hash approach), this still works for exact block matches.
The radix tree advantage shows with partial prefix overlap (e.g., few-shot examples where
the examples differ but share a common introduction).

## Example 2: AR Serving — Chunked Prefill with Decode Interleaving

Long-context request arrives while decode requests are running.

```
Running: [req_A (decode, 1 token), req_B (decode, 1 token)]
Waiting: [req_C (prefill, 32000 tokens)]
Config: chunked_prefill_threshold = 8192, max_tokens_per_step = 16384
```

```
Step 1:
  Phase 1 (running): schedule req_A (1 tok) + req_B (1 tok) = 2 tokens
  Phase 2 (waiting): req_C has 32000 tokens, cap at 8192. Budget left = 16384 - 2 = 16382.
    Schedule req_C chunk 1: 8192 tokens.
  Total batch: 8194 tokens (2 decode + 8192 prefill chunk).
  req_C.num_computed_tokens = 8192, status = Running.

Step 2:
  Phase 1 (running): req_A (1) + req_B (1) + req_C (8192 more, capped at 8192) = 8194 tokens
  req_C.num_computed_tokens = 16384

Step 3:
  Phase 1: req_A (1) + req_B (1) + req_C (8192 more) = 8194 tokens
  req_C.num_computed_tokens = 24576

Step 4:
  Phase 1: req_A (1) + req_B (1) + req_C (7424 remaining) = 7426 tokens
  req_C.num_computed_tokens = 32000. Prefill complete, enters decode.

Step 5+:
  All three requests decode together: 3 tokens per step.
```

Decode requests (A, B) are never starved. The long prefill is chunked and interleaved.

## Example 3: AR Serving — Speculative Decoding (EAGLE)

Draft model proposes 4 tokens, target model verifies.

```
Running: [req_X (decode phase, num_computed = 1050)]
Draft model: small Llama-68M
Target model: Qwen3-32B
```

```
Step 1: Draft phase
  Draft model generates 4 tokens for req_X: [t1, t2, t3, t4]
  req_X.draft_token_ids = [t1, t2, t3, t4]

Step 2: Verify phase (standard step())
  req_X.num_new_tokens() = 1050 + 1 (output) + 4 (drafts) - 1050 (computed) = 5
  Schedule req_X with token_ids = [last_accepted, t1, t2, t3, t4]
  Block allocation: may need 1 new block for 5 tokens.
  Total batch: 5 tokens.

Step 3: update()
  Model returns sampled = [s0, s1, s2] (3 tokens accepted, t3 rejected)
  num_accepted = 3 - 1 = 2 (t1, t2 accepted)
  num_rejected = 4 - 2 = 2 (t3, t4 rejected)
  req_X.num_computed_tokens -= 2 (rewind from 1055 to 1053)
  req_X.output_tokens.extend([s0, s1, s2])
  req_X.draft_token_ids.clear()

Net gain: 3 tokens in 1 target forward pass (vs 1 token without speculation).
```

## Example 4: AR Serving — Preemption Under Memory Pressure

KV cache full, new high-priority request arrives.

```
Running: [req_A (4096 blocks, low priority), req_B (2048 blocks, medium priority)]
Waiting: [req_C (needs 3000 blocks, high priority)]
Free blocks: 500
Total blocks: 10000
```

```
Step 1: Phase 2 (waiting)
  req_C needs 3000 blocks, only 500 available.
  Cannot allocate. Need to preempt.

  select_preempt_victim(): req_B has fewer computed tokens → preempt req_B
  preempt(req_B): free 2048 blocks. Free = 2548. Still not enough.

  select_preempt_victim(): only req_A left → preempt req_A
  preempt(req_A): free 4096 blocks. Free = 6644.

  Allocate 3000 blocks for req_C. Success.
  req_C enters Running.

Step 2+:
  req_B re-enters waiting queue (prepended = high priority within FCFS).
  On next step, req_B gets scheduled. Prefix cache may recover some blocks.
  req_A waits until more blocks are freed.
```

## Example 5: Diffusion — Flux Image Generation Batch

4 concurrent image generation requests with different step counts.

```
Active: []
Waiting: [img_A (20 steps), img_B (30 steps), img_C (20 steps), img_D (50 steps)]
max_concurrent: 3 (GPU memory limit for 1024x1024 latents)
```

```
Step 0: Fill active slots
  Active: [img_A, img_B, img_C]. img_D waits.
  DiffusionScheduledBatch: 3 jobs, timestep = schedule[0] for each.
  Model runner: batch forward with 3 latent tensors (or 6 with CFG).

Steps 1-19: All three advance
  Each step: 3 DiT forward passes (batched).

Step 20: img_A and img_C finish (20 steps done)
  Active: [img_B]. Slots open.
  img_D enters active: Active: [img_B, img_D].

Steps 21-29: img_B and img_D advance
  Each step: 2 DiT forward passes.

Step 30: img_B finishes
  Active: [img_D].

Steps 31-49: img_D alone
  Each step: 1 DiT forward pass.

Step 50: img_D finishes. Pipeline empty.
```

No KV cache, no prefix caching, no preemption. Just a job queue with GPU memory as the constraint.

## Example 6: TTS — Qwen3-Omni Streaming Pipeline

User sends text, receives streaming audio. Four-stage pipeline with overlap.

```
Input: "Hello, how are you doing today?"
Stages: Thinker (LLM) → Talker (codec AR) → Code Predictor → Code2Wav
Chunk size: 10 tokens
```

```
Time 0-50ms: Thinker prefills text, emits first hidden state chunk (10 tokens)
  thinker_to_talker buffer: [chunk_0]

Time 50-80ms: Talker consumes chunk_0, produces layer-0 codec codes
  Thinker emits chunk_1.
  talker_to_predictor buffer: [codes_0]

Time 80-100ms: Code Predictor fills remaining RVQ layers for codes_0
  Talker consumes chunk_1, produces codes_1.
  Thinker emits chunk_2.
  predictor_to_vocoder buffer: [full_codes_0]

Time 100-110ms: Code2Wav decodes full_codes_0 → first audio chunk (~100ms of speech)
  → Stream to client. First-audio latency: ~110ms.

Time 110ms+: Pipeline runs at steady state.
  Each stage processes its current chunk while downstream stages process earlier chunks.
  Audio streams continuously at ~real-time or faster.
```

The key insight: the TTS pipeline reuses `ArScheduler` for the AR stages (Thinker, Talker,
Code Predictor). Only the inter-stage streaming buffers and the pipeline orchestrator are new.
Code2Wav doesn't need a scheduler at all — it's a pure function (codes in → audio out).

## Example 7: Embedding — BGE OneShot Processing

Embedding 10000 documents. Pure throughput, no latency constraint.

```
Config: max_tokens_per_batch = 32768
Documents: 10000 documents, average 200 tokens each
```

```
Step 1: batch = docs[0..160] (160 docs × 200 tokens ≈ 32000 tokens)
  OneShotPlan: 160 requests, total_tokens = 32000
  Model forward: one pass, no KV cache.
  Output: 160 embedding vectors.

Step 2: batch = docs[160..320]
  ...

Step 63: batch = docs[9920..10000]
  Last batch: 80 docs, 16000 tokens. Under budget, that's fine.
  Output: 80 embedding vectors.
```

Total: 63 forward passes for 10000 documents. No scheduling complexity.

## Example 8: DLLM — LLaDA2 Block-Level Demasking with KV Cache

LLaDA2 generating text in blocks of 32 tokens. Each block goes through multiple demasking
rounds. Prefix KV is cached and reused across rounds — only the demasking block is recomputed.

```
Request: "The capital of France is" + generate 64 tokens
Block size: 32
```

```
Block 0 (tokens 0-31):
  Sequence: [prompt (6 tokens)] + [32 × MASK]
  Prefix: [prompt] → KV cached via PrefixCache

  Round 0:
    Forward: prefix KV from cache (6 tokens, no recompute) + block prefill (32 tokens)
    Predict all [MASK] positions. Accept high-confidence: "Paris", ","
    Tokens: [prompt] + ["Paris", ",", MASK, MASK, ..., MASK]

  Round 1:
    Forward: prefix KV from cache (6 tokens) + block recompute (32 tokens, 2 tokens changed)
    More tokens accepted: "a", "city", "known", "for"
    ...

  Round N: All 32 masks replaced. Block 0 complete.
    Insert [prompt + block_0 tokens] into PrefixCache.
    block_0 KV blocks become part of cached prefix.

Block 1 (tokens 32-63):
  Sequence: [prompt + block_0 (38 tokens, all cached)] + [32 × MASK]
  Prefix: 38 tokens → KV from PrefixCache (zero recompute!)

  Round 0:
    Forward: prefix KV from cache (38 tokens) + block prefill (32 tokens)
    ...
  Round N: All masks replaced. Done.
```

Key points:
- **KV cache IS used.** Prefix (confirmed tokens) is cached via `BlockAllocator` + `PrefixCache`.
  Only the 32-token demasking block is recomputed each round.
- **Not the same as DiffusionScheduler.** Diffusion has no KV cache, no block allocator,
  no prefix caching. DLLM uses the same paged attention infrastructure as AR.
- **Not the same as ArScheduler.** AR generates 1 token per step. DLLM resolves 32 tokens
  through multiple demasking rounds, then moves to the next block.
- **Prefix grows across blocks.** After block 0 completes, its 32 tokens become cached prefix
  for block 1. The second block's forward skips 38 tokens of compute.

## Example 9: Disaggregated Serving — Prefill/Decode Separation

2 prefill workers + 4 decode workers serving Qwen3-32B. Shows the full request lifecycle
across workers, including prefix cache-aware routing.

```
Setup:
  Prefill workers: P0, P1 (high-FLOPS GPUs, small batch)
  Decode workers:  D0, D1, D2, D3 (high-bandwidth GPUs, large batch)
  Coordinator knows: D0 has system_prompt cached, D1-D3 do not.
```

```
Step 1: Request arrives — "system_prompt + What is Rust?"
  Coordinator:
    - Check decode worker prefix caches:
      D0: prefix hit = 2000 tokens (system_prompt cached)
      D1-D3: prefix hit = 0
    - Route: prefill on P0 (least loaded), decode on D0 (best prefix hit)

Step 2: P0 prefills the request
  P0's ArScheduler: standard prefill, num_new_tokens = 2006
  After prefill: num_computed_tokens = 2006
  P0 initiates KV transfer → D0
  Request status on P0: Finished(Transferred)

Step 3: KV transfer P0 → D0
  Transfer layer: only send blocks for "What is Rust?" (6 tokens = 1 block)
  D0 already has system_prompt blocks cached (prefix cache hit).
  Transfer size: 1 block instead of 126 blocks. ~99% savings.

Step 4: D0 receives the request
  D0's ArScheduler: receive_transferred_request()
    - import_blocks([block_new]) — register the 1 new block
    - req.block_ids = [125 cached blocks] + [1 new block]
    - req.num_computed_tokens = 2006
    - req.preloaded_blocks = Some([block_new])
    - Push to waiting queue

Step 5: D0 starts decoding
  D0's ArScheduler.step():
    Phase 2: pick up the request, num_new_tokens = 1 (first decode token)
    Standard decode. No awareness of disaggregation.

Step 6+: D0 decodes until EOS
  Standard continuous batching on D0. This request is mixed with other
  decode requests on D0. No special handling.
```

Key points:
- **ArScheduler on P0 and D0 is the same code.** P0 just never enters decode; D0 just
  never enters prefill (for transferred requests).
- **Prefix cache-aware routing** saves 99% of KV transfer bandwidth for requests sharing
  system prompts. This is the coordinator's job, not the scheduler's.
- **KV transfer is opaque to the scheduler.** P0 calls `transfer.send()`, D0 receives
  via `receive_transferred_request()`. The transport (NVLink/RDMA/TCP) is invisible.
- **D0's decode batching is unchanged.** The transferred request joins the decode batch
  alongside locally-originated requests. No special code path.

## Example 10: Attention-FFN Disaggregation — Qwen3-MoE (128 Experts)

2 attention GPUs + 4 FFN GPUs serving Qwen3-MoE-A3B. Attention GPUs hold KV cache,
FFN GPUs hold expert weights. Shows the per-layer data flow.

```
Setup:
  Attention GPUs: A0, A1 (TP=2, hold KV cache + attention weights)
  FFN GPUs: F0, F1, F2, F3 (EP=4, 32 experts each, hold expert weights)
  Model: 36 layers, 128 routed experts, top-2 routing
```

```
Per-layer execution for one batch (e.g., 64 decode tokens):

1. Attention side (A0, A1):
   ArScheduler.step() → ScheduledBatch (64 tokens, same as non-AFD)
   Executor.execute():
     for each layer:
       a) residual_norm → QKV projection → RoPE → paged_attention → O projection
          (all on A0/A1, standard TP all-reduce between A0/A1)
       b) residual_norm
       c) modules::moe_layer(&h, &gate, &weights, &afd_config, ops):
          → gate.route(h): compute top-2 expert IDs per token (on A0/A1)
          → ops.comm.send(h, topk_ids, topk_weights → FFN pool)
          → ops.comm.recv(result ← FFN pool)
          → return result
       d) residual add

2. FFN side (F0-F3), running in parallel:
   run_ffn_follower() waits for AfdSignal::Forward
     for each layer:
       a) ops.comm.recv(h, topk_ids, topk_weights ← attention pool)
       b) Dispatch tokens to expert-owning ranks (all-to-all among F0-F3)
       c) ops.gemm.grouped_gemm on local 32 experts
       d) Combine results (all-to-all among F0-F3)
       e) ops.comm.send(result → attention pool)

3. After all 36 layers:
   Attention side: lm_head → sample → StepResult
   ArScheduler.update(result)
```

**Memory layout comparison (per GPU):**

```
Without AFD (TP=2, all on 2 GPUs):
  Per GPU: attention weights (2GB) + expert weights (24GB) + KV cache (54GB) = 80GB
  → KV cache limited to 54GB → max batch ~800 sequences

With AFD (2 attn GPUs + 4 FFN GPUs):
  Attention GPU: attention weights (2GB) + KV cache (78GB) = 80GB
  FFN GPU: 32 expert weights (12GB) + workspace (4GB) = 16GB
  → KV cache 78GB → max batch ~1150 sequences (+44%)
  → FFN GPUs can be smaller/cheaper (e.g., 24GB consumer vs 80GB datacenter)
```

**Micro-batch pipelining (optimization):**

```
Without pipelining (layer L):
  A: send(h) ──────────────────── recv(result) ──── [idle while FFN computes]
  F:          recv(h) ── compute ── send(result)

With 2 micro-batches (layer L):
  A: send(h_0) ─ send(h_1) ─── recv(r_0) ─ recv(r_1)
  F:     recv(h_0) ─ compute_0 ─ send(r_0) ─ recv(h_1) ─ compute_1 ─ send(r_1)

  Overlap: while F computes micro-batch 0, A sends micro-batch 1.
  Hides communication latency behind compute.
```

Key points:
- **ArScheduler is identical to non-AFD.** It produces the same `ScheduledBatch` for 64 decode tokens.
  The AFD communication happens inside `modules::moe_layer` during `model.forward()`.
- **FFN follower is not an ArScheduler.** It's `run_ffn_follower()` — a passive loop that
  waits for `AfdSignal::Forward` and runs FFN layers. No request management, no block allocation.
- **Model code is unchanged.** The Qwen3-MoE model calls `modules::moe_layer(...)` exactly
  as in non-AFD mode. The `MoeMode::Disaggregated` config is set at load time.
- **Heterogeneous hardware.** Attention GPUs need large memory (KV cache). FFN GPUs need
  compute (expert matmuls). This enables cost-efficient deployments with mixed GPU types.
- **Micro-batch pipelining** is a model runner optimization, not a scheduler concern.

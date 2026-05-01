# Continuous Batching MVP

## Context

The current goal is not to land a single `mixed forward` kernel in one shot. Instead,
the first step is to move multi-token generation from the batch-flush model to
iteration-level scheduling, and to allow new requests to be admitted while
existing decodes are still running.

Core changes that have landed in this step:
1. `ScheduledEngine` now routes requests to a scheduler by task type, rather
   than packing every task into a single generation loop.
2. Multi-token generation has its own `continuous-generation` runtime.
3. The runtime reuses the existing `batch_prefill_paged` and
   `batch_decode_paged` helpers instead of rewriting the execution layer.

## Routing Rules

- `classify` / `embed` → `batch-runtime`
- prepared generation with `max_new <= 1` → `batch-runtime`
- prepared generation with `max_new > 1` and paged decode available → `continuous-generation`
- prepared generation with `max_new > 1` but the required capability is missing → falls back to `batch-runtime`

## Current Shape

### 1. Batch Scheduler

The original batch-and-dispatch main loop is kept; its responsibilities are now:
- the stable batching path for classify/embed
- prefill-only generation
- the generation fallback when the continuous runtime is unavailable

### 2. Continuous Generation Runtime

A new, separate runtime that holds per-sequence state:
- prepared request
- response channel
- pending token
- block table
- prompt length / next decode position
- streamed text offset
- finish reason / token logprobs / timing

It uses `Scheduler` for iteration-level decisions:
- `Scheduler::add_request()`
- `Scheduler::schedule_step()`
- `Scheduler::on_token_generated()`
- `Scheduler::finish_request()`

### 3. Mixed Scheduler Step

A `SchedulerStep` can now carry both:
- `prefill_request_ids`
- `decode_request_ids`

That is, a single iteration can:
- admit new prefill requests
- continue decoding already-running sequences

This "mixed" step is **mixed at the scheduling layer**, not **mixed inside a single GPU forward**.

## Execution Model

Each tick of the continuous runtime roughly does the following:

```text
1. drain the channel, picking up new prepared generation requests
2. scheduler.schedule_step()
3. for requests admitted this tick, call batch_prefill_paged
4. for requests decoding this tick, call batch_decode_paged
5. sample token / stop detection / streaming emit
6. for finished requests, release the block table and deltanet slot
```

Key points:
- Prefill and decode can happen in the same scheduler iteration.
- But execution still goes through two helpers: one for prefill, one for decode.
- Streaming and non-streaming share the same state-advancement logic.

## Resource Handling

The resource policy in this MVP is deliberately conservative:
- Prefill admission first trims the batch against currently-free blocks so that
  `batch_prefill_paged` doesn't get overloaded.
- Active sequences get paged blocks appended on demand before decode.
- On block exhaustion, the runtime ends the affected sequence or defers new
  prefill, rather than doing scheduler-level preemption.

## Verification

- Default build: `cargo test -p prelude-core scheduler --lib`
- Paged-decode build: `cargo test -p prelude-core scheduler --lib --features paged-attn,flash-attn-v3`

Regression points covered:
- classify/embed still run on the batch scheduler.
- prepared prefill-only generation still runs on the batch scheduler.
- prepared multi-token generation runs on the continuous runtime.
- `SchedulerStep` can emit a mixed step while decodes are running.

## Current Limitations

This implementation is an MVP, not the final shape:

1. There is no single `forward_mixed_step()` yet, so prefill and decode are
   still not combined into a real single-pass GPU forward.
2. The continuous runtime is not yet wired to block-manager-driven preemption.
3. Resource admission is a coarse, conservative heuristic — not full
   cache-aware scheduling.
4. The continuous runtime only covers the paged-decode path.

## Next Steps

More sensible next steps:
- Converge prefill + decode into a true single-forward mixed executor.
- Wire the block manager and preemption into the continuous runtime.
- Free the scheduler's memory budgeting from the conservative approximation.

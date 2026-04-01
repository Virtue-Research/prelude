# Design Comparisons

How our scheduler design relates to vLLM and SGLang.
See [main design doc](README.md) for architecture and implementation details.

## What we take from vLLM V1

| Feature | vLLM's approach | Our approach | Why |
|---------|----------------|-------------|-----|
| Token-centric scheduling | `num_computed_tokens` tracking | Same | Clean, unified prefill/decode |
| Block-level prefix cache | SHA256 per-block hash table | Radix tree (from SGLang) | Better prefix sharing for multi-turn |
| SchedulerOutput | new/cached split with deltas | Flat `ScheduledBatch` | Simpler, no IPC overhead in Rust |
| Preemption | Full recompute, FCFS victim | Full recompute, min-waste victim | Same, but better victim selection |
| Chunked prefill | `long_prefill_token_threshold` | Same | Prevents head-of-line blocking |

## What we take from SGLang

| Feature | SGLang's approach | Our approach | Why |
|---------|------------------|-------------|-----|
| Radix tree prefix cache | Full radix with node splitting, eviction policies | Same structure, LRU-only eviction | Core innovation, but configurable eviction is premature |
| Cache-aware scheduling | LPM/DFS-weight policies | LPM as queue policy | Good idea, simple integration |
| Two-batch overlap | Explicit pipeline with FutureMap | Simpler: async execute + CPU step | Same benefit, less machinery |
| `new_token_ratio` dynamic budget | Decay on OOM, increase on success | Not adopted | Overcomplicates budget logic; fix OOM by preemption instead |

## What we discard from both

| Feature | Why discarded |
|---------|-------------|
| vLLM's KV connector coupling | 1600+ line scheduler with connector edge cases everywhere. External KV transfer should be a separate system that the scheduler calls, not woven into every scheduling branch |
| vLLM's multiple request status states | `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`, `WAITING_FOR_REMOTE_KVS`, `WAITING_FOR_STREAMING_REQ` — these are connector/grammar concerns leaked into core scheduler state machine. Our scheduler has 4 states: Waiting, Running, Preempted, Finished |
| SGLang's configurable eviction policies | LRU/LFU/FIFO/MRU/FILO/Priority/SLRU — 7 policies. LRU is sufficient. If evidence shows another policy is better, replace the implementation. Don't add config surface for hypothetical benefit |
| SGLang's `extra_key` namespace | LoRA cache isolation. Solve when we add multi-LoRA, not before |
| SGLang's FutureMap | Complex circular buffer for cross-batch speculative decode state. Speculative decode is a simpler problem in our design (draft tokens are just extra entries in the batch) |
| vLLM's encoder-specific scheduler paths | Encoder-decoder models add `scheduled_encoder_inputs`, `encoder_cache_manager`, `max_num_encoder_input_tokens`. We handle encoder outputs as pre-computed embeddings injected before the first layer (see ops dispatch doc Example 4), not as a scheduler concern |

# Workflows

Back to [main design doc](README.md).


End-to-end call flows for common scenarios. Each step annotated with the file it lives in.

### Workflow 1: AR Online Serving (Single Request Lifecycle)

A request arrives, gets prefilled, decoded, and returned.

```
1. Client sends POST /generate                          # (HTTP server, outside this doc)
      ↓
2. Engine::serve() dispatches to run::ar()              # engine/mod.rs → engine/run.rs
      ↓
3. run::ar() main loop:
   ┌─────────────────────────────────────────────────┐
   │ a. ArScheduler::step()                          │  # scheduler/ar.rs
   │    ├─ Phase 1: schedule running requests        │
   │    │   └─ BlockAllocator::allocate()            │  # scheduler/components/block_allocator.rs
   │    └─ Phase 2: schedule waiting requests        │
   │        ├─ RequestQueue::pop()                   │  # scheduler/components/request_queue.rs
   │        ├─ PrefixCache::match_prefix()           │  # scheduler/components/prefix_cache.rs
   │        └─ BlockAllocator::allocate()            │
   │    → produces ScheduledBatch                    │  # scheduler/types.rs
   │                                                 │
   │ b. ModelRunner::execute(&batch)                 │  # engine/model_runner/mod.rs
   │    ├─ build tensors (input_ids, cu_seqlens,     │
   │    │   block_tables, slot_mapping)              │
   │    ├─ ops.session.begin_forward()               │  # (ops dispatch layer)
   │    ├─ ops.session.precompute_paged_plan()       │
   │    ├─ model.forward()                           │
   │    ├─ ops.session.end_forward()                 │
   │    └─ sample() → StepResult                     │  # scheduler/types.rs
   │                                                 │
   │ c. ArScheduler::update(&result)                 │  # scheduler/ar.rs
   │    ├─ advance num_computed_tokens               │
   │    ├─ append output tokens                      │
   │    ├─ check stop conditions                     │
   │    └─ if finished:                              │
   │        ├─ PrefixCache::insert()                 │  # scheduler/components/prefix_cache.rs
   │        ├─ BlockAllocator::free()                │  # scheduler/components/block_allocator.rs
   │        └─ emit result to client                 │
   └─────────────────────────────────────────────────┘
      ↑ repeat until request finishes
```

### Workflow 2: AR Online Serving with Overlap

Same as Workflow 1, but GPU execution and CPU scheduling overlap.

```
   Step N                              Step N+1
   ┌──────────────────────────┐
   │ GPU: runner.execute(N)   │─────────────────────────→
   └──────────────────────────┘
                    ┌──────────────────────────────────┐
                    │ CPU: scheduler.update(N-1)       │  # scheduler/ar.rs
                    │       scheduler.step() → batch   │  # scheduler/ar.rs
                    └──────────────────────────────────┘
                                                         # engine/run.rs (overlap logic)
```

### Workflow 3: DLLM Block Demasking (One Block)

One demasking block resolved through multiple iterations.

```
1. DllmScheduler::step()                                # scheduler/dllm.rs
   ├─ PrefixCache::match_prefix(prefix_tokens)          # scheduler/components/prefix_cache.rs
   ├─ BlockAllocator::allocate(block_size)              # scheduler/components/block_allocator.rs
   └─ → DemaskingStepInput (tokens, mask_positions, block_table)

2. Demasking iterations (inside run::dllm):             # engine/run.rs
   ┌──────────────────────────────────────────────┐
   │ a. ModelRunner::execute()                    │     # engine/model_runner/mod.rs
   │    ├─ prefix KV from cache (no recompute)    │
   │    └─ block tokens: full prefill (32 tokens) │
   │                                              │
   │ b. DllmScheduler::update()                   │     # scheduler/dllm.rs
   │    ├─ accept high-confidence predictions     │
   │    ├─ replace [MASK] → predicted tokens      │
   │    └─ remove from mask_positions             │
   └──────────────────────────────────────────────┘
      ↑ repeat until mask_positions is empty

3. Block complete:                                      # scheduler/dllm.rs
   ├─ PrefixCache::insert(prefix + block_tokens)        # scheduler/components/prefix_cache.rs
   ├─ block_offset += block_size
   └─ start next block (or finish if all blocks done)
```

### Workflow 4: Diffusion Image Generation

```
1. DiffusionScheduler::step()                           # scheduler/diffusion.rs
   └─ → Vec<DiffusionStepInput> (latents, timestep, text_embeds)

2. run::diffusion():                                    # engine/run.rs
   ┌──────────────────────────────────────────────┐
   │ ModelRunner executes DiT forward:            │     # engine/model_runner/mod.rs
   │  ├─ CFG: duplicate latents (2x batch)        │
   │  ├─ dit.forward(latents, text_embeds, temb)  │     # (model code, ops dispatch layer)
   │  ├─ CFG: guided = uncond + scale*(cond-uncond)│
   │  └─ Euler step: latents += dt * guided       │
   └──────────────────────────────────────────────┘

3. DiffusionScheduler::update()                         # scheduler/diffusion.rs
   ├─ job.latents = new_latents
   ├─ job.current_step += 1
   └─ if current_step >= num_steps: job finished
```

### Workflow 5: P/D Disaggregated Serving

```
1. Coordinator routes request                           # disaggregated/pd/coordinator.rs
   ├─ find decode worker with best prefix cache hit
   ├─ pick least-loaded prefill worker
   └─ send request to prefill worker

2. Prefill Worker (normal ArScheduler):                 # scheduler/ar.rs + engine/run.rs
   ├─ ArScheduler::step() → prefill
   ├─ ModelRunner::execute() → forward
   ├─ ArScheduler::update() → FinishReason::Transferred
   └─ KvTransfer::send(block_ids) → decode worker      # disaggregated/pd/kv_transfer.rs

3. Decode Worker:                                       # scheduler/ar.rs
   ├─ receive_transferred_request()
   │   ├─ BlockAllocator::import_blocks()               # scheduler/components/block_allocator.rs
   │   └─ req.num_computed_tokens = prompt_len
   └─ normal decode loop (same as Workflow 1 step 3)
```

### Workflow 6: Adding a New Scheduler (Developer Guide)

Example: adding a `VideoScheduler` for video generation.

```
1. Create scheduler/video.rs                            # new file
   ├─ define VideoScheduler struct
   ├─ define VideoJob struct
   ├─ implement step() → VideoStepInput
   └─ implement update(VideoStepResult)
   Optionally use components/:
   ├─ BlockAllocator (if model uses KV cache)
   └─ RequestQueue (for request ordering)

2. Add run function in engine/run.rs                    # add run::video()
   └─ loop { scheduler.step() → runner.execute() → scheduler.update() }

3. Add mode to engine/config.rs                         # add Mode::Video
   └─ Engine::serve() dispatches to run::video()

Files touched: 3 (video.rs, run.rs, config.rs)
Files NOT touched: ar.rs, dllm.rs, diffusion.rs, oneshot.rs, tts.rs,
                   model_runner/, components/, any model code
```


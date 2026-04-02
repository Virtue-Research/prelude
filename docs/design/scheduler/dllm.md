# DLLM Scheduler (`scheduler/dllm.rs`)

Back to [main design doc](README.md).


Manages block-level iterative demasking for diffusion LLMs (LLaDA2).
**Uses KV cache** — the prefix (tokens before the current demasking block) is cached
via `BlockAllocator` + `PrefixCache`, and only the demasking block is recomputed each round.

```
Sequence: [prefix (confirmed tokens)] + [demasking block (32 tokens, some [MASK])]
           ↑ KV cache cached, reused      ↑ recomputed each demasking round
```

### Core State

```rust
// scheduler/dllm.rs

struct DllmScheduler {
    /// Active demasking jobs.
    active: Vec<DemaskingJob>,
    /// Waiting jobs.
    waiting: RequestQueue,
    /// KV cache block allocator (same component as ArScheduler uses).
    block_allocator: BlockAllocator,
    /// Prefix cache for KV reuse across demasking rounds.
    prefix_cache: PrefixCache,
    /// Config.
    block_size: usize,             // demasking block size (e.g., 32 for LLaDA2)
    max_concurrent: usize,
}

struct DemaskingJob {
    id: RequestId,
    tokens: Vec<u32>,              // prompt + current [MASK] / predicted tokens
    mask_positions: Vec<usize>,    // positions in current block that are still [MASK]
    block_offset: usize,           // start position of current demasking block
    block_ids: Vec<u32>,           // allocated KV cache blocks
    num_prefix_tokens: usize,      // tokens with cached KV (= block_offset)
    confidence_threshold: f32,
}
```

### Scheduling Loop

```rust
// scheduler/dllm.rs

impl DllmScheduler {
    fn step(&mut self) -> Vec<DemaskingStepInput> {
        // Fill active slots
        while self.active.len() < self.max_concurrent {
            let job_id = match self.waiting.pop() {
                Some(id) => id,
                None => break,
            };
            let job = &mut self.jobs[job_id];

            // Prefix cache lookup — reuse KV for already-confirmed tokens
            let (cached_blocks, num_cached) =
                self.prefix_cache.match_prefix(&job.tokens[..job.block_offset]);
            self.block_allocator.share(&cached_blocks);
            job.block_ids.extend_from_slice(&cached_blocks);
            job.num_prefix_tokens = num_cached;

            // Allocate blocks for demasking block
            let new_blocks = self.block_allocator.allocate(
                blocks_needed(job.block_size)
            ).unwrap();
            job.block_ids.extend_from_slice(&new_blocks);

            self.active.push(job_id);
        }

        // Each active job produces a step input
        self.active.iter().map(|&job_id| {
            let job = &self.jobs[job_id];
            DemaskingStepInput {
                job_id: job.id,
                token_ids: &job.tokens[job.block_offset..job.block_offset + self.block_size],
                block_table: &job.block_ids,
                num_prefix_tokens: job.num_prefix_tokens,
                mask_positions: &job.mask_positions,
            }
        }).collect()
    }

    fn update(&mut self, results: &[DemaskingStepResult]) {
        for result in results {
            let job = &mut self.jobs[result.job_id];

            // Replace high-confidence [MASK] with predicted tokens
            for (&pos, &token) in result.accepted.iter() {
                job.tokens[job.block_offset + pos] = token;
            }
            job.mask_positions.retain(|p| !result.accepted.contains_key(p));

            if job.mask_positions.is_empty() {
                // Block complete — confirmed tokens become prefix for next block
                self.prefix_cache.insert(
                    &job.tokens[..job.block_offset + self.block_size],
                    &job.block_ids,
                );
                job.block_offset += self.block_size;

                if job.block_offset >= job.tokens.len() {
                    // All blocks done — finish
                    self.finish_job(result.job_id);
                } else {
                    // Next block — new mask positions, allocate new KV blocks
                    job.mask_positions = (0..self.block_size).collect();
                    job.num_prefix_tokens = job.block_offset;
                    // ... allocate blocks for next demasking block ...
                }
            }
        }
    }
}
```

Key points:
- **Uses `BlockAllocator` and `PrefixCache`** — same standalone components as ArScheduler.
  Prefix KV is cached and reused across demasking rounds, only the block is recomputed.
- **Not a variant of ArScheduler.** The scheduling loop is fundamentally different: AR generates
  1 token per step, DLLM resolves a block of 32 tokens through multiple demasking iterations.
- **Not the same as DiffusionScheduler.** Diffusion has no KV cache. DLLM uses paged attention
  with cached prefix — closer to AR's infrastructure, different scheduling logic.
- **A DLLM researcher can read this scheduler end-to-end** without knowing how ArScheduler
  or DiffusionScheduler work. The only prerequisite is understanding `BlockAllocator` and
  `PrefixCache`, which are self-explanatory standalone components.


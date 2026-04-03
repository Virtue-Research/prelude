# OneShot Scheduler (`scheduler/oneshot.rs`)

Back to [main design doc](README.md).


Embedding, reranking, classify, prefill-only generation. Simplest scheduler. One forward pass per request, no KV cache, no decode loop, no
cross-step state. Covers embedding, reranking, classify, and prefill-only generation
(`max_new_tokens <= 1`).

```rust
// scheduler/oneshot.rs

struct OneShotScheduler {
    waiting: VecDeque<OneShotRequest>,
    max_tokens_per_batch: usize,
}

struct OneShotRequest {
    id: RequestId,
    tokens: Vec<u32>,
}

impl OneShotScheduler {
    fn step(&mut self) -> Option<OneShotPlan> {
        if self.waiting.is_empty() {
            return None;
        }

        let mut batch = Vec::new();
        let mut total_tokens = 0;

        while let Some(req) = self.waiting.front() {
            if total_tokens + req.tokens.len() > self.max_tokens_per_batch {
                break;
            }
            let req = self.waiting.pop_front().unwrap();
            total_tokens += req.tokens.len();
            batch.push(req);
        }

        Some(OneShotPlan { requests: batch, total_tokens })
    }
}
```

No block allocator, no prefix cache, no preemption, no state tracking.
One forward pass per batch, results returned immediately.


# Diffusion Scheduler (`scheduler/diffusion.rs`)

Back to [main design doc](README.md).


Manages denoising loops for image/video generation. No KV cache, no continuous batching.
The "batch" is a set of denoising jobs, each iterating through a fixed number of steps.

### Core State

```rust
// scheduler/diffusion.rs

struct DiffusionScheduler {
    /// Active denoising jobs.
    active: Vec<DenoisingJob>,
    /// Waiting jobs (not yet started).
    waiting: VecDeque<DenoisingJob>,
    /// Maximum concurrent jobs (limited by GPU memory for latents).
    max_concurrent: usize,
}

struct DenoisingJob {
    id: RequestId,
    current_step: usize,
    num_steps: usize,
    timesteps: Vec<f32>,           // [1.0, 0.95, ..., 0.0]
    latents: Tensor,               // current latent (GPU)
    text_embeds: Tensor,           // computed once, reused
    guidance_scale: f32,
    use_cfg: bool,
}
```

### Scheduling Loop

```rust
// scheduler/diffusion.rs

impl DiffusionScheduler {
    fn step(&mut self) -> Vec<DiffusionStepInput> {
        // Fill active slots from waiting queue
        while self.active.len() < self.max_concurrent {
            match self.waiting.pop_front() {
                Some(job) => self.active.push(job),
                None => break,
            }
        }

        // All active jobs advance one denoising step
        self.active.iter().map(|job| DiffusionStepInput {
            job_id: job.id,
            latents: &job.latents,
            text_embeds: &job.text_embeds,
            timestep: job.timesteps[job.current_step],
            dt: job.timesteps[job.current_step] - job.timesteps[job.current_step + 1],
            guidance_scale: job.guidance_scale,
            use_cfg: job.use_cfg,
        }).collect()
    }

    fn update(&mut self, results: &[DiffusionStepResult]) {
        for result in results {
            let job = self.active.iter_mut().find(|j| j.id == result.job_id).unwrap();
            job.latents = result.new_latents.clone();
            job.current_step += 1;
        }

        // Remove finished jobs
        self.active.retain(|job| job.current_step < job.num_steps);
    }
}
```

### Model Runner for Diffusion

```rust
// engine/run.rs (diffusion section)

fn execute_diffusion(plan: &DiffusionScheduledBatch, dit: &FluxDiT, ops: &OpsBundle) -> Vec<DiffusionStepResult> {
    ops.begin_forward();

    let results: Vec<_> = plan.jobs.iter().map(|input| {
        let latents = if input.use_cfg {
            // CFG: duplicate latents (conditional + unconditional)
            cat(&[&input.latents, &input.latents], 0).unwrap()
        } else {
            input.latents.clone()
        };

        let temb = timestep_embed(input.timestep);
        let noise_pred = dit.forward(&latents, &input.text_embeds, &temb, ops).unwrap();

        // CFG: guided = uncond + scale * (cond - uncond)
        let guided = if input.use_cfg {
            let (cond, uncond) = noise_pred.chunk(2, 0).unwrap();
            (&uncond + input.guidance_scale * &(&cond - &uncond).unwrap()).unwrap()
        } else {
            noise_pred
        };

        // Euler step
        let new_latents = (&input.latents + input.dt * &guided).unwrap();
        DiffusionStepResult { job_id: input.job_id, new_latents }
    }).collect();

    ops.end_forward();
    results
}
```

Key points:
- **No block allocator, no prefix cache.** Diffusion has no KV cache.
- **Batch = all active denoising jobs.** Each job is independent.
- **CFG is model runner logic, not scheduler logic.** The scheduler doesn't know about CFG —
  it just passes the `guidance_scale` to the model runner, which handles the 2x batch.
- **Fixed step count.** Each job runs exactly `num_steps` iterations, then finishes.
- **GPU memory is the only constraint.** `max_concurrent` is set based on latent tensor size.


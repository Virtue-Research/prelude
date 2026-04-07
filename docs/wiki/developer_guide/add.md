# Adding a Model


## Common Contribution Paths

### Adding a new model
See [Adding a Model](adding-a-model.md) for a step-by-step guide. The short version: implement `ModelForward` in `prelude-core/src/models/`, register via the `model_config!` macro, and call `ops.xxx()` for all compute — no device-specific code in model files.

### Adding a new attention backend
Add a file in `prelude-cuda/src/attn/` implementing `AttentionOps`. Add one dispatch branch in `attn/mod.rs`. No changes needed in model code.

### Adding a new device backend
Create a new crate (e.g. `prelude-newdevice/`) that:
1. Implements the 9 op traits (`TensorOps`, `AttentionOps`, `GemmOps`, etc.) as a struct
2. Implements `Executor` for submitting scheduled batches
3. Calls `register()` from `lib.rs` with a priority so the engine picks it up automatically

### Modifying the scheduler
The scheduler in `prelude-core/src/scheduler/ar.rs` (for AR LLMs) is pure CPU — no GPU calls. Sequence state machine: `Waiting → Prefilling → Decoding → Finished` (with preemption back to `Waiting`). Budget constraints (`max_running_requests`, `max_prefill_tokens`, `max_total_tokens`) are all configurable via CLI.

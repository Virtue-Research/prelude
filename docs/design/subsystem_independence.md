## Subsystem Independence

Each subsystem can be developed by one person who only knows the trait signatures (shared contract)
and their own internals. No cross-subsystem code reading required.

| Subsystem | Needs to know | Does NOT need to know |
|-----------|--------------|----------------------|
| **Model impl** (Qwen3, Flux, TTS) | Module APIs (`Linear`, `residual_norm`, ...), `Ops` bundle | Any device impl, kernel library, engine |
| **Modules** (Linear, residual_norm, gated_mlp, ...) | Op trait signatures, `FusedOps` match pattern | Device internals, model specifics |
| **Device crate** (CudaOps+Executor, RocmOps+Executor, ...) | Op traits, Executor trait, kernel library APIs | Model code, other devices |
| **Kernel wrapper** (FA4, FlashInfer, DeepGEMM) | Kernel library C API | Op traits, model code, other wrappers |
| **Comm backend** (NCCL, RCCL, XLA coll.) | `CommOps` trait, communication library API | Model code, attention kernels |
| **Engine/Scheduler** | `Executor` trait, `OpsSession`, model `forward()` signature | Kernel implementations, device internals |
| **KV Cache Manager** | Block allocation logic, `block_tables`/`slot_mapping` layout | Attention kernels, device code |

**Three design choices that enable this:**

1. **`FusedOps` default methods** — adding a new fusion only touches the trait definition (1 line with `{ None }` default) and the device that implements it. Other devices don't change. Model developer adds call site + fallback. No cross-team coordination needed beyond agreeing on the method signature.

2. **Tensor layout conventions** — formalized in the doc (above), not just comments. Every device implementation accepts and returns canonical layouts. Device-internal transformations (TPU 128-byte padding, Metal transpose) are invisible to callers.

3. **Graph capture is Executor-internal** — CUDA graph capture/replay lives in `CudaExecutor`, not in `OpsSession` or the engine's run loops. The engine calls `executor.submit()` / `executor.collect()` — whether that internally uses graph replay or individual kernel launches is the Executor's decision. Model code and schedulers never know.

## Session Lifecycle

Some devices have per-forward-pass state that must be managed:
- **FlashInfer**: plan cache (expensive scheduling, computed once, reused across N layers)
- **XLA (TPU)**: compilation cache, trace-based execution
- **CUDA/HIP graphs**: pre-allocated buffers with fixed GPU addresses

```rust
// prelude-core/src/ops/traits/session.rs

trait OpsSession: Send + Sync {
    /// Initialize per-forward-pass state. Called before model.forward().
    fn begin_forward(&self);

    /// Clear per-forward-pass state. Called after model.forward().
    fn end_forward(&self);

    /// Pre-compute paged attention scheduling for the current batch.
    /// Converts block_tables → kernel-specific metadata (e.g., FlashInfer indptr/indices).
    /// Called once before model.forward(), reused across all N layers.
    ///
    /// Same `&[u32]` convention as attention params — device uploads internally.
    fn precompute_paged_plan(
        &self,
        block_tables: &[u32],       // [batch * max_blocks_per_seq]
        cu_seqlens_k: &[u32],       // [batch + 1]
        block_size: usize,
    ) -> Result<()>;
}
```

**Devices without session state** (CPU, Metal, Vulkan) implement all methods as no-ops.

### Graph Capture (CUDA/HIP-internal concern)

CUDA graphs require pre-allocated buffers at fixed GPU addresses. This is a CUDA-specific
optimization that does NOT belong in the shared `OpsSession` trait — the engine doesn't
need to know about it.

Instead, `CudaOps` exposes a **device-specific** graph capture API. The engine's CUDA
graph runner downcasts to `CudaOps` (since it already knows it's on CUDA) and calls
device-specific methods:

```rust
// prelude-cuda/src/graph.rs — CUDA-specific extension, not in OpsSession trait.

impl CudaOps {
    /// Allocate pre-sized GPU buffers for graph metadata.
    pub fn allocate_graph_buffers(&self, max_batch: usize, max_blocks: usize) -> GraphMetaBuffers;

    /// Pre-compute paged plan into fixed-address graph buffers (for capture/replay).
    /// Same &[u32] input — CudaOps memcpys into graph_buffers' fixed-address GPU tensors.
    pub fn precompute_paged_plan_graphed(
        &self,
        block_tables: &[u32],
        cu_seqlens_k: &[u32],
        block_size: usize,
        graph_buffers: &GraphMetaBuffers,
    ) -> Result<()>;
}

/// Pre-allocated GPU tensors with fixed addresses for CUDA graph replay.
struct GraphMetaBuffers {
    pub indptr: Tensor,
    pub indices: Tensor,
    pub last_page_len: Tensor,
}
```

**CUDA graph flow** (engine's CUDA graph runner, not generic engine):

```
Capture:
    let cuda_ops: &CudaOps = ops.downcast();   // engine knows it's on CUDA
    let graph_bufs = cuda_ops.allocate_graph_buffers(max_batch, max_blocks);
    cuda_ops.precompute_paged_plan_graphed(.., &graph_bufs);    // outside capture
    stream.begin_capture();
    model.forward(..);                                           // captured
    stream.end_capture() → graph;

Replay:
    update_input_buffers(..);
    cuda_ops.precompute_paged_plan_graphed(.., &graph_bufs);    // update metadata
    graph.launch();
```

**Why not in OpsSession:** `GraphMetaBuffers` is meaningless on Metal/Vulkan/TPU/CPU.
Putting it in the shared trait forces every device to handle CUDA-specific types.
Keeping it as a `CudaOps` method means the engine's CUDA graph runner is the only code
that knows about it — and that runner already knows it's on CUDA.

**HIP graphs:** Same pattern. `RocmOps` has equivalent `precompute_paged_plan_graphed` method.
HIP 6.1+ supports graph capture/replay with the same API shape.

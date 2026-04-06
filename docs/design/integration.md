# External Integration

Prelude operates in two deployment modes:

1. **Standalone** — `prelude-server` binary, self-contained HTTP serving with own scheduler + disaggregation orchestration.
2. **Dynamo backend** — `prelude-dynamo` binary, runs as a worker under NVIDIA Dynamo's multi-node orchestration.

Both modes share the same `prelude-core` library. The difference is who owns the orchestration layer.

## Deployment Modes

```
Standalone mode:                              Dynamo backend mode:

Client (HTTP)                                 Client (HTTP)
    │                                             │
prelude-server (Axum)                         Dynamo Router (KV-aware)
    │                                             │ (TCP/HTTP/NATS)
Engine::new(config)                           prelude-dynamo
    │                                             │
Scheduler + Executor + Model                  Engine (via AsyncEngine trait)
    │ (if disaggregated)                          │
Coordinator + KvTransfer                      Scheduler + Executor + Model
    │                                             │ (if P/D disagg)
Mooncake TransferEngine                       Dynamo NIXL/Mooncake (Dynamo owns transfer)
```

## Dynamo Integration (`prelude-dynamo/`)

NVIDIA Dynamo is an open-source datacenter-scale inference orchestration layer. It handles
multi-node request routing, KV-aware scheduling, P/D disaggregation orchestration,
multi-tier KV cache management (KVBM), and SLA-driven autoscaling.

Current Dynamo backends (vLLM, SGLang, TensorRT-LLM) are all Python. Prelude integrates
as a **native Rust backend** — no Python wrapper, direct `AsyncEngine` trait implementation.

### Interface

Dynamo's Rust runtime defines:

```rust
#[async_trait]
pub trait AsyncEngine<Req, Resp, E>: Send + Sync {
    async fn generate(&self, request: Req) -> Result<Resp, E>;
}
```

`prelude-dynamo` wraps `prelude-core::Engine` to implement this:

```rust
// prelude-dynamo/src/lib.rs

struct PreludeBackend {
    engine: Engine,
}

#[async_trait]
impl AsyncEngine<PreprocessedRequest, ManyOut<LLMEngineOutput>, Error>
    for PreludeBackend
{
    async fn generate(&self, request: PreprocessedRequest)
        -> Result<ManyOut<LLMEngineOutput>, Error>
    {
        // Map Dynamo's PreprocessedRequest → Prelude's internal request format
        // Stream tokens back via ManyOut
    }
}
```

### Worker Lifecycle

```rust
// prelude-dynamo/src/main.rs

async fn app(runtime: Runtime) -> anyhow::Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime).await?;
    let ns = distributed.namespace("prelude")?;
    let component = ns.component("backend")?;
    let endpoint = component.endpoint("generate");

    let engine = Engine::new(&config);
    let backend = PreludeBackend { engine };

    endpoint
        .endpoint_builder()
        .handler(Ingress::for_engine(Arc::new(backend))?)
        .start()
        .await?;

    Ok(())
}
```

The binary is self-registering: it reads Dynamo env vars (`DYN_DISCOVERY_BACKEND`,
`DYN_REQUEST_PLANE`, etc.), connects to service discovery (etcd), and publishes
its endpoint. The Dynamo router discovers and routes requests to it.

### KV Cache Exposure for Disaggregation

When running under Dynamo with P/D disaggregation, **Dynamo owns the KV transfer**.
Prelude does NOT initialize its own Mooncake/NIXL instance. Instead, it exposes
KV cache memory addresses for Dynamo's transfer layer:

```rust
// prelude-core exposes:
impl BlockAllocator {
    /// Returns GPU memory addresses for the given block IDs.
    /// Used by external orchestrators (Dynamo) to register with their transfer layer.
    pub fn get_block_memory_info(&self, block_ids: &[BlockId])
        -> Vec<BlockMemoryInfo>
    {
        // addr, length, device_id for each block
    }
}
```

Dynamo registers these addresses with NIXL/Mooncake and initiates GPU-to-GPU transfers.
Prelude's scheduler receives transferred blocks via `BlockAllocator::import_blocks()`
as usual — the source of the blocks (local Mooncake vs Dynamo NIXL) is transparent.

### KV Event Publishing

For Dynamo's KV-aware router, Prelude publishes cache events:

```rust
// When blocks are cached (prefix hit possible):
kv_publisher.publish_stored_event(block_hash, prefix_hash, block_size);

// When blocks are evicted:
kv_publisher.publish_removed_event(block_hash);
```

This feeds Dynamo's KVIndexer, which builds a global prefix tree across all workers
for cache-aware routing decisions.

### What changes in prelude-core

Nothing. The `Engine` and `InferenceEngine` trait are already usable as a library.
`prelude-dynamo` wraps them externally:

| Component | Standalone | Dynamo backend |
|-----------|-----------|----------------|
| Scheduler (ArScheduler) | Unchanged | Unchanged |
| Model code | Unchanged | Unchanged |
| Ops layer | Unchanged | Unchanged |
| BlockAllocator | Unchanged | + `get_block_memory_info()` (read-only query) |
| KvTransfer | Prelude owns (Mooncake) | Dynamo owns (NIXL/Mooncake) |
| Coordinator | Prelude's `pd/coordinator.rs` | Dynamo's Router |
| HTTP API | prelude-server (Axum) | Dynamo's Frontend |

## Mooncake Transport (`prelude-mooncake/`)

[Mooncake](https://github.com/kvcache-ai/Mooncake) is a KVCache-centric disaggregated
architecture for LLM serving (FAST 2025 Best Paper). Its Transfer Engine provides
high-performance data transfer across multiple transports with topology-aware path selection.

Prelude uses Mooncake as the transport backend for standalone disaggregated serving.

### Why Mooncake

1. **Multi-transport**: RDMA, NVLink, TCP, NVMe-of, EFA, HIP, CXL — auto-selects best path.
2. **Topology-aware**: NIC selection based on NUMA affinity and PCIe proximity.
3. **Multi-NIC aggregation**: Up to 190 GB/s on 8x400 Gbps RoCE.
4. **Fault handling**: Automatic NIC failure detection and alternative path resubmission.
5. **Rust API available**: FFI bindings to Transfer Engine exist upstream.
6. **Production-proven**: Powers Kimi's serving infrastructure.

### Integration

`prelude-mooncake` implements Prelude's `KvTransfer` trait using Mooncake's Transfer Engine:

```rust
// prelude-mooncake/src/lib.rs

struct MooncakeTransfer {
    engine: mooncake::TransferEngine,
}

impl KvTransfer for MooncakeTransfer {
    fn send(&self, req: KvTransferRequest) -> Result<()> {
        // Register KV cache blocks with Mooncake
        // Submit batch transfer to target worker's segment
    }

    fn receive(&self) -> Result<KvTransferResult> {
        // Poll for incoming transfers
        // Import received blocks into local BlockAllocator
    }
}
```

### Transport Selection

Mooncake auto-detects and selects the optimal transport:

| Transport | When | Latency | Bandwidth |
|-----------|------|---------|-----------|
| NVLink P2P | Same node, NVLink connected | ~us | 900 GB/s |
| GPU RDMA (GDR) | Same cluster, RDMA fabric | ~100us | 87-190 GB/s |
| TCP relay | Cross-node, no RDMA | ~ms | Network-limited |

The scheduler calls `transfer.send()` / `transfer.receive()` without knowing
which transport is used. Mooncake handles path selection internally.

### Dependencies

Mooncake's dependencies (ibverbs, numa, etcd client) are standard multi-node
infrastructure — any cluster running disaggregated inference already has these.
The crate is feature-gated; single-machine deployments don't pull in any of this.

### Mooncake Store

Mooncake also provides a distributed KV cache storage layer (Mooncake Store)
with put/get semantics, zero-copy transfers, and multi-replica support.
Upstream Rust bindings for Store are in progress
([kvcache-ai/Mooncake#1809](https://github.com/kvcache-ai/Mooncake/issues/1809)).
Once available, `prelude-mooncake` can integrate Store for richer KV cache management
(e.g., multi-tier caching, cross-worker cache sharing).

## RL Training Integration

Prelude serves as the rollout (inference) engine for RL post-training frameworks.
The critical requirement: **logprob consistency between training and inference forward passes.**

### The Problem

In on-policy RL (PPO, GRPO), the training loop is:
1. Inference engine: `policy.forward(prompt)` → tokens + `logprob_rollout`
2. Training engine: `policy.forward(prompt + tokens)` → `logprob_train`
3. `loss = f(logprob_train, logprob_rollout, reward)`

If `logprob_train ≠ logprob_rollout` for the same input, the policy gradient is biased.
Sources of mismatch: different kernel implementations, different reduction orders,
different dtype cast timing, different fusion strategies.

### GPU: Batch Invariance + Algorithmic Correction

On GPU (CUDA/ROCm), bit-wise consistency between training and inference is impractical.
Different frameworks (Megatron, FSDP, PyTorch) use different kernel implementations
(cuBLAS, Triton, custom CUDA), and floating-point non-associativity means different
reduction orders produce different results. SGLang spent over a year on per-op alignment
and still has open issues. Slime removed their true-on-policy mode due to maintenance cost.

Prelude's GPU strategy:

1. **Batch invariance** (`--enable-deterministic-inference`): same input produces same output
   regardless of batch composition. Achieved by using fixed-order reduction kernels and
   disabling non-deterministic ops. FusedOps that change accumulation order return `None`,
   falling back to deterministic unfused ops.

2. **FP32 logprob computation**: logits and log_softmax computed in FP32 regardless of
   model dtype, reducing numerical drift.

3. **Algorithmic correction**: training frameworks (Slime, OpenRLHF, etc.) use TIS
   (Truncated Importance Sampling) to correct for the remaining logprob mismatch.
   Prelude provides `train_rollout_logprob_abs_diff` as a monitoring metric.

4. **Weight hot-update API**: `/update_weights_from_tensor`, `/flush_cache`,
   `/pause_generation`, `/continue_generation` endpoints for RL training loop integration.

### TPU: True On-Policy via XLA Determinism

On TPU, true on-policy (bit-wise zero logprob diff) is achievable because:

1. **Hardware determinism**: TPU's MXU (128×128 systolic array) has a fixed dataflow
   with deterministic accumulation order. No atomicAdd races, no dynamic thread scheduling,
   no cuBLAS autotuning. XLA compiles the entire computation graph statically.

2. **Single compiler**: both training (JAX) and inference (Prelude TpuOps) generate XLA HLO.
   Same XLA ops in the same order → same compiled TPU instructions → same results.

3. **ICI determinism**: TPU's Inter-Chip Interconnect uses deterministic reduction trees
   for collective operations, unlike GPU NCCL which can vary reduction order.

**Architecture for true on-policy on TPU:**

```
Training (JAX/Flax):
  model.forward(x) → XLA HLO → TPU → logprob_train

Inference (Prelude TpuOps):
  model.forward(x) → XLA HLO → TPU → logprob_rollout

Same XLA ops + same sharding → logprob_train == logprob_rollout (bit-wise)
```

**Requirements:**
- Prelude TpuOps must generate the same XLA ops as the training framework for each layer
  (dot_general for matmul, reduce for softmax, etc.)
- Sharding configuration must match between training and inference (same TP degree)
- Training-side logprob recomputation uses inference-matched parallelism (RLAX approach,
  <10% overhead since rollout generation dominates)

**What Prelude provides:**
- `XLATensorOps`: generates standard XLA ops that match JAX's op semantics
- ComposedOps composes them identically to how JAX models compose ops
- Same sharding annotations → same XLA SPMD compilation → same TPU instructions

**No additional libraries needed** — XLA and the TPU hardware guarantee the rest.
This is a unique advantage of TPU over GPU for RL workloads, and a unique advantage
of Prelude's XLA-based TpuOps over Python inference engines that cannot run on TPU.

### API for RL Frameworks

Both GPU and TPU backends expose the same RL integration API:

```
POST /update_weights_from_tensor   — hot-reload weights from training process
POST /flush_cache                  — clear KV cache after weight update
POST /pause_generation             — pause inference during weight update
POST /continue_generation          — resume inference after weight update
GET  /health_generate              — health check for rollout readiness
```

Logprob output is available via `return_logprob=true` in generation requests.
FP32 logprob computation is enabled via `--enable-fp32-lm-head`.

## No Conflict Between Modes

Mooncake is a transport tool, not state. The rule is simple:

- **Standalone**: Prelude initializes Mooncake TransferEngine, owns all transfers.
- **Dynamo backend**: Prelude does NOT initialize Mooncake. Dynamo owns transfers via NIXL/Mooncake.

Same KV cache memory, same `BlockAllocator`, same `import_blocks()` path.
Only the transfer ownership differs. No resource conflicts (NIC, memory registration, etcd metadata).

```rust
// Engine initialization
match mode {
    DeployMode::Standalone { mooncake_config } => {
        let transfer = MooncakeTransfer::new(mooncake_config)?;
        let coordinator = Coordinator::new(transfer);
        // Prelude manages everything
    }
    DeployMode::DynamoBackend => {
        // No transfer layer initialized
        // Dynamo handles routing + KV transfer
        // Prelude just exposes block memory via get_block_memory_info()
    }
}
```

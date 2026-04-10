## Op Traits

### Attention

```rust
// prelude-core/src/ops/traits/attention.rs

trait AttentionOps: Send + Sync {
    /// Varlen attention over contiguous Q, K, V.
    ///
    /// Covers: LLM prefill (causal), diffusion self-attention (bidirectional),
    /// cross-attention (Q and K/V from different sources with different cu_seqlens),
    /// sliding window, softcap.
    fn varlen_attention(
        &self,
        q: &Tensor, k: &Tensor, v: &Tensor,
        params: &VarlenParams,
    ) -> Result<Tensor>;

    /// Paged attention: Q attends to K/V in block cache.
    ///
    /// Covers: LLM decode (Q=1), chunked prefill (mixed Q lengths),
    /// LLM prefill with prefix cache reuse (Q>1, K/V already in cache).
    fn paged_attention(
        &self,
        q: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        params: &PagedParams,
    ) -> Result<Tensor>;
}

/// Scheduling metadata uses `&Tensor`, not `&[u32]`.
///
/// Rationale: cu_seqlens and block_tables may live on any device. Using Tensor
/// lets the backend read them wherever they are. Our own Tensor is a zero-cost
/// wrapper over host slices for CPU data, so there's no overhead vs `&[u32]`
/// for the CPU case.
struct VarlenParams<'a> {
    pub cu_seqlens_q: &'a Tensor,  // [batch+1], cumulative sequence offsets
    pub cu_seqlens_k: &'a Tensor,  // [batch+1], may differ from cu_seqlens_q (cross-attention)
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,     // Gemma2/3 logit capping
}

struct PagedParams<'a> {
    pub block_tables: &'a Tensor,  // [batch * max_blocks_per_seq], flattened block indices
    pub cu_seqlens_q: &'a Tensor,  // [batch+1]
    pub cu_seqlens_k: &'a Tensor,  // [batch+1]
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub scale: f32,
    pub mask: MaskType,
    pub softcap: Option<f32>,     // Gemma2/3 logit capping (same as VarlenParams)
}

enum MaskType {
    Causal,
    Bidirectional,
    SlidingWindow { left: usize, right: usize },
    /// Custom attention mask tensor. Used for speculative decoding tree attention:
    /// each token attends to its ancestors in the draft tree, not a simple causal pattern.
    /// Mask shape: [max_seqlen_q, max_seqlen_k], values are 0.0 (attend) or -inf (mask).
    /// Passed to attention kernel as additive bias on logits (before softmax).
    Custom(Tensor),
}
```

**Design decisions:**

- **Two methods, not one.** Contiguous and paged have different tensor layouts (`cu_seqlens_k` vs `block_tables + seqused_k`). A single method with `Option<PagedKvRef>` conflates them and makes both signatures worse.

- **Cross-attention is varlen_attention.** Q from decoder, K/V from encoder — just different `cu_seqlens_q` and `cu_seqlens_k`. No special method needed.

- **head_dim asymmetry (MLA) is derived from tensor shapes.** `VarlenParams` does not carry head_dim. The implementation inspects Q shape `[_, _, head_dim_q]` and K shape `[_, _, head_dim_k]` to select the correct kernel.

- **Decode is paged_attention with max_seqlen_q=1.** The implementation dispatches to a decode-specialized kernel (FlashInfer decode) or a varlen kernel that handles Q=1 (FlashInfer FA3 prefill). Model code does not distinguish prefill vs decode — it's always `paged_attention()`.

- **Chunked prefill (mixed Q lengths in one batch) is paged_attention with max_seqlen_q>1.** The varlen kernel handles mixed Q lengths via `cu_seqlens_q`. No special API.

### KV Cache

```rust
// prelude-core/src/ops/traits/kv_cache.rs

trait KvCacheOps: Send + Sync {
    /// Query per-head cache slot layout for KV cache allocation.
    ///
    /// The engine calls this once at model load time to determine how much memory
    /// each KV cache slot needs. Standard bf16 cache returns head_dim elements at
    /// the model dtype. KV cache quantization (e.g., TurboQuant, KV-FP8) returns
    /// a different slot size and/or dtype to reflect the compressed representation.
    ///
    /// Default: uncompressed — head_dim elements at the given dtype.
    fn cache_slot_spec(&self, head_dim: usize, dtype: DType) -> CacheSlotSpec {
        CacheSlotSpec { slot_size: head_dim, dtype }
    }

    /// Write K/V to paged cache at given slot positions.
    ///
    /// Separate from attention because models control timing.
    /// Example: Qwen3 runs fused_knorm_rope_cache_write() before attention.
    ///
    /// When KV cache quantization is enabled, this method handles encoding
    /// (quantize + pack) internally — callers always pass bf16/fp16 K/V tensors.
    fn reshape_and_cache(
        &self,
        key: &Tensor, value: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor,   // scheduling metadata, same rationale as VarlenParams
    ) -> Result<()>;
}

/// Cache slot layout descriptor. Returned by `cache_slot_spec`.
struct CacheSlotSpec {
    pub slot_size: usize,  // number of elements (or bytes for packed formats) per head per token
    pub dtype: DType,      // element type: bf16, fp8, u8 (for bit-packed quantization)
}
```

Separate from `AttentionOps` because:
1. Not all models use KV cache (diffusion doesn't).
2. Models must control when cache writes happen relative to other fusions.
3. Some devices may not support paged KV at all (Vulkan, early TPU).

**Design decisions:**

- **`cache_slot_spec` enables KV cache quantization without model changes.** The engine queries slot layout at load time. Standard cache returns `(head_dim, bf16)`. Quantized cache (TurboQuant, KV-FP8) returns a different size/dtype. The engine allocates accordingly. Model code never knows — it passes bf16 K/V to `reshape_and_cache`, and the device impl encodes internally.

- **Encode/decode is device-internal.** `reshape_and_cache` handles encoding (quantize + pack). `paged_attention` handles decoding (unpack + dequant) before running the attention kernel. Because `CudaOps` implements both `KvCacheOps` and `AttentionOps`, encode/decode coordination is internal state — no new trait methods, no model changes.

### GEMM

```rust
// prelude-core/src/ops/traits/gemm.rs

trait GemmOps: Send + Sync {
    /// Matrix multiply. Dispatch: DeepGEMM > CUTLASS > CK > XLA > CPU BLAS.
    ///
    /// Dtype-aware: if inputs are FP8/INT8, the implementation routes to
    /// quantized GEMM kernels (DeepGEMM FP8, CUTLASS INT8, CK FP8, etc.).
    /// Scale factors are tensor metadata, not separate parameters.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Quantized matmul with explicit per-tensor/per-channel scaling.
    ///
    /// Covers FP8 (DeepGEMM/CK), W4A16 (AWQ), W4A4 (Nunchaku/GPTQ) variants.
    /// For weight-only quantization: a is activations (BF16/FP16), b is quantized weights.
    /// scale_a: per-token or per-tensor activation scale (None for weight-only quant).
    /// scale_b: per-channel or per-group weight scale.
    fn quantized_matmul(
        &self,
        a: &Tensor, b: &Tensor,
        scale_a: Option<&Tensor>,
        scale_b: Option<&Tensor>,
        quant: QuantScheme,
    ) -> Result<Tensor>;

    /// Grouped GEMM for MoE: per-expert weights applied to routed tokens.
    fn grouped_gemm(
        &self,
        input: &Tensor,
        weights: &Tensor,                // [num_experts, N, K]
        sorted_token_ids: &Tensor,
        sorted_expert_ids: &Tensor,
        num_tokens_per_expert: &Tensor,
    ) -> Result<Tensor>;
}

/// Quantization scheme for quantized_matmul dispatch.
enum QuantScheme {
    /// FP8 E4M3 with per-tensor/per-token scaling (DeepGEMM, CUTLASS, CK).
    Fp8E4m3,
    /// Weight-only 4-bit with per-group scaling (AWQ, GPTQ).
    W4A16 { group_size: usize },
    /// 4-bit weights + 4-bit activations (Nunchaku SVD-Q).
    W4A4 { group_size: usize },
    /// INT8 symmetric quantization (SmoothQuant).
    Int8,
}
```

### Normalization

```rust
// prelude-core/src/ops/traits/norm.rs

trait NormOps: Send + Sync {
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
    fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor>;
    fn group_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
                  num_groups: usize, eps: f32) -> Result<Tensor>;
}
```

- LLM uses `rms_norm`.
- Diffusion uses `group_norm`.
- TTS/vision uses `layer_norm`.
- All three are needed for multi-model support.

### Activation

```rust
// prelude-core/src/ops/traits/activation.rs

trait ActivationOps: Send + Sync {
    fn silu(&self, x: &Tensor) -> Result<Tensor>;
    fn gelu(&self, x: &Tensor) -> Result<Tensor>;
    fn gelu_approximate(&self, x: &Tensor) -> Result<Tensor>; // tanh approx, used by Flux/DiT
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor>;
    fn sigmoid(&self, x: &Tensor) -> Result<Tensor>;           // default: 1/(1+exp(-x))
    fn log_softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor>; // default: x - log(sum(exp(x)))
}
```

### Convolution

```rust
// prelude-core/src/ops/traits/conv.rs

trait ConvOps: Send + Sync {
    fn conv1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
              stride: usize, padding: usize) -> Result<Tensor>;
    fn conv_transpose1d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
                        stride: usize, padding: usize, output_padding: usize) -> Result<Tensor>;
    fn conv2d(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
              stride: [usize; 2], padding: [usize; 2]) -> Result<Tensor>;
}
```

- Diffusion: conv2d (UNet, DiT).
- TTS: conv1d + conv_transpose1d (vocoder upsampling, encoder).
- LLM: conv1d (DeltaNet/Mamba causal conv).

### Communication (Distributed)

```rust
// prelude-core/src/ops/traits/comm.rs

trait CommOps: Send + Sync {
    /// Number of ranks in the communication group.
    fn world_size(&self) -> usize;

    /// This rank's index in the communication group.
    fn rank(&self) -> usize;

    /// Sum-reduce tensor across all ranks in the tensor-parallel group.
    fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor>;

    /// Concatenate tensor shards from all ranks along `dim`.
    fn all_gather(&self, x: &Tensor, dim: usize) -> Result<Tensor>;

    /// Reduce-scatter: reduce across ranks, then scatter result shards.
    fn reduce_scatter(&self, x: &Tensor, dim: usize) -> Result<Tensor>;

    /// All-to-all: each rank sends/receives different data to/from every other rank.
    /// Used for Ulysses sequence parallelism and MoE expert routing.
    fn all_to_all(&self, x: &Tensor, input_splits: &[usize], output_splits: &[usize]) -> Result<Tensor>;

    /// Point-to-point send to a specific remote rank/group.
    /// Used for attention-FFN disaggregation: attention side sends hidden states to FFN workers.
    /// Transport is device-internal (NVLink P2P, NCCL send, RDMA, StepMesh, etc.).
    fn send(&self, x: &Tensor, dst: RemoteTarget) -> Result<()> {
        bail!("point-to-point send not supported on this device")
    }

    /// Point-to-point receive from a specific remote rank/group.
    fn recv(&self, src: RemoteTarget) -> Result<Tensor> {
        bail!("point-to-point recv not supported on this device")
    }

    /// Fused MoE dispatch: quantize to FP8 + GPU-initiated send to expert owners.
    /// Uses UCCL-EP when available. None = not supported, fallback to all_to_all.
    /// Same Option pattern as FusedOps — devices override what they support.
    fn ep_dispatch_fused(
        &self,
        x: &Tensor,
        topk_ids: &Tensor,
        num_experts: usize,
        use_fp8: bool,
    ) -> Option<Result<(Tensor, Tensor)>> { None }  // (recv_tokens, scales)

    /// Fused MoE combine: receive + weighted accumulate expert outputs.
    fn ep_combine_fused(
        &self,
        x: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
    ) -> Option<Result<Tensor>> { None }
}
```

**Device implementations:**
- **CUDA**: NCCL (dlopen'd at runtime), custom all-reduce for single-node TP (P2P), UCCL-EP for fused MoE dispatch. Symmetric memory on H100+.
- **ROCm**: RCCL (dlopen'd at runtime), custom all-reduce on MI300 (QuickAllReduce), UCCL-EP for fused MoE dispatch.
- **TPU**: XLA collective ops (compiled into HLO, hardware-optimized).
- **Single-device** (Metal, Vulkan, CPU): identity passthrough (TP=1, no communication needed).

**How TP uses CommOps:**

```rust
// prelude-core/src/models/commons/linear.rs — inside Linear::forward, step 3 (TP)

match self.tp {
    TpMode::Row => ops.all_reduce_sum(&out),           // row parallel: reduce
    TpMode::Column { gather_output: true } =>
        ops.all_gather(&out, /*dim=*/-1),              // col parallel: gather
    _ => Ok(out),                                            // col { gather: false } or None
}
```

**Key insight: attention ops don't know about TP.** The QKV projection uses `Linear { tp: Column { gather_output: false } }` — each rank gets `num_heads / TP` heads. The attention kernel receives already-sharded Q/K/V and computes locally — no all-reduce inside attention. All-reduce happens in the output projection `Linear { tp: Row }`.

### Fusion

Fused kernels are **separate ops that return `Option`**. `None` = not supported on this device, model falls back to separate ops. This is not a hint system — the model explicitly checks the return value.

```rust
// prelude-core/src/ops/traits/fused.rs

/// All methods have default `{ None }` — devices only override what they support.
/// Adding a new fusion method requires NO changes to existing device implementations.
trait FusedOps: Send + Sync {
    /// Fused residual add + RMSNorm: computes (x + h) and rms_norm(x + h) in one kernel.
    fn fused_add_rmsnorm(
        &self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> { None }

    /// Fused SiLU(gate) * up.
    fn fused_silu_mul(
        &self, gate: &Tensor, up: &Tensor,
    ) -> Option<Result<Tensor>> { None }

    /// Fused QK-norm + RoPE: normalize Q and K, apply rotary embedding.
    fn fused_qknorm_rope(
        &self,
        q: &Tensor, k: &Tensor,
        q_weight: &Tensor, k_weight: &Tensor,
        cos: &Tensor, sin: &Tensor,
        position_ids: &Tensor, eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> { None }

    /// Fused K-norm + RoPE + KV cache write.
    fn fused_knorm_rope_cache_write(
        &self,
        k: &Tensor, v: &Tensor,
        k_weight: &Tensor,
        cos: &Tensor, sin: &Tensor,
        position_ids: &Tensor,
        key_cache: &Tensor, value_cache: &Tensor,
        slot_mapping: &Tensor, eps: f32,   // scheduling metadata, same rationale as VarlenParams
    ) -> Option<Result<()>> { None }

    /// Fused Adaptive Layer Norm (AdaLN-Zero).
    ///
    /// The signature normalization op for diffusion transformers (DiT, Flux, HunyuanVideo).
    /// Computes: normed = layer_norm(x) * (1 + scale) + shift, gated = normed * gate.
    /// (scale, shift, gate) are derived from timestep embedding by the model.
    ///
    /// Without fusion: 4 memory-bound element-wise ops. With fusion: single kernel pass.
    fn fused_adaln_zero(
        &self,
        x: &Tensor,
        weight: &Tensor, bias: Option<&Tensor>,
        scale: &Tensor, shift: &Tensor, gate: &Tensor,
        eps: f32,
    ) -> Option<Result<(Tensor, Tensor)>> { None }

    /// Fused scale + shift (continuous AdaLN variant).
    ///
    /// Simpler form: layer_norm(x) * (1 + scale) + shift, no gate.
    /// Used in Flux final norm, Sana, etc.
    fn fused_scale_shift(
        &self,
        x: &Tensor,
        weight: &Tensor, bias: Option<&Tensor>,
        scale: &Tensor, shift: &Tensor,
        eps: f32,
    ) -> Option<Result<Tensor>> { None }

    /// Fused multi-LoRA matmul: y = base_weight @ x + scale * (lora_b @ lora_a @ x)
    ///
    /// Each token in the batch can use a different LoRA adapter (for multi-tenant serving).
    /// adapter_indices: [batch] mapping each token to its adapter (-1 = no LoRA).
    /// lora_a: [num_adapters, rank, in_features], lora_b: [num_adapters, out_features, rank].
    ///
    /// Without fusion: split batch by adapter, N separate matmuls, merge. O(N) kernel launches.
    /// With fusion (BGMV/Punica): single kernel handles all adapters. O(1) launch.
    fn fused_lora_matmul(
        &self,
        x: &Tensor,
        base_weight: &Tensor,
        lora_a: &Tensor,
        lora_b: &Tensor,
        adapter_indices: &Tensor,
        lora_scale: f32,
    ) -> Option<Result<Tensor>> { None }

    /// Fused element-wise add. Used by Tensor `+` operator overload via thread-local
    /// Ops context. Returns `None` to fall back to the built-in add, `Some` for vectorized kernel.
    fn fused_add(
        &self, a: &Tensor, b: &Tensor,
    ) -> Option<Result<Tensor>> { None }

    /// Fused MoE routing: topk selection + weight normalization + sorting in one kernel.
    /// Returns (topk_weights, topk_ids, sorted_expert_ids, sorted_token_ids).
    fn fused_moe_routing(
        &self, router_logits: &Tensor, topk: usize,
    ) -> Option<Result<(Tensor, Tensor, Tensor, Tensor)>> { None }
}
```

**Why `Option<Result<T>>`:**

```rust
// prelude-core/src/models/ — model code, fusion boundary is explicit
let (q, k) = match ops.fused_qknorm_rope(&q, &k, ...) {
    Some(result) => result?,            // device supports it
    None => {                           // fallback to separate ops
        let q = ops.rms_norm(&q, &qw, eps)?;
        let k = ops.rms_norm(&k, &kw, eps)?;
        (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?)
    }
};
```

The model is always correct regardless of which device it runs on. CUDA returns `Some` (fused kernel), CPU/Vulkan returns `None` (separate ops).

**Why not put fusion hints in AttentionParams:**

Qwen3 forward does this:
```
step 1: fused_knorm_rope_cache_write(k, v, cache, slots)   // writes to cache
step 2: paged_attention(q, cache)                            // reads from cache
```

If cache write were a "hint" inside `paged_attention`, step 1 couldn't exist as a separate fused kernel. The model must control the boundary between cache write and attention.

**Extensibility:** New fusions are new methods on `FusedOps`. Devices that don't support them return `None`. No combinatorial explosion — each fusion is independent, not a combination of flags.

### TensorOps Dual Backend (Temporary)

Basic tensor operations (matmul, element-wise, cast, reduce) are handled by candle-core
natively. The Ops trait only covers fused/inference-specific ops. There is no separate
`TensorOps` trait — candle's `Tensor` type provides all basic operations directly.

GEMM dispatch is pluggable: CUDA registers CUTLASS/DeepGEMM via
`candle_core::cuda_backend::gemm_dispatch::register_gemm_dispatch()` at startup.
All `Tensor::matmul()` calls on CUDA route through this registered dispatch.

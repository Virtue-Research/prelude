## OpsBundle

Models receive a single struct with a **flat API** — no nesting:

```rust
// prelude-core/src/ops/traits/bundle.rs

struct OpsBundle {
    // Internal trait objects (private — models use flat methods below)
    tensor: Arc<dyn TensorOps>,
    attn: Arc<dyn AttentionOps>,
    kv_cache: Arc<dyn KvCacheOps>,
    gemm: Arc<dyn GemmOps>,
    norm: Arc<dyn NormOps>,
    act: Arc<dyn ActivationOps>,
    conv: Arc<dyn ConvOps>,
    comm: Arc<dyn CommOps>,
    fused: Arc<dyn FusedOps>,
    session: Arc<dyn OpsSession>,
}

// Flat API — models call these directly:
impl OpsBundle {
    // Primitives
    pub fn exp(&self, x: &Tensor) -> Result<Tensor> { ... }
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { ... }

    // Composed (norm, activation, attention)
    pub fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> { ... }
    pub fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> { ... }
    pub fn varlen_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, params: &VarlenParams) -> Result<Tensor> { ... }

    // Fused (try device kernel → auto-fallback to composed)
    pub fn fused_add_rmsnorm(&self, residual: &Tensor, x: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> { ... }
    pub fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> { ... }
    pub fn qknorm_rope_and_cache(&self, q: &Tensor, k: &Tensor, v: &Tensor, ...) -> Result<(Tensor, Tensor)> { ... }

    // Construction (for device crates)
    pub fn from_all<T: AllTraits>(all: Arc<T>) -> Self { ... }
    pub fn with_attn(self, v: Arc<dyn AttentionOps>) -> Self { ... }
    pub fn with_session(self, v: Arc<dyn OpsSession>) -> Self { ... }
}
```

**Design principles:**

- **Flat API**: `ops.exp()`, `ops.rms_norm()`, `ops.varlen_attention()` — no `ops.attn.xxx()` nesting.
- **Fused auto-fallback**: `ops.fused_add_rmsnorm()` tries device kernel → fallback to `add + rms_norm`. Models never see `Option`.
- **`TensorOps` base() delegation**: device backends implement `fn base() -> &dyn TensorOps`, override only hot-path methods. Rest auto-delegates.
- **`Linear` is a parameter carrier**: holds weights + LoRA state, passes them to `ops.xxx()`. All fused/fallback/device decisions live in OpsBundle.

**Model code example (Qwen3 attention):**

```rust
let (q, k, v) = self.fused_qkv_projection(x, ops)?;
let (q, k) = ops.qknorm_rope_and_cache(&q, &k, &v, ..., paged_kv)?;

if let Some(kv) = paged_kv {
    ops.paged_attention(&q, kv.key_cache, kv.value_cache, &params)?
} else {
    ops.varlen_attention(&q, &k, &v, &params)?
}
```

Zero fusion branching. Zero device branching. OpsBundle handles everything.

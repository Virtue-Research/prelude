## Non-Softmax Token Mixers

DeltaNet, Mamba, RWKV, RetNet use recurrent state, not KV cache. Their forward signature is fundamentally different from softmax attention:

```
Softmax attention: (Q, K, V, mask) → O
DeltaNet:          (x, conv_state, recurrent_state) → (o, conv_state', recurrent_state')
Mamba:             (x, conv_state, ssm_state) → (o, conv_state', ssm_state')
```

These do NOT go through `AttentionOps`. Models own their token mixer implementations. The `TransformerBlock` handles this via closure injection:

```rust
// prelude-core/src/models/commons/ — TransformerBlock is agnostic to token mixer type
fn forward<A, M>(&self, x: &Tensor, attn_fn: A, mlp_fn: M) -> Result<Tensor>
where A: FnOnce(&Tensor) -> Result<Tensor>
{
    let h = ops.rms_norm(x, &self.ln1_weight, eps)?;
    let h = attn_fn(&h)?;  // softmax attention OR DeltaNet OR Mamba
    // ...
}
```

The model decides per-layer:
```rust
// prelude-core/src/models/qwen35.rs
match self.layer_type(i) {
    Softmax  => block.forward(x, |h| ops.varlen_attention(h, ...)),
    DeltaNet => block.forward(x, |h| self.deltanet[i].forward(h, ...)),
}
```

If DeltaNet needs multi-device support in the future, it gets its own trait (`LinearAttentionOps` or `RecurrentOps`). It does not share a trait with softmax attention.

## Hybrid Cache Management

Models mixing softmax attention + recurrent layers (Qwen3.5 DeltaNet, Jamba Mamba+Attention)
need different cache types per layer. Softmax layers use paged KV cache. Recurrent layers
use conv_state + ssm_state (fixed size per request, no paging).

### CacheSpec Per Layer

```rust
// prelude-core/src/ops/traits/kv_cache.rs

/// Declares what cache a layer needs. Models return one spec per layer.
/// The engine groups layers by spec and allocates accordingly.
enum LayerCacheSpec {
    /// Standard KV cache (softmax attention). Paged.
    Attention {
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
    },
    /// Recurrent state (Mamba, DeltaNet, RWKV). Fixed size per request.
    Recurrent {
        state_shapes: Vec<Vec<usize>>,  // e.g., [(d_conv, d_inner), (d_state, d_inner)]
        state_dtypes: Vec<DType>,
    },
    /// No cache (diffusion, embedding, encoder).
    None,
}

/// Model declares cache requirements per layer at load time.
trait Model {
    fn cache_specs(&self) -> Vec<LayerCacheSpec>;
    fn forward(&mut self, x: &Tensor, ctx: &BatchState, ops: &OpsBundle, cache: &Cache) -> Result<Tensor>;
}
```

### Engine Cache Allocation

```rust
// prelude-core/src/engine/executor.rs

/// Group layers by cache spec, allocate one pool per group.
fn allocate_cache(model: &dyn Model, ops: &OpsBundle, config: &CacheConfig) -> Cache {
    let specs = model.cache_specs();
    
    // Group: layers 0,2,4 = Attention, layers 1,3,5 = Recurrent
    let groups = group_by_spec(&specs);
    
    // Attention group: paged blocks via BlockAllocator
    // Recurrent group: fixed-size state buffer per request
    
    Cache { groups }
}
```

**Design choice (learn from vLLM, simplify):**
- vLLM: `KVCacheSpec` hierarchy (FullAttentionSpec, SlidingWindowSpec, MambaSpec) +
  `HybridKVCacheCoordinator` with fixed-point algorithm for multi-group cache hits +
  automatic page-size unification. Sophisticated but complex (admits "support for >2 types
  not implemented").
- SGLang: separate pool classes (`HybridLinearKVPool` + `MambaPool`), no unified coordinator.
  Simple but doesn't optimize cache layout across types.
- **We use**: `LayerCacheSpec` enum (3 variants, not a class hierarchy). Engine groups layers
  and allocates per group. Attention groups use `BlockAllocator` + `PrefixCache`.
  Recurrent groups use a simple fixed-size buffer (no paging needed — state size is constant).
  No unified page-size reconciliation (different groups can have different allocation strategies).
  Simpler than vLLM, more unified than SGLang.

### Prefix Cache for Hybrid Models

For hybrid models, prefix cache matching must consider ALL layer groups.
A prefix cache hit is valid only if ALL groups have the cached blocks:

```
Layer 0 (attention): prefix cache hit at token 512 ✓
Layer 1 (Mamba):     state available at token 512 ✓  → overall hit at 512
Layer 2 (attention): prefix cache hit at token 512 ✓
Layer 3 (Mamba):     state available at token 256 ✗  → overall hit at 256 (minimum)
```

The minimum across all groups determines the effective cache hit length.
This is the same fixed-point algorithm vLLM uses, but we only need it for
hybrid models — pure-attention models skip this entirely.

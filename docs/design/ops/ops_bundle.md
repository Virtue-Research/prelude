## Ops Bundle

Models receive a single struct:

```rust
// prelude-core/src/ops/mod.rs

struct Ops {
    pub attn: Arc<dyn AttentionOps>,
    pub kv_cache: Arc<dyn KvCacheOps>,
    pub gemm: Arc<dyn GemmOps>,
    pub norm: Arc<dyn NormOps>,
    pub act: Arc<dyn ActivationOps>,
    pub conv: Arc<dyn ConvOps>,
    pub comm: Arc<dyn CommOps>,
    pub fused: Arc<dyn FusedOps>,
    pub session: Arc<dyn OpsSession>,
}
```

All fields are always present. Devices that don't support an op category (e.g., Vulkan has no `paged_attention`) return errors from those methods. This is simpler than `Option<Arc<dyn ...>>` — the error message is more informative than a missing field.

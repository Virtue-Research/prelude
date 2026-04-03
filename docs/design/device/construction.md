## Construction

Each device crate exports its own `create_ops` function. The top-level binary crate
(composition root) selects which device to use — `prelude-core` never depends on any
device crate, avoiding circular dependencies.

**Each device crate:**

```rust
// prelude-cuda/src/lib.rs

pub fn create_ops(config: &OpsConfig) -> Ops {
    let cuda = Arc::new(CudaOps::new(config));
    Ops {
        attn: cuda.clone(), kv_cache: cuda.clone(), gemm: cuda.clone(),
        norm: cuda.clone(), act: cuda.clone(), conv: cuda.clone(),
        comm: cuda.clone(), fused: cuda.clone(), session: cuda,
    }
}
```

```rust
// prelude-rocm/src/lib.rs

pub fn create_ops(config: &OpsConfig) -> Ops {
    let rocm = Arc::new(RocmOps::new(config));
    Ops {
        attn: rocm.clone(), kv_cache: rocm.clone(), gemm: rocm.clone(),
        norm: rocm.clone(), act: rocm.clone(), conv: rocm.clone(),
        comm: rocm.clone(), fused: rocm.clone(), session: rocm,
    }
}
```

Other device crates (`prelude-metal`, `prelude-vulkan`, `prelude-tpu`, `prelude-cpu`)
follow the same pattern.

**Binary crate (composition root):**

Device features are **additive** — `cargo build --features cuda,rocm` compiles both.
Runtime auto-detection selects the best available backend.

```rust
// prelude-server/src/main.rs — runtime device detection, not compile-time selection

fn main() {
    let config = parse_config();

    let ops = match detect_gpu() {
        #[cfg(feature = "cuda")]
        Gpu::Nvidia => prelude_cuda::create_ops(&config),
        #[cfg(feature = "rocm")]
        Gpu::Amd => prelude_rocm::create_ops(&config),
        #[cfg(feature = "metal")]
        Gpu::Apple => prelude_metal::create_ops(&config),
        #[cfg(feature = "vulkan")]
        Gpu::Vulkan => prelude_vulkan::create_ops(&config),
        _ => prelude_cpu::create_ops(&config),  // fallback
    };

    let grammar = prelude_xgrammar::create_backend();
    let engine = Engine::new(ops, grammar, &config);
    engine.serve();
}
```

**Build targets:**

```bash
# Linux: all GPU backends in one binary
cargo build --features cuda,rocm,vulkan        # → prelude-linux-x86_64

# macOS: Metal only (CUDA/ROCm not available on macOS)
cargo build --features metal                    # → prelude-darwin-aarch64
```

NCCL/RCCL are dlopen'd at runtime (not statically linked), so CUDA + ROCm in the
same binary has no symbol conflicts. Runtime detection picks the right backend.

**Dependency graph (no cycles):**

```
prelude-server (binary)
    ├── prelude-core               (pure Rust leaf — no device dependency)
    ├── plugins/prelude-xgrammar   (impl GrammarBackend)
    ├── prelude-cuda               (feature-gated, additive)
    │       ├── prelude-core
    │       ├── fa4/, flashinfer/, deepgemm/, nccl/, uccl-ep/
    │       └── (each sub-crate compiles from third_party/)
    ├── prelude-rocm               (feature-gated, additive)
    │       ├── prelude-core
    │       └── ck/, aiter/, rccl/, uccl-ep/
    └── prelude-cpu                (always included as fallback)
```

`prelude-core` is a pure leaf — it depends on no device crate and compiles no C++.
`Engine` receives `Ops` + `GrammarBackend` via dependency injection, never
constructing them internally.

All `Ops` fields are always populated. Methods on unsupported ops return errors
(`bail!("paged attention not supported on {device}")`) rather than panicking or
silently degrading. Model code has zero `#[cfg]`.

## Construction

Device crates register their Ops + Executor at startup via explicit `register()` calls.
The core registry uses **priority + probe** for automatic backend selection — no device-specific
types in core, no hardware detection code. Each device crate provides its own `probe()` function
that checks whether its hardware is available, and a `priority` that determines selection order.

**Device crates register at startup:**

```rust
// prelude-cuda/src/lib.rs

pub fn register() {
    prelude_core::ops::register_backend(OpsBackend {
        name: "cuda",
        priority: 100,                         // GPU backends: high priority
        probe: || cuda_device(0).is_ok(),      // hardware detection in device crate
        supports: |d| d.is_cuda(),
        create_ops: cuda_ops,
    });
    prelude_core::engine::executor::register_executor(ExecutorBackend {
        name: "cuda",
        priority: 100,
        probe: || cuda_device(0).is_ok(),
        supports: |d| d.is_cuda(),
        create: |engine| Box::new(CudaExecutor::new(engine)),
    });
}
```

```rust
// prelude-cpu/src/lib.rs

pub fn register() {
    prelude_core::ops::register_backend(OpsBackend {
        name: "cpu",
        priority: 10,                          // lowest — always the fallback
        probe: || true,                        // CPU is always available
        supports: |d| d.is_cpu(),
        create_ops: cpu_ops,
    });
    prelude_core::engine::executor::register_executor(ExecutorBackend {
        name: "cpu",
        priority: 10,
        probe: || true,
        supports: |d| d.is_cpu(),
        create: |engine| Box::new(CpuExecutor::new(engine)),
    });
}
```

Other device crates (`prelude-rocm`, `prelude-metal`, `prelude-vulkan`, `prelude-tpu`) follow the
same pattern. ROCm uses `priority: 100`, Metal uses `priority: 100`, Vulkan uses `priority: 50`
(prefer native backends).

**Core registry (`prelude-core/src/ops/mod.rs`) — zero device knowledge:**

```rust
// prelude-core/src/ops/mod.rs

pub struct OpsBackend {
    pub name: &'static str,
    pub priority: u32,                          // higher = preferred
    pub probe: fn() -> bool,                   // device crate's hardware check
    pub supports: fn(&Device) -> bool,         // device kind matching
    pub create_ops: fn() -> &'static dyn Ops,
}

static OPS_REGISTRY: Mutex<Vec<OpsBackend>> = Mutex::new(Vec::new());
static RESOLVED_CPU: OnceLock<&'static dyn Ops> = OnceLock::new();
static RESOLVED_GPU: OnceLock<&'static dyn Ops> = OnceLock::new();

pub fn register_backend(entry: OpsBackend) {
    OPS_REGISTRY.lock().unwrap().push(entry);
}

/// Select the best available backend for a device.
/// Filters by supports(device), then by probe(), picks highest priority.
/// Falls back to bare_ops() (Device or CubeCL backend) if nothing matches.
pub fn select_ops(device: &Device) -> &'static dyn Ops {
    let lock = if device.is_cuda() { &RESOLVED_GPU } else { &RESOLVED_CPU };
    *lock.get_or_init(|| resolve_for(device))
}
```

**Key: prelude-core has NO device types.** No `DeviceType::Cuda`, no `has_nvidia_gpu()`.
Device crates register their Ops via explicit `register()` calls at startup.
`bare_ops()` provides pure Rust primitives as the lowest-priority fallback.

**Design principles:**
- Unified `Ops` trait: models call `ops.xxx()` for everything. All methods have defaults.
- Device backends override only hot-path methods. Everything else auto-inherits.
- `Linear` is a parameter carrier: holds weights, passes them to `ops.xxx()`.
  All fused/fallback/device decisions live in Ops, never in Linear or model code.
- Fused ops: `ops.fused_add_rmsnorm()` tries device kernel → auto-fallback to composed.

**CUDA + ROCm in one binary:** Both register at priority 100. Runtime: `probe()` checks
for actual hardware. Machine with NVIDIA GPU → CUDA probe returns true. Machine
with AMD GPU → ROCm probe returns true. Neither → falls back to CPU (priority 10, always true).

**Server registers backends explicitly at startup:**

```rust
// prelude-server/src/main.rs

fn main() {
    prelude_cpu::register();
    #[cfg(feature = "cuda")]
    prelude_cuda::register();

    let config = EngineConfig::from_env();
    let engine = Engine::new(&config);  // internally calls select_ops()
    engine.serve();
}
```

**Weight loading with device allocation:**

Weight loader is in `prelude-core/src/engine/weight_loader.rs` but needs to allocate
tensors on the correct device. It receives an allocator callback from the Executor:

```rust
// prelude-core/src/engine/weight_loader.rs

pub fn load_weights(
    path: &Path,
    alloc: &dyn Fn(&[usize], DType) -> Tensor,  // device crate provides this
) -> Result<HashMap<String, Tensor>> {
    let mut weights = HashMap::new();
    for (name, cpu_tensor) in WeightIterator::open(path)? {
        let device_tensor = alloc(cpu_tensor.shape(), cpu_tensor.dtype());
        device_tensor.copy_from(&cpu_tensor);  // H2D transfer
        weights.insert(name, device_tensor);
    }
    Ok(weights)
}
```

The allocator is provided by the Executor (which knows the device). prelude-core never
imports device-specific allocation functions.

**Build targets:**

```bash
# Linux: all GPU backends in one binary
cargo build --features cuda,rocm,vulkan        # → prelude-linux-x86_64

# macOS: Metal only (CUDA/ROCm not available on macOS)
cargo build --features metal                   # → prelude-darwin-aarch64
```

Device features are **additive** — `--features cuda,rocm` compiles both. NCCL/RCCL are
dlopen'd at runtime (not statically linked), so CUDA + ROCm in the same binary has no
symbol conflicts.

**Dependency graph (no cycles):**

```
prelude-server (binary)
    ├── prelude-core               (Ops trait, models, engine — no C++, no device types)
    │       ├── cubecl                 (pure Rust: IR + TensorOps primitives, generic over CubeRuntime)
    │       └── llguidance             (constrained decoding, pure Rust)
    ├── prelude-cuda               (feature-gated, register() at startup)
    │       ├── prelude-core
    │       ├── cubecl (features=["cuda"])  (CubeCL CUDA runtime for TensorOps)
    │       └── fa4/, flashinfer/, deepgemm/, nccl/, uccl-ep/
    ├── prelude-rocm               (feature-gated, register() at startup)
    │       ├── prelude-core
    │       ├── cubecl (features=["hip"])
    │       └── ck/, aiter/, rccl/, uccl-ep/
    └── prelude-tpu                (feature-gated, same pattern — XLATensorOps instead of CubeCL)
            ├── prelude-core
            └── pjrt C API (XLA runtime, dlopen libpjrt_tpu.so)
```

`prelude-core` compiles no C++ and contains no device-specific types. It depends on CubeCL
(pure Rust) for `CubeCLTensorOps<R: Runtime>` — a generic TensorOps implementation that
device crates instantiate with their runtime. `bare_ops()` provides pure Rust primitives
as the lowest-priority fallback. CubeCL's CPU runtime serves as an alternative.

**Why this design:**

- **Zero device knowledge in core**: `prelude-core` has no `DeviceType` enum, no hardware
  detection. Device crates own their probe logic.
- **Additive features**: adding a new device crate = zero changes to server or core.
- **Multi-GPU binary**: CUDA + ROCm both register, runtime probe picks the right one.
- **Uniform override pattern**: all backends implement the Ops trait, override hot-path methods.
  Defaults compose the rest. No exceptions — TPU follows the same pattern (XLA provides TensorOps).
- **Model code has zero `#[cfg]`**: all device branching is in the registry layer.

## Construction

Device crates auto-register their `OpsBundle` + `Executor` at link time using `#[ctor::ctor]`.
The core registry has **zero device-specific types** — no `DeviceType` enum, no hardware
detection code. Each device crate provides its own `probe()` function that checks whether
its hardware is available.

**Device crates register via `ctor`:**

```rust
// prelude-cuda/src/lib.rs — device knowledge stays here

#[ctor::ctor]
fn init() {
    prelude_core::ops::register_backend(RegisteredBackend {
        name: "cuda",
        priority: 100,                     // GPU backends: high priority
        probe: || {                         // hardware detection in device crate
            // dlopen libcuda.so, check cuDeviceGetCount > 0
            cuda_is_available()
        },
        create: || {
            let ops = Arc::new(CudaOps::new());
            let executor = Arc::new(CudaExecutor::new());
            (ops, executor)
        },
    });
}
```

```rust
// prelude-rocm/src/lib.rs

#[ctor::ctor]
fn init() {
    prelude_core::ops::register_backend(RegisteredBackend {
        name: "rocm",
        priority: 100,                     // same priority as CUDA
        probe: || rocm_is_available(),     // dlopen libamdhip64.so, check hipGetDeviceCount
        create: || (Arc::new(RocmOps::new()), Arc::new(RocmExecutor::new())),
    });
}
```

```rust
// prelude-cpu/src/lib.rs

#[ctor::ctor]
fn init() {
    prelude_core::ops::register_backend(RegisteredBackend {
        name: "cpu",
        priority: 0,                       // lowest — always the fallback
        probe: || true,                     // CPU is always available
        create: || (Arc::new(CpuOps::new()), Arc::new(CpuExecutor::new())),
    });
}
```

Other device crates (`prelude-metal`, `prelude-vulkan`, `prelude-tpu`) follow the same
pattern. Metal uses `priority: 100`, Vulkan uses `priority: 50` (prefer native backends).

**Core registry (`prelude-core/src/ops/mod.rs`) — zero device knowledge:**

```rust
// prelude-core/src/ops/mod.rs

pub struct RegisteredBackend {
    pub name: &'static str,
    pub priority: u32,                      // higher = preferred
    pub probe: fn() -> bool,               // device crate's hardware check
    pub create: fn() -> (Arc<dyn OpsBundle>, Arc<dyn Executor>),
}

static BACKENDS: Mutex<Vec<RegisteredBackend>> = Mutex::new(Vec::new());

pub fn register_backend(backend: RegisteredBackend) {
    BACKENDS.lock().unwrap().push(backend);
}

/// Select the best available backend.
/// Tries registered backends in priority order, picks the first whose probe() returns true.
/// Falls back to ComposedOps (CubeCL CPU) if nothing is registered.
pub fn select_ops(device: &Device) -> &'static OpsBundle {
    if device.is_cuda() {
        if let Some(factory) = GPU_OPS_FACTORY.get() { return factory(); }
    }
    if let Some(factory) = CPU_OPS_FACTORY.get() { return factory(); }
    primitives::default_cpu_ops()  // CubeCL CPU runtime + ComposedOps
}
```

**Key: prelude-core has NO device types.** No `DeviceType::Cuda`, no `has_nvidia_gpu()`.
Device crates register their `OpsBundle` via `#[ctor]` at link time.
`ComposedOps` is pure composition logic (calls TensorOps trait methods). It doesn't know whether
the primitives come from CubeCL or XLA. Device crates inject the concrete TensorOps at construction.

**Design principles:**
- `OpsBundle` flat API: models call `ops.xxx()` for everything. No nesting.
- `TensorOps` uses `base()` delegation: device backends override only what they need.
- `Linear` is a parameter carrier: holds weights + LoRA state, passes them to `ops.xxx()`.
  All fused/fallback/device decisions live in OpsBundle, never in Linear or model code.
- Fused ops: `ops.fused_add_rmsnorm()` tries device kernel → auto-fallback to composed.

**CUDA + ROCm in one binary:** Both register at priority 100. Runtime: `probe()` checks
for actual hardware. Machine with NVIDIA GPU → CUDA probe returns true first. Machine
with AMD GPU → ROCm probe returns true. Neither → falls back to CPU (priority 0, always true).

**Engine selects backend internally — server has zero device logic:**

```rust
// prelude-server/src/main.rs — no device imports, no detect_gpu, no create_ops

fn main() {
    let config = EngineConfig::from_env();
    let engine = Engine::new(&config);  // internally calls select_backend()
    engine.serve();
}
```

```rust
// prelude-core/src/engine/mod.rs

impl Engine {
    pub fn new(config: &EngineConfig) -> Self {
        let (ops, executor) = crate::ops::select_backend();
        // ...
    }
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
    ├── prelude-core               (OpsBundle, ComposedOps, CubeCLTensorOps<R>, llguidance — no C++, no device types)
    │       ├── cubecl                 (pure Rust: IR + TensorOps primitives, generic over CubeRuntime)
    │       └── llguidance             (constrained decoding, pure Rust)
    ├── prelude-cuda               (feature-gated, ctor registers at link time)
    │       ├── prelude-core
    │       ├── cubecl (features=["cuda"])  (CubeCL CUDA runtime for TensorOps)
    │       └── fa4/, flashinfer/, deepgemm/, nccl/, uccl-ep/
    ├── prelude-rocm               (feature-gated, ctor registers at link time)
    │       ├── prelude-core
    │       ├── cubecl (features=["hip"])
    │       └── ck/, aiter/, rccl/, uccl-ep/
    └── prelude-tpu                (feature-gated, same override pattern — XLATensorOps instead of CubeCL)
            ├── prelude-core
            └── pjrt C API (XLA runtime, dlopen libpjrt_tpu.so)
```

`prelude-core` compiles no C++ and contains no device-specific types. It depends on CubeCL
(pure Rust) for `CubeCLTensorOps<R: Runtime>` — a generic TensorOps implementation that
device crates instantiate with their runtime. `ComposedOps` composes TensorOps primitives into
higher-level ops (NormOps, ActivationOps, AttentionOps, etc.) — pure logic, no device dependency.
CubeCL's CPU runtime serves as the lowest-priority fallback.

**Why this design:**

- **Zero device knowledge in core**: `prelude-core` has no `DeviceType` enum, no hardware
  detection. Device crates own their probe logic.
- **Additive features**: adding a new device crate = zero changes to server or core.
- **Multi-GPU binary**: CUDA + ROCm both register, runtime probe picks the right one.
- **Uniform override pattern**: all backends provide TensorOps primitives + hot-path overrides.
  ComposedOps composes the rest. No exceptions — TPU follows the same pattern (XLA provides TensorOps).
- **Model code has zero `#[cfg]`**: all device branching is in the registry layer.

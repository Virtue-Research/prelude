## Construction

Device crates auto-register their `Ops` + `Executor` at link time using `#[ctor::ctor]`.
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
    pub create: fn() -> (Arc<dyn Ops>, Arc<dyn Executor>),
}

static BACKENDS: Mutex<Vec<RegisteredBackend>> = Mutex::new(Vec::new());

pub fn register_backend(backend: RegisteredBackend) {
    BACKENDS.lock().unwrap().push(backend);
}

/// Select the best available backend.
/// Tries registered backends in priority order, picks the first whose probe() returns true.
/// Falls back to naive_ops if nothing is registered (always works, never panics).
pub fn select_backend() -> (Arc<dyn Ops>, Arc<dyn Executor>) {
    let mut backends = BACKENDS.lock().unwrap();
    backends.sort_by(|a, b| b.priority.cmp(&a.priority));
    for b in backends.iter() {
        if (b.probe)() {
            return (b.create)();
        }
    }
    // No registered backend available — use built-in naive fallback
    (Arc::new(NaiveOps), Arc::new(NaiveExecutor))
}
```

**Key: prelude-core has NO device types.** No `DeviceType::Cuda`, no `has_nvidia_gpu()`.
The core only knows "backends register with a name, priority, and probe function."
All hardware detection logic lives in the device crate's `probe()` closure.

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
    ├── prelude-core               (pure Rust leaf — no device dependency, no device types)
    ├── plugins/prelude-xgrammar   (impl GrammarBackend)
    ├── prelude-cuda               (feature-gated, ctor registers at link time)
    │       ├── prelude-core
    │       └── fa4/, flashinfer/, deepgemm/, nccl/, uccl-ep/
    ├── prelude-rocm               (feature-gated, ctor registers at link time)
    │       ├── prelude-core
    │       └── ck/, aiter/, rccl/, uccl-ep/
    └── prelude-cpu                (always included, priority 0 fallback)
```

`prelude-core` is a pure leaf — it depends on no device crate, compiles no C++, and
contains no device-specific types. The `RegisteredBackend` struct uses only generic
fields (`name`, `priority`, `probe`, `create`).

**Why this design:**

- **Zero device knowledge in core**: `prelude-core` has no `DeviceType` enum, no hardware
  detection. Device crates own their probe logic.
- **Additive features**: adding a new device crate = zero changes to server or core.
- **Multi-GPU binary**: CUDA + ROCm both register, runtime probe picks the right one.
- **Three-tier fallback**: registered GPU → registered CPU → built-in naive_ops. Always works.
- **Model code has zero `#[cfg]`**: all device branching is in the registry layer.

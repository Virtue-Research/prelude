## Tensor Type

The `Tensor` is the fundamental data structure. It is a **thin handle** — holds a device
pointer, shape, and dtype. No computation methods. All computation goes through `Ops` traits.

```rust
// prelude-core/src/tensor.rs

pub struct Tensor {
    data: TensorData,             // device memory (opaque pointer)
    shape: Vec<usize>,            // e.g., [batch, seq_len, hidden_dim]
    dtype: DType,                 // BF16, FP16, FP32, FP8, U8, U32, I64
    strides: Vec<usize>,          // for non-contiguous views
}

/// Opaque device memory. Tensor doesn't know which device — the Ops impl does.
enum TensorData {
    Owned(DeviceBuffer),          // owns the memory, drops when Tensor drops
    View { base: Arc<Tensor>, offset: usize },  // view into another tensor, no copy
}

/// Device-allocated memory buffer. Created by device crate, opaque to prelude-core.
pub struct DeviceBuffer {
    ptr: *mut u8,                 // device pointer (CUDA/HIP/CPU/Metal address)
    len: usize,                   // bytes
    drop_fn: Option<Box<dyn FnOnce(*mut u8)>>,  // device-specific deallocation
}
```

### What Tensor CAN do (metadata, no compute)

```rust
impl Tensor {
    // ── Shape queries ───────────────────────────────────
    pub fn shape(&self) -> &[usize];
    pub fn dims(&self) -> usize;              // number of dimensions
    pub fn dim(&self, d: usize) -> usize;     // size of dimension d
    pub fn elem_count(&self) -> usize;        // product of all dims
    pub fn dtype(&self) -> DType;
    pub fn is_contiguous(&self) -> bool;

    // ── View operations (no data copy, just new metadata) ──
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor>;
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor>;
    pub fn squeeze(&self, dim: usize) -> Result<Tensor>;
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor>;
    pub fn transpose(&self, d1: usize, d2: usize) -> Result<Tensor>;
    pub fn chunk(&self, n: usize, dim: usize) -> Result<Vec<Tensor>>;

    // ── Raw pointer access (unsafe, for FFI to kernels) ──
    pub fn as_ptr(&self) -> *const u8;
    pub fn as_mut_ptr(&mut self) -> *mut u8;
}
```

### What Tensor CANNOT do (computation → goes through Ops)

```
❌ tensor.matmul(&other)         →  ops.gemm.matmul(&a, &b)
❌ tensor.softmax(dim)           →  ops.act.softmax(&x, dim)
❌ tensor.add(&other)            →  element-wise via Ops or operator overload delegating to Ops
❌ tensor.to_device(cuda)        →  device crate allocates + copies
❌ tensor.to_dtype(bf16)         →  device crate handles cast kernel
❌ Tensor::zeros(shape, device)  →  device crate allocates zeroed memory
```

**Why:** Computation requires knowing the device and dispatching to the right kernel.
If `Tensor` could do `matmul()`, it would need to know about CudaOps/CpuOps — breaking
the "prelude-core is device-agnostic" principle.

### Memory Management

**Allocation:** Device crates create Tensors via their own allocators.

```rust
// prelude-cuda/src/cuda_ops.rs
impl CudaOps {
    pub fn alloc_tensor(&self, shape: &[usize], dtype: DType) -> Tensor {
        let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
        let ptr = cuda_malloc(bytes);  // cudaMalloc
        Tensor::from_device_buffer(DeviceBuffer {
            ptr,
            len: bytes,
            drop_fn: Some(Box::new(|p| cuda_free(p))),  // cudaFree on drop
        }, shape, dtype)
    }
}
```

**Deallocation:** Automatic via `Drop`. `DeviceBuffer::drop_fn` calls the device-specific
free function (cudaFree, hipFree, free, etc.). Tensor doesn't know which — the closure captures it.

**Views:** `narrow()`, `reshape()`, etc. create views that share the base Tensor's memory
via `Arc<Tensor>`. The underlying memory is freed only when all views are dropped.

**Transfer:** Moving data between devices (CPU → GPU, GPU → CPU) is a device crate operation:

```rust
// prelude-cuda/src/cuda_ops.rs
impl CudaOps {
    pub fn to_device(&self, tensor: &Tensor) -> Tensor {
        let gpu_tensor = self.alloc_tensor(tensor.shape(), tensor.dtype());
        cuda_memcpy_h2d(gpu_tensor.as_mut_ptr(), tensor.as_ptr(), tensor.byte_len());
        gpu_tensor
    }
}
```

### Element-wise Operators

For ergonomics, `Tensor` supports `+`, `-`, `*`, `/` via Rust operator overloading.
These delegate to a **thread-local Ops context** set by the engine before each forward pass:

```rust
impl std::ops::Add for &Tensor {
    type Output = Result<Tensor>;
    fn add(self, rhs: &Tensor) -> Result<Tensor> {
        // Delegates to the current Ops context's element-wise add
        with_current_ops(|ops| ops.elementwise_add(self, rhs))
    }
}
```

This is the ONLY place where Tensor indirectly touches Ops — and it's a convenience
wrapper, not a fundamental coupling. Models can also call ops directly.

### DType

```rust
// prelude-core/src/tensor.rs

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,          // packed quantized (TurboQuant, GGUF)
    U32,
    I16,         // some quantization formats
    I32,         // integer indices
    I64,
    BF16,
    F16,
    F32,
    F64,         // high-precision math
    F8E4M3,     // FP8 (SM89+, gfx942+)
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            U8 | F8E4M3 => 1,
            I16 | BF16 | F16 => 2,
            U32 | I32 | F32 => 4,
            F8E4M3 | U8 => 1,
            I64 => 8,
        }
    }
}
```

### BatchState

Per-batch runtime state passed to model forward and `Linear.forward`.
Separate from `Ops` (per-device, static) and model weights (per-model, static).

```rust
// prelude-core/src/tensor.rs

struct BatchState<'a> {
    /// Per-token LoRA adapter index. [batch_size] mapping each token to its adapter.
    /// -1 = no LoRA for this token. None = LoRA not active for this batch.
    pub adapter_ids: Option<&'a Tensor>,
}
```

`Linear.forward` reads `batch_state.adapter_ids` to decide whether to apply LoRA.
If `None`, LoRA step is skipped entirely (zero overhead).

Models that don't use LoRA still receive `BatchState` but never inspect it — they just
forward it to `Linear` and module functions. The engine constructs it before each
forward pass with the current batch's adapter routing.

### Relation to candle-core

The current codebase uses `candle_core::Tensor` which bundles computation + device dispatch
inside the Tensor itself. Migrating to this minimal Tensor is part of the Ops trait refactoring:

1. Replace `candle_core::Tensor` with `prelude_core::Tensor` (thin handle)
2. Move `Storage::Cuda` pattern matches into `CudaOps` trait impls
3. Move `Storage::Cpu` pattern matches into `CpuOps` trait impls
4. Model code changes `tensor.matmul(&b)` → `ops.gemm.matmul(&a, &b)`
5. candle-core becomes unused, remove dependency

This is the same work as implementing the Ops trait system — not additional effort.

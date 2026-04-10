## Tensor Type

`Tensor` is backed by **candle-core** (from our fork `rucnyz/candle`, `prelude` branch).
We re-export candle's types as our tensor API — basic tensor ops (matmul, reshape, cast,
element-wise, etc.) go through candle's zero-overhead dispatch. Custom inference ops
(fused kernels, paged attention) go through the Ops trait.

```rust
// prelude-core/src/tensor/mod.rs — re-exports from candle-core

pub use candle_core::{
    Tensor, DType, Device, Shape, Layout, Error, Result, Module,
    Storage, CpuStorage, D,
};
#[cfg(feature = "cuda")]
pub use candle_core::{CudaDevice, CudaStorage};
```

Candle's `Tensor` internally holds `Arc<RwLock<Storage>>` + `Layout`. Storage is an enum
over `CpuStorage` / `CudaStorage` / etc. Views (reshape, narrow, squeeze) share storage
via Arc — no data copy.

### Basic ops: candle handles natively

```rust
tensor.matmul(&other)?          // → candle dispatch → registered GEMM (CUTLASS/DeepGEMM)
tensor.add(&other)?             // → candle CUDA kernel
tensor.to_dtype(DType::BF16)?   // → candle cast kernel
tensor.contiguous()?            // → candle copy kernel if needed
Tensor::zeros(shape, dtype, &device)?  // → candle allocation
```

GEMM dispatch is pluggable: `candle_core::cuda_backend::gemm_dispatch::register_gemm_dispatch()`
routes all `Tensor::matmul()` on CUDA through our CUTLASS/DeepGEMM implementation.
Registered once at startup when `CudaOps` is first accessed.

### Inference-specific ops: through Ops trait

```rust
ops.rms_norm(&x, &weight, eps)              // fused CUDA kernel
ops.fused_add_rmsnorm(&residual, &x, &w, eps)  // fused add + norm
ops.varlen_attention(q, k, v, &params)      // FA4 / FlashInfer
ops.paged_attention(q, kc, vc, &params)     // paged KV cache attention
ops.fused_silu_mul(&gate, &up)              // fused activation
```

### CUDA storage access pattern (for custom ops)

Fused CUDA ops access tensor storage directly via candle's API — zero-copy:

```rust
let (storage, layout) = tensor.storage_and_layout();  // RwLock read guard
let cuda = match &*storage {
    Storage::Cuda(s) => s,
    _ => bail!("requires CUDA"),
};
let dev = cuda.device().clone();              // candle CudaDevice
let stream = dev.cuda_stream();               // the one CUDA stream
let slice = cuda.as_cuda_slice::<T>()?;       // &CudaSlice<T>, no copy
let ptr = slice.device_ptr(&stream);          // raw CUdeviceptr for FFI
let out = unsafe { dev.alloc::<T>(n) }?;      // allocate output
// ... launch kernel ...
drop(storage);  // release read lock before creating output tensor
let out_storage = CudaStorage::wrap_cuda_slice(out, dev);
Tensor::from_storage(Storage::Cuda(out_storage), shape, BackpropOp::none(), false)
```

One CUDA context, one stream — no registry, no fallback. Everything comes from the
tensor's own `CudaDevice`.

### DType

Re-exported from candle-core: `U8`, `U32`, `I64`, `BF16`, `F16`, `F32`, `F64`.
FP8 types are available via candle's `F8E4M3` when the candle fork supports it.

### BatchState

Per-batch runtime state passed to model forward and `Linear.forward`.
Separate from `OpsBundle` (per-device, static) and model weights (per-model, static).

```rust
struct BatchState<'a> {
    pub adapter_ids: Option<&'a Tensor>,  // per-token LoRA adapter index
}
```

### Why candle-core instead of own Tensor

The original design envisioned a thin `Tensor` handle with all computation through Ops.
We switched to candle-core because:

1. **Performance parity with origin** — origin uses candle-core directly. Matching the
   tensor backend eliminates an entire class of overhead (storage conversion, D2D copies).
2. **Mature CUDA backend** — candle handles matmul dispatch, memory management, dtype cast,
   and basic ops with well-tested CUDA kernels.
3. **Less code to maintain** — deleted ~1500 lines (own storage, layout, shape, error types).

The trade-off: `prelude-core` now depends on candle-core (with optional CUDA feature).
This means prelude-core is no longer fully device-agnostic at the type level. In practice
this is fine — candle-core is a Rust library with no C++ in CPU-only mode, and the CUDA
feature is gated behind `prelude-core/cuda`.

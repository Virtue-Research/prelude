# Writing Custom CUDA Ops

This guide covers how to add fused CUDA kernels to Prelude. Prelude uses PTX compilation at build time with cudarc runtime loading -- no CUDA C++ compilation at runtime.

## Overview

The pattern for every fused kernel in Prelude is:

1. Write CUDA kernel in `.cu` file
2. Compile to PTX at build time (via `build.rs`)
3. Load PTX module at runtime via cudarc
4. Wrap in a safe Rust function

## File Structure

```
crates/prelude-core/src/ops/gpu/
  mod.rs              -- module exports, PTX loading
  elementwise.rs      -- vectorized_add, fast_silu_mul
  rmsnorm.rs          -- fast_rmsnorm, fused_add_rmsnorm
  rope.rs             -- fused_qknorm_rope_varlen
  kv_cache.rs         -- fused_knorm_rope_kv_cache_write_varlen
  moe.rs              -- MoE routing ops
  kernels/            -- .cu source files
    add.cu
    silu_mul.cu
    rmsnorm.cu
    add_rmsnorm.cu
    qknorm_rope.cu
    knorm_rope_kv_write.cu
    moe_routing.cu
```

## Existing Kernels

| Kernel | File | Purpose | Inputs | Launch Config |
|--------|------|---------|--------|---------------|
| `vectorized_add` | `elementwise.rs` | BF16 vector add (8 BF16/thread, 128-bit loads) | Two BF16 tensors | 256 threads |
| `fast_silu_mul` | `elementwise.rs` | Fused SiLU(x) * y | Two BF16 tensors | 256 threads |
| `fast_rmsnorm` | `rmsnorm.rs` | BF16 RMSNorm | Input + weight + eps | 256 threads |
| `fused_add_rmsnorm` | `rmsnorm.rs` | Residual add + RMSNorm in one pass | Residual + input + weight + eps | 256 threads |
| `fused_qknorm_rope` | `rope.rs` | Q/K per-head RMS norm + RoPE | Q/K + norm weight + cos/sin tables | Per-head |
| `fused_knorm_rope_kv_write` | `kv_cache.rs` | K-norm + RoPE + paged KV cache write | K/V + norm weight + cos/sin + cache + slot_mapping | 256 threads (8 warps) |

## Step-by-Step: Adding a New Kernel

### 1. Write the CUDA kernel

Create `kernels/my_kernel.cu`:

```cuda
extern "C" __global__ void my_kernel_bf16(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // your computation here
        output[idx] = input[idx];
    }
}
```

**Tips:**
- Use `__nv_bfloat16` (not `half`) for BF16
- Use `__restrict__` on all pointers
- Use vectorized loads (`float4`) when possible for memory bandwidth
- Suffix kernel names with `_bf16` or `_f32` for dtype variants

### 2. Compile to PTX

Add to `build.rs` in `prelude-core`:

```rust
// In the cuda build section
compile_ptx("src/ops/gpu/kernels/my_kernel.cu", "my_kernel.ptx");
```

The PTX file is written to `OUT_DIR` and included at compile time.

### 3. Load the PTX module

In `ops/gpu/mod.rs`, add the PTX loading:

```rust
const PTX_MY_KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/my_kernel.ptx"));
const MOD_MY_KERNEL: &str = "my_kernel_module";
```

### 4. Write the Rust wrapper

Create `ops/gpu/my_kernel.rs`:

```rust
use candle_core::{CudaDevice, CudaStorage, Result, Tensor};
use super::{PTX_MY_KERNEL, MOD_MY_KERNEL};

pub fn my_operation(input: &Tensor) -> Result<Tensor> {
    let device = input.device();
    let cuda_dev = match device {
        candle_core::Device::Cuda(dev) => dev,
        _ => candle_core::bail!("my_operation requires CUDA"),
    };

    let (storage, layout) = input.storage_and_layout();
    let input_slice = match &*storage {
        candle_core::Storage::Cuda(s) => s.as_cuda_slice::<half::bf16>()?,
        _ => candle_core::bail!("expected CUDA storage"),
    };

    let n = layout.shape().elem_count();
    let output = unsafe { cuda_dev.alloc::<half::bf16>(n) }?;

    // Load module (cached by cudarc)
    cuda_dev.load_ptx(PTX_MY_KERNEL.into(), MOD_MY_KERNEL, &["my_kernel_bf16"])?;
    let func = cuda_dev.get_func(MOD_MY_KERNEL, "my_kernel_bf16").unwrap();

    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;

    unsafe {
        func.launch(
            cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (input_slice, &output, n as i32),
        )?;
    }

    // Wrap output into a Tensor
    let storage = candle_core::CudaStorage::wrap_cuda_slice(output, cuda_dev.clone());
    Ok(Tensor::from_storage(storage.into(), layout.shape(), device))
}
```

### 5. Export and use

In `ops/gpu/mod.rs`:

```rust
mod my_kernel;
pub use my_kernel::my_operation;
```

Call from model code:

```rust
#[cfg(feature = "cuda")]
let result = crate::ops::gpu::my_operation(&input)?;

#[cfg(not(feature = "cuda"))]
let result = fallback_cpu_implementation(&input)?;
```

## Runtime Disable for Debugging

Fused kernels can introduce numerical drift. To debug, use an environment variable to disable at runtime:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use crate::config::parse_env_bool;

static MY_KERNEL_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn init() {
    if parse_env_bool("PRELUDE_NO_MY_KERNEL") {
        MY_KERNEL_ENABLED.store(false, Ordering::Relaxed);
    }
}

pub fn my_operation(input: &Tensor) -> Result<Tensor> {
    if !MY_KERNEL_ENABLED.load(Ordering::Relaxed) {
        return fallback(input);
    }
    // ... fused kernel path
}
```

This way users can run `PRELUDE_NO_MY_KERNEL=1` to fall back to the unfused path without recompilation.

## Profiling

```bash
# Profile with nsys
nsys profile --trace=cuda ./target/release/prelude-server --model Qwen/Qwen3-4B

# Enable sync timing to get accurate per-kernel times
PRELUDE_SYNC_TIMING=1 ./target/release/prelude-server --model Qwen/Qwen3-4B
```

## Common Pitfalls

- **BF16 precision:** Accumulate in FP32, convert back to BF16 at the end. BF16 has only 8 bits of mantissa.
- **Bank conflicts:** Use padding in shared memory to avoid 32-way bank conflicts.
- **PTX vs CUBIN:** Prelude uses PTX (JIT-compiled by driver at load time). This provides portability across SM architectures but adds a small first-load latency.
- **Thread divergence:** Avoid branching within a warp. Use predicated execution or separate kernels for different code paths.
- **Memory coalescing:** Adjacent threads should access adjacent memory. BF16 vectorized loads (128-bit = 8 BF16 values) are preferred.

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
crates/prelude-cuda/src/
  ops/                       -- kernel wrapper modules
    mod.rs                   -- module exports, PTX loading
    elementwise.rs           -- vectorized_add, fast_silu_mul
    rmsnorm.rs               -- fast_rmsnorm, fused_add_rmsnorm
    rope.rs                  -- fused_qknorm_rope_varlen
    kv_cache.rs              -- fused_knorm_rope_kv_cache_write_varlen
    moe.rs                   -- MoE routing ops
    gemm.rs                  -- GEMM wrappers
    quant.rs                 -- quantization ops
    tiled_mmq.rs             -- tiled MMQ kernels
  kernels/kernels_src/       -- .cu source files (organized by subdirectory)
    elementwise/
      add.cu
      silu_mul.cu
    normalization/
      rmsnorm.cu
      add_rmsnorm.cu
    rope/
      qknorm_rope.cu
    kvcache/
      knorm_rope_kv_write.cu
      scatter_kv_cache.cu
      append.cu
    moe/
      routing.cu
      gateup.cu
      down.cu
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

## Stride-aware contract for packed-layout kernels

**Rule:** any custom CUDA kernel that consumes a `[num_tokens, num_heads, head_dim]`-style
packed layout **must read strides from the candle layout and honor them**, never assume
contiguous `row * d` indexing.

Why: model code produces strided Q/K/V views from fused QKV narrow
(`qkv_out.narrow(1, 0, q_size).reshape((total, H, D))`) without calling `.contiguous()`. The
stride-0 element count is `N_fused = q_size + 2*kv_size`, not `H * D`. A kernel that indexes
via `input + row * d` reads the wrong memory and silently corrupts output. We have been bitten
by this (PPL jumped to 3221 before we caught it).

The only constraint the stride-aware path imposes on callers is `stride(-1) == 1` (head_dim
contiguous). Everything else is free. This matches vLLM's convention and FlashInfer / FA4's
upstream C++ APIs.

**Implementation pattern** (see `crates/prelude-cuda/src/kernels/kernels_src/rope/qknorm_rope.cu`
and the scatter-write kernel in `kvcache/scatter_kv_cache.cu` for reference):

1. Kernel signature takes `uint32_t token_stride` (or `key_stride` / `value_stride`) as an
   explicit parameter.
2. Row addressing uses `input + (uint64_t)token * token_stride + (uint64_t)head * d`.
3. Rust wrapper extracts the stride from candle's layout:
   ```rust
   let token_stride = x_layout.stride()[0] as u32;
   // ... pass to kernel launch
   ```
4. `debug_assert!(x_layout.stride()[2] == 1)` to catch non-stride-1 last dim at dev time.

**FFI wrappers that take raw pointers** (FA4, FlashInfer) must still pass strides through
`DLTensor.strides` — never hardcode `contiguous_strides(shape)`. Both upstream kernels read
`q.stride(0) / q.stride(1)` at runtime; the binding's job is to forward the real stride from
candle's layout. See `crates/prelude-cuda/fa4/src/lib.rs::fa4_varlen_fwd` and
`crates/prelude-cuda/src/attn/flashinfer.rs::ragged_prefill` for reference.

Kernels that currently comply: `fused_qknorm_rope_bf16`, `scatter_kv_cache_flash`,
`fa4_varlen_fwd` / `fa4_varlen_paged_fwd`, `flashinfer::ragged_prefill` /
`paged_prefill_fast` / `paged_decode`. New kernels joining this set must follow the same
convention.

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

Add to `build.rs` in `prelude-cuda`:

```rust
// In the cuda build section
compile_ptx("src/kernels/kernels_src/my_kernel.cu", "my_kernel.ptx");
```

The PTX file is written to `OUT_DIR` and included at compile time.

### 3. Load the PTX module

In `ops/mod.rs` (inside `prelude-cuda`), add the PTX loading:

```rust
const PTX_MY_KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/my_kernel.ptx"));
const MOD_MY_KERNEL: &str = "my_kernel_module";
```

### 4. Write the Rust wrapper

Create `ops/my_kernel.rs` (inside `prelude-cuda/src/`):

```rust
use prelude_core::tensor::{bail, Result, Tensor, Device};
use crate::device::{CudaStorage, as_cuda_slice_bf16, load_ptx, get_func, cuda_stream};
use super::{PTX_MY_KERNEL, MOD_MY_KERNEL};

pub fn my_operation(input: &Tensor) -> Result<Tensor> {
    let device = input.device();
    let ordinal = match device {
        Device::Cuda(ord) => *ord,
        _ => bail!("my_operation requires CUDA"),
    };

    let input_slice = as_cuda_slice_bf16(input)?;
    let n = input.elem_count();
    let stream = cuda_stream(ordinal)?;
    let output = unsafe { stream.ctx().alloc::<half::bf16>(n) }?;

    // Load module (cached by device.rs registry)
    load_ptx(ordinal, PTX_MY_KERNEL, MOD_MY_KERNEL, &["my_kernel_bf16"])?;
    let func = get_func(ordinal, MOD_MY_KERNEL, "my_kernel_bf16")?;

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
    let storage = CudaStorage::wrap_cuda_slice(output, ordinal);
    Ok(Tensor::from_storage(storage.into(), input.shape(), device))
}
```

### 5. Export and use

In `ops/mod.rs` (inside `prelude-cuda`):

```rust
mod my_kernel;
pub use my_kernel::my_operation;
```

Model code does **not** call kernel functions directly. Instead, expose the operation through an Ops trait (e.g., `FusedOps`) so that models remain device-agnostic:

```rust
// In prelude-cuda: implement the trait method to call your kernel
impl FusedOps for CudaFusedOps {
    fn my_operation(&self, input: &Tensor) -> Option<Result<Tensor>> {
        Some(my_kernel::my_operation(input))
    }
}

// In model code (prelude-core): call via Ops trait, no #[cfg] needed
let result = match ops.fused.my_operation(&input) {
    Some(r) => r?,
    None => fallback_cpu_implementation(&input)?,
};
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

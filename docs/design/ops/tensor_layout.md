## Tensor Layout Conventions

All op traits use a shared set of tensor layout conventions. These are **part of the contract**
— every device implementation must accept and produce tensors in these layouts.
Implementations that need different internal layouts (e.g., TPU 128-byte alignment)
must transpose/pad internally and return the canonical layout.

```
Model data (Tensor — lives on device):
  Q, K, V (attention):     [total_tokens, num_heads, head_dim]       — varlen (packed batch)
  O (attention output):     [total_tokens, num_heads, head_dim_v]
  key_cache, value_cache:   [num_blocks, block_size, num_heads_k, head_dim]  — paged KV
  Linear weights:           [out_features, in_features]               — row-major
  Norm weights:             [hidden_dim]
  Bias:                     [out_features]
  Conv1d input:             [batch, channels, length]
  Conv2d input:             [batch, channels, height, width]

Scheduling metadata (&[u32] — plain host-side data, device uploads internally):
  cu_seqlens:               [batch_size + 1]                          — cumulative sequence offsets
  block_tables:             [batch_size * max_blocks_per_seq]         — flattened block indices
  slot_mapping:             [total_tokens]                            — flat slot indices
```

**Why scheduling metadata is `&[u32]`, not `Tensor`:**
Scheduling metadata (cu_seqlens, block_tables, slot_mapping) describes batch structure,
not model computation. It is constructed by the engine on the host. The trait boundary
uses `&[u32]` to keep the interface device-agnostic — no assumptions about where integer
metadata lives or what type it uses on the device side.

**Device implementations convert internally — no performance overhead:**
The `&[u32]` at the trait boundary does NOT mean the kernel sees host memory.
Each device impl converts to its optimal internal representation:
- **CUDA/ROCm**: maintains a pre-allocated GPU buffer internally. On each call,
  async memcpy from `&[u32]` into the GPU buffer (overlaps with kernel launch),
  then passes the GPU pointer to FlashInfer/FA4. Zero extra cost vs passing a Tensor
  — the memcpy was always needed (engine computes metadata on CPU).
- **Metal**: unified memory. The `&[u32]` slice is already GPU-accessible, zero-copy.
- **TPU**: folds the values into XLA trace as compile-time constants.
- **CPU/Vulkan**: uses the slice directly.

The trait says "what" (u32 scheduling data). The device impl decides "where" (GPU buffer,
unified memory, XLA constant). This is the same encapsulation as the rest of the design.

**Why packed varlen** (`[total_tokens, ...]` with `cu_seqlens`) instead of padded batch (`[batch, max_seq, ...]`):
- No wasted compute on padding tokens.
- Natural for continuous batching (variable-length sequences in one batch).
- Required by FlashAttention, FlashInfer, CK, Pallas — all major attention kernels.
- Diffusion uses this too: batch of images with different token counts → single packed tensor.

**Device-internal exceptions** (invisible to model code):
- TPU pads head_dim to 128-byte alignment internally.
- Metal may transpose for simdgroup_multiply_accumulate efficiency.
- Vulkan may pad workgroup-aligned dimensions.
These are implementation details — the trait boundary always uses canonical layouts.

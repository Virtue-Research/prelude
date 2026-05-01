# FlashInfer Integration

FlashInfer is one of two GPU attention backends in prelude. It covers SM80+
(Ampere/Ada via FA2) and SM90+ (Hopper via FA3) as a fallback to
[Flash Attention v4](../crates/prelude-cuda/fa4/README.md) (SM90+ TMA path).

**Dispatch order** (GPU): FA4 → FlashInfer → composed F32 SDPA.
Set via `prelude_cuda::cuda_ops::varlen_attention` / `paged_attention`.

## Crate layout

```
crates/prelude-cuda/flashinfer/
├── Cargo.toml
├── build.rs                    # AOT build pipeline
├── scripts/compile_kernels.py  # Generate + compile FlashInfer kernel variants
└── src/
    ├── lib.rs                  # Public API
    ├── loader.rs               # KernelRegistry, PrefillVariant, DecodeVariant
    └── types.rs                # TVM-FFI + DLPack types (shared ABI with FA4)
```

The upstream FlashInfer source is a submodule at `third_party/flashinfer/`;
the `FLASHINFER_SRC` env var can override this for local development.

## Key design points

- **KV write stays ours** — we keep `scatter_kv_cache_flash`
  (`crates/prelude-cuda/src/ops/kv_cache.rs`, PTX kernel) for the cache
  write. FlashInfer's `append_paged_kv_cache` uses
  `batch_indices + positions`, which doesn't match our `slot_mapping`
  approach.

- **Plan caching** — FlashInfer requires a host-side `plan()` step before
  `run()`. Without caching we'd call `plan()` per-layer-per-step
  (28 × per decode step on Qwen3-0.6B → ~6× decode slowdown from the
  `to_vec1()` GPU→CPU copies alone). `crate::attn::flashinfer` caches the
  plan once per `model.forward()` via thread-local `PlanCache`;
  `begin_forward()` / `end_forward()` bracket each forward pass.

- **Flash layout only** — FlashInfer NHD `[num_blocks, block_size, num_kv_heads, head_dim]`
  matches our paged-KV layout. No cache-format conversion needed.

- **Symbol collision avoidance** — every compiled kernel variant's TVM-FFI
  exports get unique renames at codegen time
  (e.g. `plan` → `fi_prefill_fa2_bf16_h128_plan`) so static linking of ~70
  variants doesn't trip duplicate symbols.

- **U32 → I32 indptr** — candle's `cu_seqlens` are U32, FlashInfer expects
  I32. We pass the U32 raw pointer with an I32 DLTensor header (identical
  bit pattern) rather than a second cast kernel.

## Plan host-side requirement

FlashInfer's `plan()` reads `qo_indptr`, `kv_indptr`, and `kv_len_arr` on
**CPU**, not GPU. The dispatch layer `to_vec1()`s them before building the
plan-side DLTensors, and keeps GPU DLTensors only for the run step. If
a plan call segfaults, the first thing to check is which side of the
CPU/GPU split each indptr is living on.

## Environment variables (build-time)

| Variable | Default | Description |
|---|---|---|
| `FLASHINFER_SRC` | `third_party/flashinfer` | Path to FlashInfer source checkout |
| `PRELUDE_FLASHINFER_ARCHS` | `sm_80,sm_90` | Target SM architectures |
| `PRELUDE_FLASHINFER_HEAD_DIMS` | `64,96,128` | Head dimensions to compile |
| `PRELUDE_FLASHINFER_DTYPES` | `bf16` | Data types to compile |
| `PRELUDE_FLASHINFER_WORKERS` | `min(cpus, 8)` | Parallel compile workers |
| `PRELUDE_FLASHINFER_MLA_DIMS` | — | Optional MLA head dims to compile |

## Known limitations

- **Blackwell (SM100/SM103)** — upstream FlashInfer's prefill kernels are
  compiled as SM90-only `BatchPrefillWithPagedKVCacheSM90Run`; on B300 they
  fail at load time with "no kernel image is available for execution on the
  device". FA4 covers the SM90+ prefill path, so FlashInfer is
  effectively a decode-only backend on Blackwell until upstream adds
  Blackwell prefill kernels.

- **Group sizes** — `DISPATCH_GQA_GROUP_SIZE` in upstream FlashInfer only
  instantiates `{1, 2, 3, 4, 8}`. Models with `num_qo_heads / num_kv_heads`
  outside that set (e.g. Qwen3-14B = 5, Qwen3-32B = 6, Llama-3-70B = 7)
  crash on the decode path. Prelude carries a local patch that extends
  the set to `{1..8, 16}`; this lives as an uncommitted modification to
  the vendored `third_party/flashinfer/` submodule and should be upstreamed.

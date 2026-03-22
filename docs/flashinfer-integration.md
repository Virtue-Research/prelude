# FlashInfer Integration

Branch: `feat/flashinfer-integration`

## Goal

Replace 3 attention backend crates with FlashInfer, reducing to only 2 upstream dependencies:

| Before | After |
|---|---|
| `candle-flash-attn` (FA2, SM80+) | **removed** |
| `candle-flash-attn-v3` (FA3, SM90) | **removed** |
| `candle-paged-attn` (vLLM paged decode) | **removed** |
| `prelude-flash-attn-v4` (FA4, SM90+) | **kept** |
| — | **`prelude-flashinfer`** (SM80+, FA2/FA3 backend) |

FA4 handles SM90+ (Hopper/Blackwell). FlashInfer handles SM80+ (Ampere/Ada) and serves as fallback.

## What's Done

### 1. New crate: `crates/prelude-flashinfer/`

```
crates/prelude-flashinfer/
├── Cargo.toml
├── build.rs                      # AOT build pipeline
├── scripts/
│   └── compile_kernels.py        # Generates & compiles FlashInfer kernel variants
└── src/
    ├── lib.rs                    # Public API
    ├── types.rs                  # TVM FFI + DLPack types (shared ABI with FA4)
    └── loader.rs                 # KernelRegistry, PrefillVariant, DecodeVariant
```

**Build pipeline** (mirrors FA4 pattern):
1. Find FlashInfer source (`FLASHINFER_SRC` env var)
2. Run `compile_kernels.py` → Jinja template → `.cu` → nvcc → `.o`
3. Archive `.o` into `libflashinfer_kernels.a` (static)
4. Compile vendored TVM FFI (reuses FA4's `vendor/tvm_ffi/`)
5. Generate `fi_dispatch.rs` from `manifest.json`

**Symbol collision solution**: Each kernel variant's TVM FFI exports are renamed during source generation:
```
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, ...)
→ TVM_FFI_DLL_EXPORT_TYPED_FUNC(fi_prefill_fa2_bf16_h128_plan, ...)
```
This produces unique `__tvm_ffi_fi_prefill_fa2_bf16_h128_plan` symbols, safe for static linking.

**Kernel variants compiled**:
- `batch_decode` (FA2): per (dtype, head_dim) → plan + run
- `batch_prefill` (FA2): per (dtype, head_dim) → plan + ragged_run + paged_run
- `batch_prefill` (FA3/SM90): per (dtype, head_dim) → plan + ragged_run + paged_run

Default config: BF16, head_dims 64/96/128, no sliding window, no softcap.

### 2. Feature flag: `flashinfer`

In `prelude-core/Cargo.toml`:
```toml
flashinfer = ["cuda", "prelude-flashinfer"]
```

### 3. Attention dispatch: `attn/mod.rs`

Priority: **FA4 → FlashInfer → FA3 → FA2 → CPU**

FlashInfer branches added to all 5 dispatch functions:
- `varlen_attention` (paged + non-paged paths)
- `varlen_attention_windowed`
- `varlen_attention_bidirectional`
- `varlen_attention_paged` (fused KV write path)
- `reshape_and_cache` (FlashInfer uses flash layout, same as FA3/FA4)

### 4. Dispatch module: `attn/flashinfer.rs`

- Stub implementations for `varlen_causal`, `varlen_bidirectional`, `varlen_windowed`, `varlen_paged`
- `convert_paged_metadata()` implemented: converts our `block_tables + cu_seqlens_k` to FlashInfer's `paged_kv_indptr + paged_kv_indices + paged_kv_last_page_len`
- Global `KernelRegistry` via `OnceLock`

### 5. KV cache layout

FlashInfer NHD layout = our flash layout: `[num_blocks, block_size, num_kv_heads, head_dim]`.
No cache format conversion needed. KV write uses existing `scatter_kv_cache_flash` PTX kernel.

## What's Done (2026-03-20)

### Phase 1: First AOT compilation — DONE

FlashInfer source cloned at `/scratch/yuzhou/projects/flashinfer`. Python deps (jinja2, tvm_ffi) pre-installed in `.venv`.

**Bugs fixed in `compile_kernels.py`:**
1. `do_compile` local function → moved to module level (ProcessPoolExecutor pickling)
2. `ADDITIONAL_PARAMS_SETTER` macro missing `\` line continuation → `" \\\n".join()`
3. `AdditionalParams` struct used `TensorView` (TVM FFI type) instead of raw pointers → rewrote to match FlashInfer's `generate_additional_params()` format
4. SM90 `variant_decl` pointed to FA2 `variants.cuh` → changed to `hopper/variants.cuh`
5. Missing CUTLASS include paths for SM90 → added `3rdparty/cutlass/{include,tools/util/include}`

**Result:** 25 `.o` files, 3 variants: decode FA2, prefill FA2, prefill FA3 (SM90). All 4 mask modes (None/Causal/Custom/MultiItemScoring).

6. `kMultiItem` → `kMultiItemScoring` (enum name mismatch with this FlashInfer version)
7. All 4 mask modes needed (0/1/2/3) — `DISPATCH_MASK_MODE` macro generates switch for all modes
8. FA2 arch flags: dual-target `compute_80` PTX + `sm_90` SASS for forward compatibility

### Phase 2: FFI wrappers — DONE

`attn/flashinfer.rs` fully implemented with plan-then-run FFI calls (FA2 backend):

- **Workspace**: 128MB GPU float + 8MB GPU int + 8MB CPU pinned, allocated once via `cudaMalloc`/`cudaMallocHost`
- **`varlen_causal`**: prefill plan + `ragged_run` (MaskMode::Causal)
- **`varlen_bidirectional`**: prefill plan + `ragged_run` (MaskMode::NonCausal)
- **`varlen_windowed`**: prefill plan + `ragged_run` with `window_left` param
- **`varlen_paged`**: decode plan+run (Q=1) or prefill plan+paged_run (Q>1)
- **`convert_paged_metadata`**: block_tables → FlashInfer indptr/indices/last_page_len

All additional params follow FlashInfer's `generate_additional_params()` format (raw pointers in struct, `Optional<ffi::Tensor>` in binding signature, pointer extraction in setter macro).

**Not yet implemented:** FA3 prefill path (SM90 Hopper). Currently uses FA2 for all paths. FA3 has different additional_params (fewer args, `additional_params` nested struct). Can add once FA2 path is validated.

### cfg gate cleanup — DONE

Replaced hardcoded `#[cfg(feature = "flash-attn-v3")]` gates across the codebase with `#[cfg(any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer"))]`. These gates control:

- **Cache layout** (flash NHD vs v1): `cache/manager.rs`, `cache/paged.rs`, `engine/types.rs`
- **Forward pipeline** (prefill/decode/paged): `engine/forward/*.rs`
- **Runtime** (continuous/batch/gpu_queue): `runtime/*.rs`
- **Config** (default block_size): `config.rs`
- **Model caps** (`supports_varlen`): all model `meta.rs` files → changed to `cfg!(feature = "cuda")`
- **Server feature**: added `flashinfer = ["prelude-core/flashinfer"]` to `prelude-server/Cargo.toml`
- **`OwnedBatchDecodeSeq`**: expanded gate to include flashinfer

**Important**: `attn/mod.rs` keeps per-backend `#[cfg]` gates — that's the dispatch layer and should NOT be changed.

**U32/I32 dtype issue**: candle's cu_seqlens are U32, FlashInfer expects I32. `candle-kernels` doesn't have `cast_u32_i32`. Fix: pass U32 raw pointers with I32 DLTensor dtype (same bit pattern, no cast needed). `convert_paged_metadata` pulls to CPU as `u32` then converts via `as i32`.

### Phase 3: Runtime debugging — DONE

**Build**: `cargo build -p prelude-server --release --features flashinfer` compiles and links successfully.

**Server startup**: Server starts, health check passes, model loads correctly.

**FFI test binary**: `crates/prelude-flashinfer/examples/test_ffi.rs` — minimal standalone test that calls FlashInfer TVM FFI directly. Plan + ragged_run + decode plan all succeed.

**Bugs fixed (2026-03-20)**:

1. **Plan segfault — CPU indptrs**: FlashInfer's plan functions read `qo_indptr`, `kv_indptr`, and `kv_len_arr` on the **CPU** (not GPU). Our Rust code was passing GPU DLTensors, causing the plan to segfault when trying to dereference device pointers from host code. Fix: pull indptrs to CPU with `to_vec1()`, create CPU DLTensors for plan, use GPU DLTensors for run.

2. **Ragged_run segfault — null strides**: TVM's `TensorView::stride()` asserts `strides != nullptr` (`ICHECK`). Our DLTensors had `strides: null_ptr()` (DLPack allows NULL to mean contiguous). Fix: always compute and pass explicit contiguous strides for all DLTensors.

3. **U32/I32 block_tables**: `block_tables` are U32 in the engine, but `convert_paged_metadata` was calling `to_vec2::<i32>()` which fails. Fix: `to_vec2::<u32>()` then cast to i32.

4. **Rope defaults**: Changed `rope_rcp_scale` from 0.0 to 1.0 and `rope_rcp_theta` from 0.0 to 1e4 (matching FlashInfer Python defaults).

**Accuracy test results**: 10/10 passed (Qwen/Qwen3-0.6B, BF16, vs HF transformers):
- 3 exact text matches
- 7 close matches (all bidirectional cross-containment OK at first divergence)
- Max logprob diff: 1.19

```bash
# Run minimal FFI test
cd crates/prelude-flashinfer
CUDA_VISIBLE_DEVICES=0 FLASHINFER_SRC=/scratch/yuzhou/projects/flashinfer \
  cargo run --example test_ffi --release

# Build server
FLASHINFER_SRC=/scratch/yuzhou/projects/flashinfer \
  cargo build -p prelude-server --release --features flashinfer

# Run accuracy test
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/accuracy/run_accuracy_test.py \
  --variant gpu --server prelude \
  --binary target/release/prelude-server \
  --model Qwen/Qwen3-0.6B
```

### Performance gap analysis (2026-03-21)

**Root cause of ~6x decode slowdown**: We call `plan()` inside every `varlen_attention()` → every layer × every step. For Qwen3-0.6B (28 layers), this means **28 plan calls per step** instead of 1. Each plan call does `to_vec1()` (GPU→CPU copy) + CPU-side scheduling computation.

SGLang/vLLM usage: `handler.plan()` once per batch step → `handler.run(q,k,v)` per layer. Plan only depends on sequence lengths, not on Q/K/V data.

**Fix — DONE**: Thread-local plan cache in `flashinfer.rs`. `begin_forward()` / `end_forward()` bracket each `model.forward()` call. First attention layer computes and caches the plan; layers 1-27 reuse it. No signature changes needed — the cache is transparent to models and the dispatch layer.

Implementation:
1. `flashinfer.rs`: `PlanCache` struct with thread-local `RefCell<Option<PlanCache>>`, three slots (ragged/decode/paged_prefill)
2. Engine forward paths: `fi_begin_forward()` / `fi_end_forward()` around every `model.forward()` call (cfg-gated)
3. Each `ragged_prefill()`, `paged_decode()`, `paged_prefill()` checks cache before computing plan — skips `to_vec1()` GPU→CPU copies + plan FFI call when cached

**Result (H200, Qwen3-0.6B, single request, 32 decode tokens)**:

| Metric | Before (28 plans/step) | After (1 plan/step) | Speedup |
|--------|----------------------|---------------------|---------|
| Decode TPS | 32.3 | 54.4 | **1.68x** |
| TPOT | 30.9ms | 18.4ms | **1.68x** |

### Phase 4: Remove old backends

Once FlashInfer passes accuracy tests:
1. Remove `candle-flash-attn` (FA2) feature and crate dependency
2. Remove `candle-flash-attn-v3` (FA3) feature and crate dependency
3. Remove `candle-paged-attn` crate dependency
4. Remove `flash_v2.rs`, `flash_v3.rs` from `attn/`
5. Update `reshape_and_cache` and `varlen_attention_paged` to remove old fallback paths
6. Update build docs and feature flags

## Key Design Decisions

1. **KV write stays ours**: We keep `scatter_kv_cache_flash` (PTX kernel) for KV cache writes. FlashInfer's `append_paged_kv_cache` uses a different indexing scheme (batch_indices + positions) that doesn't match our slot_mapping-based approach.

2. **Plan-then-run hidden behind same interface**: The plan step runs implicitly inside each `flashinfer::varlen_*` call, keeping the same function signatures as FA3/FA4. Can optimize later to plan once per batch if needed.

3. **Static linking with symbol renaming**: Each kernel variant's TVM FFI exports get unique symbol names to avoid collisions when statically linked. No runtime dlopen needed.

4. **Flash layout only**: FlashInfer NHD = our flash layout. No v1 layout support needed. This means `candle-paged-attn`'s v1 operations are fully replaced.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASHINFER_SRC` | auto-detect | Path to FlashInfer source |
| `PRELUDE_FLASHINFER_ARCHS` | `sm_80,sm_90` | Target SM architectures |
| `PRELUDE_FLASHINFER_HEAD_DIMS` | `64,96,128` | Head dimensions to compile |
| `PRELUDE_FLASHINFER_DTYPES` | `bf16` | Data types to compile |
| `PRELUDE_FLASHINFER_WORKERS` | `min(cpus, 8)` | Parallel compilation workers |

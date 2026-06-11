# Task A — porting fast+correct ragged causal varlen into candle-fa3-0102

Goal: make candle-fa3-0102's varlen path match vLLM-FA3 latency on the real ragged
prefill workload (currently: correct only for single/uniform; **hangs** on ragged causal;
and ~2.4x slower even when it runs, due to SingleTileScheduler).

## Precise diagnosis (this session, evidence-based)

1. **varlen causal correctness** — FIXED (commit c2ea0e29). Binding window-clamp turned
   causal into full attention; now single/uniform varlen causal is bit-identical to vLLM.

2. **ragged causal varlen HANGS** — root cause localized, NOT yet fixed:
   - `SingleTileScheduler` (tile_scheduler.hpp): grid = `num_blocks_m × num_head × num_batch`,
     1 tile/CTA, `num_blocks_m = ceil(MAX_seqlen_q / kBlockM)` (rectangular over the batch).
   - For ragged batches, short sequences have m_blocks beyond their `actual_seq_len`. Those
     CTAs early-exit via `continue` (flash_fwd_kernel.h:124 producer / :177 consumer),
     skipping all pipeline loads/MMAs, then unconditionally hit `load_tail` (:144) /
     `store_tail` (:208) on a never-used MainloopPipeline → **deadlock**.
   - Uniform batches (seqlen % kBlockM == 0) have no beyond-seqlen tiles → no hang.
   - Confirmed: `1743 1559` (ragged) hangs; `1743` and `2048 2048` run fine.

3. **~2.4x slower** even when running: 0.10.2 hardcodes `UseVarSeqLen → SingleTileScheduler`
   (flash_fwd_launch_template.h:36), no persistence → low SM utilization at small batch.
   Routing varlen to `DynamicPersistentTileScheduler` BREAKS correctness (cos 0.43) — that
   scheduler is not varlen-aware in 0.10.2.

## Port plan (the real work)

Bring the newer FA3 Hopper varlen machinery into this kernel. Reference sources in-repo:
the prelude fork `crates/candle-flash-attn-v3` (hkernel_v3 / FlashAttnFwdSm90) and vLLM's
vllm-flash-attn both have a working **persistent varlen scheduler** that:
  - precomputes per-(batch,head) block counts (`prepare_varlen_num_blocks`) so only real
    tiles are scheduled (no beyond-seqlen padding tiles → no empty-pipeline tail → no hang),
  - distributes tiles across all SMs (persistent) → fills the GPU at low batch.

Approaches:
- **A-graft from the prelude fork — RULED OUT.** Compared `crates/candle-flash-attn-v3`
  (the prelude fork): its `flash_fwd_launch_template.h` has the IDENTICAL
  `UseVarSeqLen → SingleTileScheduler` selection, and its `flash_fwd_kernel.h` has the SAME
  beyond-seqlen `continue` early-exits + unconditional `load_tail`/`store_tail`. So the fork
  would hang on ragged causal varlen too — it just isn't used for prelude's prefill (prelude
  prefills with FA4 / flashinfer). There is NO working fast varlen kernel in either candle
  fork in this repo to graft from.
- **A-swap (the only real path)**: import a newer FA3 (flash-attention 3.x, what vLLM bundles
  as vllm-flash-attn) hopper kernel that has the persistent varlen scheduler
  (`prepare_varlen_num_blocks` + FlashAttnFwdSm90 varlen) and adapt it to candle's binding +
  the CUTLASS pin. Needs the upstream source (network) and substantial integration +
  per-step correctness validation. Multi-session kernel engineering, not a same-session task.

This is multi-session kernel engineering. Interim safe fix for the HANG only (not speed):
make `load_tail`/`store_tail` safe for the no-work case, or skip the pipeline tail when a
CTA's tile had zero loaded blocks — but this must be done without desyncing producer/
consumer (high risk of silent-wrong-output; needs careful MainloopPipeline understanding).

Status: diagnosis complete; port not yet started (requires the reference kernel + careful
correctness validation against the xcheck_vs_vllm.py varlen path at each step).

---

## Endgame (2026-06-09): cubin transplant verdict → root cause `cp_world_size=0`

The port worked (correct + no hang) but ran 1.19x vLLM on the dataset. The residual was
chased with a no-sudo/no-ncu forensic pipeline; conclusion: NOT a build-artifact difference.

1. **Source/flags audit**: env vLLM = 0.22.0, pins `vllm-project/flash-attention@bce2942`.
   Vendored hopper sources diff CLEAN vs that commit (only our off-by-default clock probe);
   CUTLASS submodule identical (62750a2b / 3.8.0). Macro/flag deltas vs their CMake only
   prune host dispatch, not the measured kernel instance.
2. **ELF/SASS**: same mangled kernel on both sides: REG:168, LOCAL:0 (no spill). SASS body
   5560 (cc12.9) / 5512 (cc13.2) / 5520 (vLLM, cc13.0) instructions — noise-level.
3. **Param-layout trap**: the SAME source compiles to a 2240-byte kernel param block on
   CUDA 12.9 but 2688 bytes on CUDA 13.x (CUTLASS has version-gated struct members).
   A 12.9-built host launching the 13.0 vLLM cubin = illegal memory access. Host must be
   rebuilt with 13.x before any transplant A/B.
4. **Cubin transplant (driver API hook in flash_fwd_launch_template.h, build with
   -DFA3_TRANSPLANT_HOOK, env FA3_CUBIN=path)**: with hosts aligned on 13.2 —
   ours 339.8us ≈ vLLM's cubin 342.5us in OUR harness (bit-identical output), while the
   same cubin does 250.8us inside vLLM. Verdict: artifact hypothesis DEAD; the slowdown is
   something we pass into the kernel.
5. **Param diff (LD_PRELOAD dump of vLLM's cudaLaunchKernelExC arg block vs ours)**:
   the ONLY non-pointer scalar delta was `mainloop.cp_world_size`: ours 0, vLLM 1.
   `flash_api_v3.cu` zero-inits `Flash_fwd_params{}` and never set the context-parallelism
   fields. The mainloop computes `n_block_min_causal_local_mask` with a divide by
   `cp_world_size`; /0 yields a garbage bound, so EVERY KV block took the element-wise
   causal-mask slow path. Semantics unchanged (always bit-correct) — a pure perf bug,
   uniform across CTAs, which is exactly why the clock64 probe saw "perfectly balanced,
   uniformly slow".

Fix: `params.cp_world_size = 1; params.cp_rank = 0; params.cp_tot_seqused_k = nullptr;`
Follow-up: with the mask bug gone, PackGQA=true is now FASTER (274.6 vs 286.5 us/batch);
default flipped to `pack_gqa = (h != h_k)` to match vLLM's auto-pick (env `V3_PACKGQA`
still overrides).

Final numbers (H200, same harnesses as above):
- uniform 4x2048 kernel-only: candle v3 254us vs vLLM 251us (1.01x)
- dataset 1473 ragged batches: candle v3 402 ms/pass vs vLLM 399 (1.01x)
- xcheck vs vLLM: max_abs=0.0, cos=1.000000 (both GQA paths); test suite 14/15
  (hdim512 = pre-existing upstream smem limit, untouched by this work).

Methodology takeaway: "diff the kernel arguments" belongs on every forensic checklist;
a cubin transplant is the gold standard for bisecting artifact-vs-runtime without ncu,
but requires (a) prepare-kernel compatibility, (b) dynSmem opt-in via cuFuncSetAttribute,
(c) bit-identical re-verification, and (d) matching host/cubin toolchain MAJOR version.

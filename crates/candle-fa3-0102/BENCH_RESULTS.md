# FA3/FA4 kernel microbench — candle-fa3-0102 vs FA4 vs vLLM-FA3

**GPU:** NVIDIA H200 (sm90a) · single GPU
**Config:** batch=128 sequences × seqlen, GQA **32 q / 8 kv** heads, head_dim **128**,
**causal**, **bf16** (Qwen3-8B prefill shape). warmup 30, 200 iters.
**Date:** 2026-06-09.

All numbers use **random bf16 inputs** (the fair, apples-to-apples basis — FA3 IS mildly
data-dependent via online-softmax rescaling, so random is ~13–26% slower than zeros at
mid seqlen; see "Input data-dependence" below). candle uses CUDA mempool retention (matching
PyTorch/vLLM's caching allocator). candle timing is wall-clock incl. host launch (conservative);
vLLM is CUDA-event GPU-only.

## Latency — microseconds per forward (lower = better), AFTER optimization

| seqlen | candle-fa3-0102 | prelude FA4 | vLLM-FA3 | candle / vLLM |
|-------:|----------------:|------------:|---------:|--------------:|
|    128 |           133.7 |       237.6 |    172.8 | **0.77× (faster)** |
|   1024 |          2395.6 |      3058.8 |   2704.5 | **0.89× (faster)** |
|   2048 |          8332.6 |      8986.2 |   8758.7 | **0.95× (faster)** |
|   8192 |        115314.2 |    104889.2 | 116108.8 | **0.99× (tied)**  |

**candle-fa3-0102 now matches or beats vLLM-FA3 at every seqlen** (0.77–0.99×): clearly faster
at small seqlen (host overhead eliminated), converging to a tie at large seqlen (compute-bound,
where candle's `flash::compute_attn_ws` kernel ≈ vLLM's `FlashAttnFwdSm90`). FA4 wins at 8192.

### How it got there (two binding/host fixes; the FA3 kernel was never the bottleneck)

| seqlen | pristine (random) | +lse fix | +mempool | vLLM |
|-------:|------------------:|---------:|---------:|-----:|
|    128 |             ~1700 |     ~310 |    133.7 | 172.8 |
|   2048 |            ~30000 |    ~8400 |   8332.6 | 8758.7 |

1. **softmax_lse 128× over-allocation** (`src/lib.rs`): dense path allocated
   `b*128*nheads*seqlen_q` instead of `b*nheads*seqlen_q` → multi-GB cudaMalloc+memset+free
   per forward. nsys showed the FA3 kernel was only ~6.6ms at s=2048 while wall-clock was
   ~28ms; the gap was this alloc. Fix → ~4× faster.
2. **CUDA mempool retention** (bench harness): cudarc uses `cuMemAllocAsync` but the default
   pool release threshold is 0, so every free returns memory to the OS. Set
   `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = u64::MAX` → freed blocks reused (like PyTorch). Kills
   the residual ~220us/call alloc overhead at small seqlen.

## Takeaways

- **After two binding/host fixes, candle-fa3-0102 matches or beats vLLM-FA3 at every seqlen
  (0.77–0.99×).** The *pristine* 0.10.2 binding was 1.6–9× slower, but that was NOT the FA3
  kernel — nsys showed the kernel was only ~6.6ms at s=2048 (faster than vLLM's 8.06ms) while
  the wall-clock was ~28ms. The gap was per-call host overhead: a 128× softmax_lse
  over-allocation + un-retained CUDA mempool. Fixing both closed it. The FA3 *kernel* itself
  was always competitive (the old-CUTLASS-3.6.0 pin is not a perf problem here).
- **Input data-dependence:** FA3 forward is NOT data-independent — the online-softmax running
  max triggers accumulator rescaling more often with real data than with zeros (where the max
  stays 0). Random is ~13% slower at the kernel (s=2048: 7.46ms random vs 6.62ms zeros) and
  up to ~26% slower wall-clock at mid seqlen. All headline numbers above use random for fairness.
- **The FA3 kernel handles batch=128×seqlen=8192 fine** (random: 115.3 ms). An earlier "failure" at
  that config was a BENCHMARK-HARNESS bug, NOT FA3: the harness built bf16 inputs via
  `randn(f32).to_dtype(bf16)`, and **candle-core 0.10.2's CUDA elementwise launch config
  truncates the element count to u32** — `cuda_backend/mod.rs:1511` does
  `LaunchConfig::for_num_elems(el as u32)`, and cudarc's `for_num_elems` (launch.rs:31)
  computes `grid = n.div_ceil(1024)`. At `el = 2^32` (128·8192·32·128), `2^32 as u32 = 0`
  → grid `(0,1,1)` → `cudaLaunchKernel` → `CUDA_ERROR_INVALID_VALUE`. PROVEN: zeros-alloc and
  RNG-fill at 2^32 are fine; only `to_dtype` (the cast kernel) fails; B=127 (<2^32) works,
  B=128 (=2^32) fails; FA3 with raw bf16-zeros inputs runs at 2^32. This is a candle-core
  framework limit (~25 `as u32` sites: ANY single candle op on a ≥2^32-element tensor breaks),
  unrelated to FA3 (whose params are int64). The bench now builds bf16 inputs without a 2^32
  cast. Random bf16 at s=8192 is supplied via `gen_random_inputs.py` (torch, 64-bit) +
  candle's safetensors LOADIN path (pure memcpy, avoids the cast).
- **FA4** is fastest at 8192 (104.9ms vs candle 115.3 / vLLM 116.1); candle/vLLM ≈ tied there.

## Numerical equivalence — candle == vLLM (bit-identical)

Cross-checked candle-fa3-0102 vs vLLM-FA3 on IDENTICAL random bf16 inputs
(s=512, hq=32, hkv=8, d=128, causal), via `xcheck_vs_vllm.py`:

| comparison | max_abs | mean_abs | cosine |
|---|---|---|---|
| candle vs vLLM (gqa_pack=false) | **0.0** | **0.0** | **1.000000** |
| candle vs vLLM (gqa_pack=true)  | **0.0** | **0.0** | **1.000000** |
| candle vs fp32 eager ref | 8.2e-3 | 2.1e-4 | 0.999998 |
| vLLM   vs fp32 eager ref | 8.2e-3 | 2.1e-4 | 0.999998 |

candle and vLLM produce **bit-identical** output (both GQA paths), and both match the
fp32 eager reference to bf16 precision → both numerically correct. Bit-identity implies the
two FA3 forward kernels share the same core computation/accumulation order, so **the speed
gap is performance only (host overhead + launch/scheduling), not a numerical shortcut.**

## Caveats (methodology — not perfectly apples-to-apples)

- **candle**: dense `(batch,S,H,D)` path, **wall-clock** timed (Instant + device.synchronize),
  median — conservative (includes host launch overhead). The 128× softmax_lse over-alloc is
  FIXED and the CUDA mempool is retained, so per-call alloc overhead is now negligible.
- **vLLM-FA3 / FA4**: varlen (uniform `cu_seqlens` = batch) path, **CUDA-event** GPU-only timed;
  q/k/v allocated once outside the loop. FA4 reports mean-of-events, vLLM median.
- candle's timing basis is *stricter* (wall-clock incl. host vs vLLM's GPU-only events), yet
  candle still wins/ties → the result is robust. For uniform lengths, dense and varlen do the
  same work.

## Reproduce

- candle: `crates/candle-fa3-0102/_run_bench_candle.sh` (env `CAUSAL=1 GQA_PACK=1`) → `examples/bench_fa3_0102.rs`
- vLLM:   `crates/candle-fa3-0102/bench_vllm_fa3.py` (vllm env, `fa_version=3`)
- FA4:    main tree `prelude/crates/prelude-cuda/fa4/examples/bench_fa4_qwen3.rs`
  (`cargo run -p flash-attn-v4 --example bench_fa4_qwen3 --release`)

Note: candle causal required a binding fix (zero-init `tile_count_semaphore`) — see the commit
that adds it; pristine 0.10.2 causal crashes with a null-pointer atomic otherwise.

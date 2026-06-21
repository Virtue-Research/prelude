#!/usr/bin/env bash
# Build the vendored vllm-fa hopper kernel (bf16/hdim128/sm90 varlen + paged) into a
# static lib for PRELUDE (prelude-cuda fa3-0102 feature) to link.
#
# Differences vs _build_v3_kernel.sh (the candle-fa3-0102 standalone build):
#   - CUDA 13.2 (prelude-server links CUDA 13 runtime libs; mixing a 12.9-built host
#     object into a 13.x process risks cudaDeviceProp ABI skew. Report sec.8 verified
#     the v3 kernel compiles on 13.2 at identical speed: 281us vs 283us).
#   - Sources from the MAIN tree (this script's own directory), not the worktree.
#   - Adds the PagedKVNonTMA=true+PackGQA instantiation (decode-path paged kernel).
#   - Drops -DFLASHATTENTION_DISABLE_PAGEDKV (only gated flash_api.cpp's host dispatch,
#     which we don't compile; dropped for clarity).
#   - Exposes run_mha_v3_prelude (stream-aware, paged-capable) alongside run_mha_v3.
set -uo pipefail
# CUDA 13.2 root. No hardcoded default — prelude-cuda/build.rs owns the literal
# (FA3_0102_DEFAULT_CUDA_HOME) and sets this automatically; standalone runs must
# export CUDA_HOME_FA3 to the CUDA 13.2 toolkit root.
CUDA=${CUDA_HOME_FA3:?set CUDA_HOME_FA3 to the CUDA 13.2 toolkit root (build.rs sets this automatically)}
CUT=${CUTLASS38:?set CUTLASS38 to the CUTLASS 3.8 include dir (e.g. /path/to/cutlass/include)}
SRC="$(cd "$(dirname "$0")" && pwd)/hkernel_v3vllm"
OUT=${FA3_0102_PRELUDE_PREBUILT_DIR:?set FA3_0102_PRELUDE_PREBUILT_DIR to the output dir for libprelude_fa3_0102.a}
mkdir -p "$OUT"
DIS="-DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_FP8 -DFLASHATTENTION_DISABLE_FP16 -DFLASHATTENTION_DISABLE_HDIM64 -DFLASHATTENTION_DISABLE_HDIM96 -DFLASHATTENTION_DISABLE_HDIM192 -DFLASHATTENTION_DISABLE_HDIM256 -DFLASHATTENTION_DISABLE_SM8 -DFLASHATTENTION_DISABLE_APPENDKV -DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_CLUSTER"
COMMON="-I $SRC -I $CUT -gencode arch=compute_90a,code=sm_90a -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler -fPIC -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG $DIS"
echo "=== build prelude fa3-0102 kernel .a START $(date -u +%H:%M:%S) (CUDA=$CUDA) ==="
pids=()
for f in instantiations/flash_fwd_hdim128_bf16_sm90 instantiations/flash_fwd_hdim128_bf16_packgqa_sm90 instantiations/flash_fwd_hdim128_bf16_paged_packgqa_sm90 flash_prepare_scheduler flash_api_v3; do
  base=$(basename "$f")
  echo "compiling $base ..."
  $CUDA/bin/nvcc -c "$SRC/$f.cu" -o "$OUT/$base.o" $COMMON 2>"$OUT/$base.err" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
if [ $fail -ne 0 ]; then echo "=== COMPILE FAILED ==="; for e in "$OUT"/*.err; do echo "--- $e ---"; grep -iE 'error|fatal' "$e" | head -5; done; exit 1; fi
ar rcs "$OUT/libprelude_fa3_0102.a" "$OUT"/*.o
echo "=== DONE $(date -u +%H:%M:%S): $(ls -lh $OUT/libprelude_fa3_0102.a | awk '{print $5}') ==="
$CUDA/bin/nm -C "$OUT/libprelude_fa3_0102.a" 2>/dev/null | grep -E 'T run_mha_v3|prepare_varlen_num_blocks' | head

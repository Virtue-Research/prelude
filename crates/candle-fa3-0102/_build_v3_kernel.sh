#!/usr/bin/env bash
# Build the vendored vllm-fa hopper kernel (bf16/hdim128/sm90 varlen) into a static lib
# for candle-fa3-0102 to link. CUDA 12.9 + CUTLASS 3.8.0. torch-free.
set -uo pipefail
CUDA=/usr/local/cuda-12.9
CUT=/data/xueying/cutlass38/include
SRC=/data/xueying/benchmark/prelude-fa3-0102/crates/candle-fa3-0102/hkernel_v3vllm
OUT=/data/xueying/fa3_v3_prebuilt
mkdir -p "$OUT"
DIS="-DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_FP8 -DFLASHATTENTION_DISABLE_FP16 -DFLASHATTENTION_DISABLE_HDIM64 -DFLASHATTENTION_DISABLE_HDIM96 -DFLASHATTENTION_DISABLE_HDIM192 -DFLASHATTENTION_DISABLE_HDIM256 -DFLASHATTENTION_DISABLE_SM8 -DFLASHATTENTION_DISABLE_PAGEDKV -DFLASHATTENTION_DISABLE_APPENDKV -DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_CLUSTER"
COMMON="-I $SRC -I $CUT -gencode arch=compute_90a,code=sm_90a -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler -fPIC -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG $DIS"
echo "=== build v3 kernel .a START $(date -u +%H:%M:%S) ==="
pids=()
for f in instantiations/flash_fwd_hdim128_bf16_sm90 instantiations/flash_fwd_hdim128_bf16_packgqa_sm90 flash_prepare_scheduler flash_api_v3; do
  base=$(basename "$f")
  echo "compiling $base ..."
  $CUDA/bin/nvcc -c "$SRC/$f.cu" -o "$OUT/$base.o" $COMMON 2>"$OUT/$base.err" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
if [ $fail -ne 0 ]; then echo "=== COMPILE FAILED ==="; for e in "$OUT"/*.err; do echo "--- $e ---"; grep -iE 'error|fatal' "$e" | head -5; done; exit 1; fi
ar rcs "$OUT/libcandle_fa3_v3.a" "$OUT"/*.o
echo "=== DONE $(date -u +%H:%M:%S): $(ls -lh $OUT/libcandle_fa3_v3.a | awk '{print $5}') ==="
$CUDA/bin/nm -C "$OUT/libcandle_fa3_v3.a" 2>/dev/null | grep -E 'run_mha_v3|prepare_varlen_num_blocks|run_mha_fwd_' | head

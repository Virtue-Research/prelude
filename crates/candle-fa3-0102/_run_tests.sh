#!/usr/bin/env bash
# Run candle-fa3-0102's bundled flash-attn tests on GPU, validating the
# CUDA-12.9-built sm90a kernels against upstream's golden numerics.
# Same toolchain as _build_first.sh (CUDA 12.9 + mold). Test-only adaptations:
# candle-nn dev-dep re-added + crate-rename alias in the test file.
set -euo pipefail
PRE=/scratch/xueying/miniforge3/envs/prelude   # mold
CUDA=/usr/local/cuda-12.9                        # 12.9 toolkit (matches the built .a)
export PATH="$CUDA/bin:$PRE/bin:$PATH"
export CUDA_HOME="$CUDA" CUDA_ROOT="$CUDA" CUDA_PATH="$CUDA"
export CUDA_COMPUTE_CAP=90
export LD_LIBRARY_PATH="$CUDA/targets/x86_64-linux/lib:$PRE/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA/targets/x86_64-linux/lib:$PRE/lib:${LIBRARY_PATH:-}"
export RUSTFLAGS="-C link-arg=-fuse-ld=mold"
export CUDA_VISIBLE_DEVICES=0                     # single H200
cd /data/xueying/benchmark/prelude-fa3-0102/crates/candle-fa3-0102
echo "=== START $(date -u +%H:%M:%S) UTC ==="
echo "nvcc=$(which nvcc) [$(nvcc --version | grep -oE 'release [0-9.]+')] gpu=$CUDA_VISIBLE_DEVICES"
cargo test --release -j "$(nproc)" -- --nocapture --test-threads=1
echo "=== DONE rc=$? $(date -u +%H:%M:%S) UTC ==="

#!/bin/bash
# Build oneDNN FFI static library.
# The first build downloads oneDNN source (~200MB) and compiles it.
# Subsequent builds are incremental.
#
# Both libonednn_ffi.a and libdnnl.a are produced. The Rust build.rs
# links them statically so the final binary needs no extra .so files.
#
# Usage:
#   ./build.sh              # default: OpenMP threading
#   ./build.sh --seq        # sequential (no OpenMP)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

USE_OMP=ON
if [[ "${1:-}" == "--seq" ]]; then
    USE_OMP=OFF
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DONEDNN_FFI_USE_OMP="$USE_OMP"

cmake --build "$BUILD_DIR" -j "$(nproc)"

echo ""
echo "Built:"
ls -lh "$BUILD_DIR/libonednn_ffi.a"
ls -lh "$BUILD_DIR/_deps/onednn-build/src/libdnnl.a"

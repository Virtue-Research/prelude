// SPDX-License-Identifier: Apache-2.0
//
// `c10/cuda/CUDAException.h` shim providing the two macros the Dao-AILab
// causal-conv1d kernels actually call:
//
//   * `C10_CUDA_CHECK(expr)` — assert a cudaError_t return value is ok
//   * `C10_CUDA_KERNEL_LAUNCH_CHECK()` — check `cudaGetLastError()` after
//     a `<<<>>>` kernel launch
//
// Upstream uses these for diagnostic logging; we mirror that with
// `fprintf` to stderr so compilation doesn't drag in libtorch's
// exception machinery. The kernels tolerate failures silently and we
// add full error handling at the Rust FFI layer above.

#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>

#define C10_CUDA_CHECK(EXPR)                                                       \
  do {                                                                             \
    cudaError_t const __err = (EXPR);                                              \
    if (__err != cudaSuccess) {                                                    \
      std::fprintf(stderr,                                                         \
                   "[causal-conv1d] CUDA error at %s:%d: %s\n",                    \
                   __FILE__, __LINE__, cudaGetErrorString(__err));                 \
    }                                                                              \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                             \
  do {                                                                             \
    cudaError_t const __err = cudaGetLastError();                                  \
    if (__err != cudaSuccess) {                                                    \
      std::fprintf(stderr,                                                         \
                   "[causal-conv1d] kernel launch failed at %s:%d: %s\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(__err));                 \
    }                                                                              \
  } while (0)

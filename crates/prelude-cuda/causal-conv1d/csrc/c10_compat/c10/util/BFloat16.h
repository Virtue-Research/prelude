// SPDX-License-Identifier: Apache-2.0
//
// Minimal `c10::BFloat16` / `at::BFloat16` compatibility shim so the
// Dao-AILab `causal-conv1d` CUDA kernels in `third_party/causal-conv1d`
// compile without libtorch/ATen.
//
// The upstream .cu files include `<c10/util/BFloat16.h>` and use
// `c10::BFloat16` (and its `at::` alias) as a template parameter for
// `input_t` / `weight_t`. The only operations they perform on it are:
//   - `sizeof(T) == 2` (size assertion in BlockLoad/BlockStore)
//   - `float(v)` (construct float from bf16)
//   - `T(f)` (construct bf16 from float)
//   - raw byte load/store via cub::BlockLoad/BlockStore
//
// Modern `__nv_bfloat16` (cuda_bf16.h, CUDA 11.0+) provides all of the
// above natively (implicit `operator float() const` and
// `__nv_bfloat16(float)` ctor). So a pure type alias is sufficient —
// no wrapper struct, no operator overloads needed.
//
// This header is added to the nvcc include path **before** the real
// PyTorch `c10/util/BFloat16.h` (which we don't have anyway), so the
// upstream `#include <c10/util/BFloat16.h>` resolves here.

#pragma once

#include <cuda_bf16.h>

namespace c10 {
using BFloat16 = __nv_bfloat16;
}  // namespace c10

// Upstream code also uses `at::BFloat16` because PyTorch's ATen
// aliases it to `c10::BFloat16`. Mirror that alias here.
namespace at {
using BFloat16 = __nv_bfloat16;
}  // namespace at

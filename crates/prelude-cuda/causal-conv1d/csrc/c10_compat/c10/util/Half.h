// SPDX-License-Identifier: Apache-2.0
//
// `c10::Half` / `at::Half` shim for Dao-AILab causal-conv1d. Same story as
// BFloat16.h — modern `__half` (cuda_fp16.h) has implicit conversions to
// and from float, so a type alias is enough.

#pragma once

#include <cuda_fp16.h>

namespace c10 {
using Half = __half;
}  // namespace c10

namespace at {
using Half = __half;
}  // namespace at

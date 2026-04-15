// SPDX-License-Identifier: Apache-2.0
//
// extern "C" wrapper around the Dao-AILab causal-conv1d kernels
// (`third_party/causal-conv1d/csrc/{causal_conv1d_fwd,causal_conv1d_update}.cu`).
//
// Upstream exposes its kernels via a torch::Tensor Python binding in
// `causal_conv1d.cpp`, which drags in all of libtorch. We only need the
// framework-agnostic `causal_conv1d_{fwd,update}_cuda<>` template entry
// points declared (implicitly) in `causal_conv1d.h` and instantiated for
// bf16/fp16/fp32 by the two .cu files. So we:
//
//   * include the upstream header (just the `ConvParamsBase` POD)
//   * forward-declare the two template functions
//   * provide extern "C" wrappers that fill in ConvParamsBase from raw
//     pointers and strides, then dispatch to the right instantiation by
//     dtype tag.
//
// The kernel templates themselves live in the upstream .cu files — our
// build.rs compiles those alongside this shim and links the whole thing
// into a static archive.

#include <cstdint>
#include <cuda_runtime_api.h>

// Resolves to our c10_compat shim because `-Icrates/.../csrc/c10_compat`
// comes first on the nvcc include path.
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

// Upstream, framework-agnostic header providing `ConvParamsBase`.
#include "causal_conv1d.h"

// Template entry points defined in upstream's causal_conv1d_{fwd,update}.cu.
// The instantiations for (input, weight) ∈ { float, at::Half, at::BFloat16 }²
// are emitted there; we only need the declarations here.
template <typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream);

namespace {

// Dtype tags for the C ABI. We keep the value space tight so the Rust
// side can treat this as a plain `i32`. Only the combinations we
// actually use in prelude-cuda are surfaced.
//
//   0 = bf16, 1 = f16, 2 = f32
//
// The full 3×3 (input, weight) matrix is populated so the caller can
// mix-and-match — this matches upstream's 9 template instantiations.
constexpr int DT_BF16 = 0;
constexpr int DT_F16 = 1;
constexpr int DT_F32 = 2;

template <typename In>
static void dispatch_weight(int weight_dtype, ConvParamsBase &params,
                            cudaStream_t stream, bool is_update) {
  if (weight_dtype == DT_BF16) {
    if (is_update)
      causal_conv1d_update_cuda<In, at::BFloat16>(params, stream);
    else
      causal_conv1d_fwd_cuda<In, at::BFloat16>(params, stream);
  } else if (weight_dtype == DT_F16) {
    if (is_update)
      causal_conv1d_update_cuda<In, at::Half>(params, stream);
    else
      causal_conv1d_fwd_cuda<In, at::Half>(params, stream);
  } else {
    if (is_update)
      causal_conv1d_update_cuda<In, float>(params, stream);
    else
      causal_conv1d_fwd_cuda<In, float>(params, stream);
  }
}

static void dispatch(int input_dtype, int weight_dtype,
                     ConvParamsBase &params, cudaStream_t stream,
                     bool is_update) {
  switch (input_dtype) {
    case DT_BF16:
      dispatch_weight<at::BFloat16>(weight_dtype, params, stream, is_update);
      break;
    case DT_F16:
      dispatch_weight<at::Half>(weight_dtype, params, stream, is_update);
      break;
    default:
      dispatch_weight<float>(weight_dtype, params, stream, is_update);
      break;
  }
}

}  // namespace

extern "C" {

// Strided causal conv1d forward (prefill).
//
// Shapes / strides are passed explicitly so the caller doesn't need to
// assume a particular tensor layout. The canonical layout we use from
// Rust is `x: [B, C, L]` (channel-last-before-time) matching upstream's
// convention. Weight is `[C, K]` depthwise.
//
//   x         : (batch, dim, seqlen) input, `input_dtype` element type
//   weight    : (dim, width) filter, `weight_dtype` element type
//   bias      : optional (dim,), nullptr ok
//   out       : (batch, dim, seqlen) output, same dtype as x
//   initial_states : optional (batch, dim, width - 1), left context. nullptr → zeros.
//   final_states   : optional (batch, dim, width - 1), output of last width-1 inputs.
//                    nullptr → don't write.
//   silu_activation : if non-zero, fuse a SiLU at the end.
//
// Returns 0 on success. Does NOT throw — if something is off, the
// kernel will cudaGetLastError() into the stderr channel via the
// c10_compat CHECK macros.
int cula_causal_conv1d_fwd(
    cudaStream_t stream,
    const void *x,
    const void *weight,
    const void *bias,
    const void *initial_states,
    void *final_states,
    void *out,
    int32_t batch,
    int32_t dim,
    int32_t seqlen,
    int32_t width,
    int32_t silu_activation,
    // strides in element counts (NOT bytes)
    int64_t x_batch_stride,
    int64_t x_c_stride,
    int64_t x_l_stride,
    int64_t weight_c_stride,
    int64_t weight_width_stride,
    int64_t out_batch_stride,
    int64_t out_c_stride,
    int64_t out_l_stride,
    int64_t initial_states_batch_stride,
    int64_t initial_states_c_stride,
    int64_t initial_states_l_stride,
    int64_t final_states_batch_stride,
    int64_t final_states_c_stride,
    int64_t final_states_l_stride,
    int32_t input_dtype,
    int32_t weight_dtype
) {
  ConvParamsBase params{};
  params.batch = batch;
  params.dim = dim;
  params.seqlen = seqlen;
  params.width = width;
  params.silu_activation = silu_activation != 0;

  params.x_batch_stride = static_cast<ConvParamsBase::index_t>(x_batch_stride);
  params.x_c_stride = static_cast<ConvParamsBase::index_t>(x_c_stride);
  params.x_l_stride = static_cast<ConvParamsBase::index_t>(x_l_stride);
  params.weight_c_stride =
      static_cast<ConvParamsBase::index_t>(weight_c_stride);
  params.weight_width_stride =
      static_cast<ConvParamsBase::index_t>(weight_width_stride);
  params.out_batch_stride =
      static_cast<ConvParamsBase::index_t>(out_batch_stride);
  params.out_c_stride = static_cast<ConvParamsBase::index_t>(out_c_stride);
  params.out_l_stride = static_cast<ConvParamsBase::index_t>(out_l_stride);

  params.x_ptr = const_cast<void *>(x);
  params.weight_ptr = const_cast<void *>(weight);
  params.bias_ptr = const_cast<void *>(bias);
  params.out_ptr = out;

  params.conv_state_ptr = nullptr;
  params.cache_seqlens = nullptr;
  params.conv_state_indices_ptr = nullptr;
  params.seq_idx_ptr = nullptr;

  params.initial_states_ptr = const_cast<void *>(initial_states);
  params.initial_states_batch_stride =
      static_cast<ConvParamsBase::index_t>(initial_states_batch_stride);
  params.initial_states_l_stride =
      static_cast<ConvParamsBase::index_t>(initial_states_l_stride);
  params.initial_states_c_stride =
      static_cast<ConvParamsBase::index_t>(initial_states_c_stride);

  params.final_states_ptr = final_states;
  params.final_states_batch_stride =
      static_cast<ConvParamsBase::index_t>(final_states_batch_stride);
  params.final_states_l_stride =
      static_cast<ConvParamsBase::index_t>(final_states_l_stride);
  params.final_states_c_stride =
      static_cast<ConvParamsBase::index_t>(final_states_c_stride);

  dispatch(input_dtype, weight_dtype, params, stream, /*is_update=*/false);
  return 0;
}

// Single-token causal conv1d update (decode).
//
// Reads conv_state (shape `(batch, dim, conv_state_len)`) plus a new
// input `(batch, dim, seqlen=1)`, shifts conv_state left by one, writes
// the new input into the last slot, computes the conv1d output at the
// new position, writes to out `(batch, dim, 1)`. conv_state is updated
// in place.
//
//   conv_state_len: the width-1 + padding dimension of conv_state.
//                   For a kernel size K the tail `width-1` positions are
//                   read; the rest (typically 0) is ignored.
int cula_causal_conv1d_update(
    cudaStream_t stream,
    const void *x,
    void *conv_state,
    const void *weight,
    const void *bias,
    void *out,
    const int32_t *conv_state_indices,
    int32_t batch,
    int32_t dim,
    int32_t seqlen,
    int32_t width,
    int32_t conv_state_len,
    int32_t silu_activation,
    int64_t x_batch_stride,
    int64_t x_c_stride,
    int64_t x_l_stride,
    int64_t weight_c_stride,
    int64_t weight_width_stride,
    int64_t out_batch_stride,
    int64_t out_c_stride,
    int64_t out_l_stride,
    int64_t conv_state_batch_stride,
    int64_t conv_state_c_stride,
    int64_t conv_state_l_stride,
    int32_t input_dtype,
    int32_t weight_dtype
) {
  ConvParamsBase params{};
  params.batch = batch;
  params.dim = dim;
  params.seqlen = seqlen;
  params.width = width;
  params.silu_activation = silu_activation != 0;
  params.conv_state_len = conv_state_len;

  params.x_batch_stride = static_cast<ConvParamsBase::index_t>(x_batch_stride);
  params.x_c_stride = static_cast<ConvParamsBase::index_t>(x_c_stride);
  params.x_l_stride = static_cast<ConvParamsBase::index_t>(x_l_stride);
  params.weight_c_stride =
      static_cast<ConvParamsBase::index_t>(weight_c_stride);
  params.weight_width_stride =
      static_cast<ConvParamsBase::index_t>(weight_width_stride);
  params.out_batch_stride =
      static_cast<ConvParamsBase::index_t>(out_batch_stride);
  params.out_c_stride = static_cast<ConvParamsBase::index_t>(out_c_stride);
  params.out_l_stride = static_cast<ConvParamsBase::index_t>(out_l_stride);
  params.conv_state_batch_stride =
      static_cast<ConvParamsBase::index_t>(conv_state_batch_stride);
  params.conv_state_c_stride =
      static_cast<ConvParamsBase::index_t>(conv_state_c_stride);
  params.conv_state_l_stride =
      static_cast<ConvParamsBase::index_t>(conv_state_l_stride);

  params.x_ptr = const_cast<void *>(x);
  params.weight_ptr = const_cast<void *>(weight);
  params.bias_ptr = const_cast<void *>(bias);
  params.out_ptr = out;
  params.conv_state_ptr = conv_state;
  params.cache_seqlens = nullptr;
  params.conv_state_indices_ptr = const_cast<int32_t *>(conv_state_indices);
  params.seq_idx_ptr = nullptr;

  params.initial_states_ptr = nullptr;
  params.final_states_ptr = nullptr;

  dispatch(input_dtype, weight_dtype, params, stream, /*is_update=*/true);
  return 0;
}

}  // extern "C"

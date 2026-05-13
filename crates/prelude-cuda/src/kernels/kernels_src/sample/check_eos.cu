// Device-side EOS / stop-token check.
//
// For each row in a `[B] u32` tensor of sampled tokens, mark whether it
// matches any token id in a small per-batch EOS table (`[E] u32`). The
// output is a `[B] u8` bitmap — 1 = sequence finished, 0 = continue.
//
// Why we need this on the device: prelude currently calls
// `engine.is_eos(next_token)` after every decode step, which forces a
// per-step D→H copy of the sampled token (\~7 ms of host stall in our
// nsys traces). Producing the EOS decision on the device lets callers
// fold it into the same async stream as the sampler and sync the
// bitmap only every K steps (PR-4).
//
// Launch geometry: 1-D grid, 256 threads per block.
//   grid  = (ceil_div(B, 256), 1, 1)
//   block = (256, 1, 1)
//
// E (number of EOS ids) is expected to be small — typically 1–4 for
// most models (Qwen3 has two: `<|im_end|>`, `<|endoftext|>`). The
// inner loop is a tight scan; for E up to ~32 there is no benefit to
// staging into shared memory.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void check_eos_u32(
    const uint32_t* __restrict__ tokens,
    const uint32_t* __restrict__ eos_ids,
    uint8_t*        __restrict__ done,
    uint32_t B,
    uint32_t E
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const uint32_t tok = tokens[b];
    uint8_t hit = 0;
    #pragma unroll 4
    for (uint32_t i = 0; i < E; i++) {
        if (tok == eos_ids[i]) {
            hit = 1;
            break;
        }
    }
    done[b] = hit;
}

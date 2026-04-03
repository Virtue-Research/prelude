// Vectorized load/store utilities for BF16 kernels
// Each thread processes 8 BF16 elements via 128-bit (float4) loads
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

// ─── Vectorized BF16 load (8 elements) ──────────────────────────────────

struct Vec8BF16 {
    float4 data;

    __device__ __forceinline__ void load(const __nv_bfloat16* ptr) {
        data = *reinterpret_cast<const float4*>(ptr);
    }

    __device__ __forceinline__ void store(__nv_bfloat16* ptr) const {
        *reinterpret_cast<float4*>(ptr) = data;
    }

    __device__ __forceinline__ const __nv_bfloat162* as_bf162() const {
        return reinterpret_cast<const __nv_bfloat162*>(&data);
    }

    __device__ __forceinline__ __nv_bfloat162* as_bf162_mut() {
        return reinterpret_cast<__nv_bfloat162*>(&data);
    }
};

// ─── Process 8 BF16 elements with a unary function ──────────────────────

template<typename Func>
__device__ __forceinline__ void vec8_unary(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    uint32_t idx,
    uint32_t n,
    Func fn
) {
    if (idx + 8 <= n) {
        Vec8BF16 v;
        v.load(in + idx);
        const __nv_bfloat162* inp = v.as_bf162();

        Vec8BF16 result;
        __nv_bfloat162* outp = result.as_bf162_mut();

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 f = __bfloat1622float2(inp[i]);
            float2 r;
            r.x = fn(f.x);
            r.y = fn(f.y);
            outp[i] = __float22bfloat162_rn(r);
        }

        result.store(out + idx);
    } else if (idx < n) {
        for (uint32_t i = idx; i < n; i++) {
            float v = __bfloat162float(in[i]);
            out[i] = __float2bfloat16(fn(v));
        }
    }
}

// ─── Process 8 BF16 elements with a binary function ─────────────────────

template<typename Func>
__device__ __forceinline__ void vec8_binary(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    uint32_t idx,
    uint32_t n,
    Func fn
) {
    if (idx + 8 <= n) {
        Vec8BF16 va, vb;
        va.load(a + idx);
        vb.load(b + idx);
        const __nv_bfloat162* ap = va.as_bf162();
        const __nv_bfloat162* bp = vb.as_bf162();

        Vec8BF16 result;
        __nv_bfloat162* outp = result.as_bf162_mut();

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 af = __bfloat1622float2(ap[i]);
            float2 bf = __bfloat1622float2(bp[i]);
            float2 r;
            r.x = fn(af.x, bf.x);
            r.y = fn(af.y, bf.y);
            outp[i] = __float22bfloat162_rn(r);
        }

        result.store(out + idx);
    } else if (idx < n) {
        for (uint32_t i = idx; i < n; i++) {
            float af = __bfloat162float(a[i]);
            float bf = __bfloat162float(b[i]);
            out[i] = __float2bfloat16(fn(af, bf));
        }
    }
}

// ─── Vectorized dot product accumulator ─────────────────────────────────

__device__ __forceinline__ float vec8_dot_accum(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    uint32_t k,
    uint32_t stride
) {
    float sum = 0.0f;

    for (uint32_t i = k; i < stride; i += 256) {  // 32 lanes * 8 elements
        Vec8BF16 va, vb;
        va.load(a + i);
        vb.load(b + i);
        const __nv_bfloat162* ap = va.as_bf162();
        const __nv_bfloat162* bp = vb.as_bf162();

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(ap[j]);
            float2 bf = __bfloat1622float2(bp[j]);
            sum += af.x * bf.x + af.y * bf.y;
        }
    }

    return sum;
}

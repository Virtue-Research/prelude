// FFI entry points for llama.cpp MMQ + MMVQ kernels.
// Bridges Rust → C → C++ template dispatch.
// Headers auto-cloned from llama.cpp main branch at build time.

#include "mmq_ffi.h"
#include "common.cuh"
#include "mmq.cuh"
#include "vecdotq.cuh"

// Include quantize implementation directly (single translation unit).
#include "quantize.cu"

// Provide symbols declared in ggml.h / common.cuh but defined in ggml.c / ggml-cuda.cu normally.

#include <cstdarg>

extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "GGML ABORT at %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

void ggml_cuda_error(
    const char * stmt, const char * func, const char * file,
    int line, const char * msg
) {
    fprintf(stderr, "CUDA error: %s at %s:%d in %s: %s\n", stmt, file, line, func, msg);
    abort();
}

// Simple CUDA memory pool using cudaMalloc/cudaFree.
struct ggml_cuda_pool_simple : ggml_cuda_pool {
    void * alloc(size_t size, size_t * actual_size) override {
        void * ptr = nullptr;
        cudaMalloc(&ptr, size);
        *actual_size = size;
        return ptr;
    }
    void free(void * ptr, size_t /*size*/) override {
        cudaFree(ptr);
    }
};

std::unique_ptr<ggml_cuda_pool>
ggml_backend_cuda_context::new_pool_for_device(int /*device*/, int /*stream_no*/) {
    return std::make_unique<ggml_cuda_pool_simple>();
}

// Highest compiled architecture (used for runtime feature detection).
int ggml_cuda_highest_compiled_arch_impl() {
    // Return the arch we were compiled for (set by -arch=sm_XX in build.rs).
    // The __CUDA_ARCH__ macro is only defined in device code, so we use a
    // device kernel to query it, or just return a safe default.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.major * 100 + prop.minor * 10;
}

// Context destructor (pools call cudaFree).
ggml_backend_cuda_context::~ggml_backend_cuda_context() = default;

// Device management stubs.
int ggml_cuda_get_device() { return 0; }
void ggml_cuda_set_device(int) {}

// Device info singleton (lazy-initialized in llama_mmq_mul_mat).
const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = {};
    return info;
}

// Explicit template instantiation for mul_mat_q_case (mmq.cuh declares extern).
template void mul_mat_q_case<GGML_TYPE_Q4_0>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q4_1>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_0>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_1>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q8_0>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_MXFP4>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q2_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q3_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q6_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ2_XS>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ2_S>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ3_S>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ1_S>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ4_NL>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_IQ4_XS>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_NVFP4>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);

#include <cuda_bf16.h>

// ── BF16 → F32 conversion kernel ───────────────────────────────────────

static __global__ void bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ in,
    float* __restrict__ out,
    int64_t n
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

// ── Q8_1 Quantization ──────────────────────────────────────────────────

extern "C" void llama_mmq_quantize_q8_1(
    const void* x_bf16, void* x_q8,
    int64_t M, int64_t K,
    int ggml_type_id,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    const int64_t n = M * K;

    // BF16 → F32 (llama.cpp quantizer expects F32)
    float* x_f32 = nullptr;
    cudaMallocAsync(&x_f32, n * sizeof(float), stream);

    {
        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);
        bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)x_bf16, x_f32, n);
    }

    quantize_mmq_q8_1_cuda(
        x_f32,
        nullptr,             // ids (nullptr for non-MoE)
        x_q8,
        (ggml_type)ggml_type_id,
        K,                   // ne00 = K
        K,                   // s01 = stride between rows
        M * K,               // s02
        M * K,               // s03
        K,                   // ne0
        M,                   // ne1
        1, 1,                // ne2, ne3
        stream
    );

    cudaFreeAsync(x_f32, stream);
}

// ── MMQ Matrix Multiply ─────────────────────────────────────────────────

template <ggml_type type>
static void launch_mmq_typed(
    const void* W, const void* x_q8, float* y,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream
) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;

    mmq_args args = {};
    args.x = (const char*)W;
    args.type_x = type;
    args.y = (const int*)x_q8;
    args.ids_dst = nullptr;
    args.expert_bounds = nullptr;
    args.dst = y;

    args.ncols_x = K;
    args.nrows_x = N;
    args.ncols_y = M;
    args.nrows_dst = N;
    args.ncols_dst = M;
    args.stride_row_x = K / qk;

    args.nchannels_x = 1;
    args.nchannels_y = 1;
    args.stride_channel_x = N * (K / qk);
    args.stride_channel_y = 0;
    args.stride_channel_dst = 0;

    args.nsamples_x = 1;
    args.nsamples_y = 1;
    args.stride_sample_x = 0;
    args.stride_sample_y = 0;
    args.stride_sample_dst = 0;

    args.ncols_max = M;

    // Match upstream: use stream-k for NVIDIA Volta+ (like llama.cpp does).
    const auto& dev = ggml_cuda_info().devices[0];
    args.use_stream_k = GGML_CUDA_CC_IS_NVIDIA(dev.cc)
        && ggml_cuda_highest_compiled_arch(dev.cc) >= GGML_CUDA_CC_VOLTA;

    ggml_backend_cuda_context ctx(0);
    ctx.streams[0][0] = stream;
    ctx.curr_stream_no = 0;

    mul_mat_q_case<type>(ctx, args, stream);
}

extern "C" void llama_mmq_mul_mat(
    const void* W, const void* x_q8, float* y,
    int64_t M, int64_t N, int64_t K,
    int ggml_type_id,
    int compute_cap,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Lazy-init device info for llama.cpp's runtime decisions
    ggml_cuda_device_info& info = const_cast<ggml_cuda_device_info&>(ggml_cuda_info());
    if (info.devices[0].cc == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        info.device_count = 1;
        auto& dev = info.devices[0];
        dev.cc = (compute_cap > 0) ? compute_cap : prop.major * 100 + prop.minor * 10;
        dev.nsm = prop.multiProcessorCount;
        dev.smpb = prop.sharedMemPerBlock;
        dev.smpbo = prop.sharedMemPerBlockOptin;
        dev.warp_size = prop.warpSize;
        dev.total_vram = prop.totalGlobalMem;
        dev.integrated = prop.integrated;
        dev.supports_cooperative_launch = (prop.cooperativeLaunch != 0);
    }

    switch (ggml_type_id) {
        case GGML_TYPE_Q4_0:    launch_mmq_typed<GGML_TYPE_Q4_0>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q4_1:    launch_mmq_typed<GGML_TYPE_Q4_1>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q5_0:    launch_mmq_typed<GGML_TYPE_Q5_0>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q5_1:    launch_mmq_typed<GGML_TYPE_Q5_1>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q8_0:    launch_mmq_typed<GGML_TYPE_Q8_0>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_MXFP4:   launch_mmq_typed<GGML_TYPE_MXFP4>  (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q2_K:    launch_mmq_typed<GGML_TYPE_Q2_K>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q3_K:    launch_mmq_typed<GGML_TYPE_Q3_K>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q4_K:    launch_mmq_typed<GGML_TYPE_Q4_K>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q5_K:    launch_mmq_typed<GGML_TYPE_Q5_K>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q6_K:    launch_mmq_typed<GGML_TYPE_Q6_K>   (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_IQ2_XXS: launch_mmq_typed<GGML_TYPE_IQ2_XXS>(W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_IQ2_XS:  launch_mmq_typed<GGML_TYPE_IQ2_XS> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_IQ2_S:   launch_mmq_typed<GGML_TYPE_IQ2_S>  (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_IQ3_XXS: launch_mmq_typed<GGML_TYPE_IQ3_XXS>(W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_IQ3_S:   launch_mmq_typed<GGML_TYPE_IQ3_S>  (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_IQ1_S:   launch_mmq_typed<GGML_TYPE_IQ1_S>  (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_IQ4_NL:  launch_mmq_typed<GGML_TYPE_IQ4_NL> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_IQ4_XS:  launch_mmq_typed<GGML_TYPE_IQ4_XS> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_NVFP4:   launch_mmq_typed<GGML_TYPE_NVFP4>  (W, x_q8, y, M, N, K, stream); break;
        default: break;
    }
}

// ── MMVQ: fused matrix-vector multiply with quantized weights ───────────
//
// vec_dot function pointer type + dispatch (from llama.cpp mmvq.cu)
typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ1_S:   return VDR_IQ1_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ1_M:   return VDR_IQ1_M_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        default:                return 1;
    }
}
//
// Uses llama.cpp's vec_dot functions from vecdotq.cuh directly.
// Kernel uses the same thread-to-work mapping as llama.cpp's mul_mat_vec_q.
// Input: quantized weights W[N,K] + Q8_1-quantized activations x[K]
// Output: float y[N]

#define MMVQ_NWARPS 4

template <ggml_type type>
static __global__ void llama_mmvq_kernel(
    const void* __restrict__ W,
    const void* __restrict__ x_q8,
    float* __restrict__ y,
    int64_t N,
    int64_t K
) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr vec_dot_q_cuda_t vec_dot = get_vec_dot_q_cuda(type);

    const int row = blockIdx.x;
    if (row >= (int)N) return;

    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int blocks_per_row = K / qk;
    const int stride_row = blocks_per_row;  // contiguous rows
    constexpr int blocks_per_iter = vdr * MMVQ_NWARPS * 32 / qi;

    const block_q8_1* q8 = (const block_q8_1*)x_q8;

    float sum = 0.0f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
        sum += vec_dot(W, q8 + kby, row * stride_row + kbx, kqs);
    }

    // Warp reduce
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    // Cross-warp reduce via shared memory
    __shared__ float sdata[MMVQ_NWARPS];
    if (threadIdx.x == 0) sdata[threadIdx.y] = sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < MMVQ_NWARPS; w++) total += sdata[w];
        y[row] = total;
    }
}

template <ggml_type type>
static void launch_mmvq_typed(
    const void* W, const void* x_q8, float* y,
    int64_t N, int64_t K,
    cudaStream_t stream
) {
    dim3 block(32, MMVQ_NWARPS);
    dim3 grid((int)N);
    llama_mmvq_kernel<type><<<grid, block, 0, stream>>>(W, x_q8, y, N, K);
}

extern "C" void llama_mmvq_mul_mat_vec(
    const void* W, const void* x_q8, float* y,
    int64_t N, int64_t K,
    int ggml_type_id,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    switch (ggml_type_id) {
        case GGML_TYPE_Q4_0:    launch_mmvq_typed<GGML_TYPE_Q4_0>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_Q4_1:    launch_mmvq_typed<GGML_TYPE_Q4_1>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_Q5_0:    launch_mmvq_typed<GGML_TYPE_Q5_0>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_Q5_1:    launch_mmvq_typed<GGML_TYPE_Q5_1>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_Q8_0:    launch_mmvq_typed<GGML_TYPE_Q8_0>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_Q2_K:    launch_mmvq_typed<GGML_TYPE_Q2_K>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_Q3_K:    launch_mmvq_typed<GGML_TYPE_Q3_K>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_Q4_K:    launch_mmvq_typed<GGML_TYPE_Q4_K>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_Q5_K:    launch_mmvq_typed<GGML_TYPE_Q5_K>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_Q6_K:    launch_mmvq_typed<GGML_TYPE_Q6_K>   (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_IQ2_XXS: launch_mmvq_typed<GGML_TYPE_IQ2_XXS>(W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_IQ2_XS:  launch_mmvq_typed<GGML_TYPE_IQ2_XS> (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_IQ2_S:   launch_mmvq_typed<GGML_TYPE_IQ2_S>  (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_IQ3_XXS: launch_mmvq_typed<GGML_TYPE_IQ3_XXS>(W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_IQ3_S:   launch_mmvq_typed<GGML_TYPE_IQ3_S>  (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_IQ4_NL:  launch_mmvq_typed<GGML_TYPE_IQ4_NL> (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_IQ4_XS:  launch_mmvq_typed<GGML_TYPE_IQ4_XS> (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_IQ1_S:   launch_mmvq_typed<GGML_TYPE_IQ1_S>  (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_IQ1_M:   launch_mmvq_typed<GGML_TYPE_IQ1_M>  (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_MXFP4:   launch_mmvq_typed<GGML_TYPE_MXFP4>  (W, x_q8, y, N, K, stream); break;
        case GGML_TYPE_NVFP4:   launch_mmvq_typed<GGML_TYPE_NVFP4>  (W, x_q8, y, N, K, stream); break;
        default: break;
    }
}

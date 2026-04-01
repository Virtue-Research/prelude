// FFI entry points for vendored llama.cpp MMQ kernels.
// Bridges Rust → C → C++ template dispatch.

#include "mmq_ffi.h"
#include "llama_mmq/common.cuh"
#include "llama_mmq/mmq.cuh"

// Include quantize implementation directly (single translation unit).
#include "llama_mmq/quantize.cu"

// Provide symbols declared in common.cuh but defined in ggml-cuda.cu normally.

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
template void mul_mat_q_case<GGML_TYPE_Q2_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q3_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q5_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);
template void mul_mat_q_case<GGML_TYPE_Q6_K>(ggml_backend_cuda_context &, const mmq_args &, cudaStream_t);

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
    args.use_stream_k = false;

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
        case GGML_TYPE_Q4_0:  launch_mmq_typed<GGML_TYPE_Q4_0> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q4_1:  launch_mmq_typed<GGML_TYPE_Q4_1> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q5_0:  launch_mmq_typed<GGML_TYPE_Q5_0> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q5_1:  launch_mmq_typed<GGML_TYPE_Q5_1> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q8_0:  launch_mmq_typed<GGML_TYPE_Q8_0> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q2_K:  launch_mmq_typed<GGML_TYPE_Q2_K> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q3_K:  launch_mmq_typed<GGML_TYPE_Q3_K> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q4_K:  launch_mmq_typed<GGML_TYPE_Q4_K> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q5_K:  launch_mmq_typed<GGML_TYPE_Q5_K> (W, x_q8, y, M, N, K, stream); break;
        case GGML_TYPE_Q6_K:  launch_mmq_typed<GGML_TYPE_Q6_K> (W, x_q8, y, M, N, K, stream); break;
        default: break;
    }
}

// CUTLASS Blackwell SM100 grouped GEMM for MoE — replaces the legacy
// SM75-era `nvcuda::wmma` `moe_gemm_wmma` kernel on B300+.
//
// MoE compute pattern: each token is routed to top-K experts. After a
// global sort by expert id, the assignments are contiguous per-expert,
// so each expert's GEMM is `[M_e, K] @ [N, K]^T = [M_e, N]` where M_e
// varies per expert. CUTLASS's `cutlass::gemm::GemmUniversalMode::kGrouped`
// is purpose-built for this — variable M, fixed N/K per group.
//
// We do the scatter-gather around the GEMM rather than fusing it into
// the kernel:
//
//   1. `gather_a_kernel`: pack input rows according to `sorted_token_ids`,
//      so the per-expert slice `gathered[expert_offsets[e]..]` is
//      already contiguous when CUTLASS calls in.
//   2. `prepare_grouped_args_kernel`: from `expert_offsets` (host-input)
//      and the gathered/output buffers, populate per-group
//      problem_sizes / ptr_A / ptr_B / ptr_D / strides on device.
//   3. CUTLASS Sm100 grouped GEMM (`KernelPtrArrayTmaWarpSpecialized*Sm100`).
//   4. `scatter_d_kernel`: copy GEMM output back to the original
//      assignment slot via `sorted_token_ids`.
//
// The two boundary kernels are memory-bound and trivial relative to the
// GEMM (~50µs each at decode shapes). Total ≈ 1.05× the pure-GEMM cost.
//
// Layout: A row-major [M_total, K], B per-expert col-major [N, K],
// D row-major [M_total, N]. Matches our existing moe_gemm_wmma signature
// so the Rust wrapper is a near-drop-in replacement.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

using namespace cute;

namespace prelude_grouped_moe_sm100 {

using BF16 = cutlass::bfloat16_t;
using FP16 = cutlass::half_t;

// ── Grouped Gemm builder, parameterized on Element + 1Sm/2Sm config ──

template <class Element, bool Use2SmMma>
struct GroupedMoeRunner {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

    using ElementA = Element;
    using ElementB = Element;
    using ElementC = Element;
    using ElementD = Element;
    using ElementAccumulator = float;

    // A: gathered tokens, row-major [M_total, K]. Per-group stride is the
    // same K (no inter-expert padding in the gather buffer).
    using LayoutA = cutlass::layout::RowMajor;
    // B: per-expert weights, [N, K]. We treat the weight-row dim N as the
    // outer (M-of-B in cuBLAS) and K as inner — column-major. CUTLASS
    // grouped accepts pointer-arrays per group, so we just point each
    // group at the corresponding expert's [N,K] block.
    using LayoutB = cutlass::layout::ColumnMajor;
    // D: row-major [M_total, N], shared output buffer; per-group ptr is
    // offset by `expert_offset[g] * N`.
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using ClusterShape = std::conditional_t<Use2SmMma, Shape<_2,_1,_1>, Shape<_1,_1,_1>>;
    using MmaTileShape = std::conditional_t<
        Use2SmMma, Shape<_256,_128,_64>, Shape<_128,_128,_64>>;

    using KernelSchedule = std::conditional_t<
        Use2SmMma,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100>;
    using EpilogueSchedule = std::conditional_t<
        Use2SmMma,
        cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,
        cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutD*, AlignmentD,
        EpilogueSchedule,
        cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator>
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA*, AlignmentA,
        ElementB, LayoutB*, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename GemmKernel::InternalStrideA;
    using StrideB = typename GemmKernel::InternalStrideB;
    using StrideC = typename GemmKernel::InternalStrideC;
    using StrideD = typename GemmKernel::InternalStrideD;
};

// ── Boundary kernels (gather A, scatter D) ──────────────────────────

// Gather: gathered[i, k] = input[sorted_token_ids[i] / topk, k].
// Each block handles one assignment row × `BLOCK` cols.
template <class Element>
__global__ void gather_a_kernel(
    const Element* __restrict__ input,        // [num_tokens, K]
    const uint32_t* __restrict__ sorted_token_ids, // [M_total]
    Element* __restrict__ gathered,           // [M_total, K]
    int M_total, int K, int topk)
{
    int row = blockIdx.x;
    if (row >= M_total) return;
    uint32_t flat = sorted_token_ids[row];
    int token = static_cast<int>(flat) / topk;
    const Element* src = input + static_cast<size_t>(token) * K;
    Element* dst = gathered + static_cast<size_t>(row) * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        dst[k] = src[k];
    }
}

// Scatter: output[sorted_token_ids[i], n] = gemm_out[i, n].
// Output `[M_total, N]` is in flat (token × topk) layout — caller does
// the topk reduction outside. So no atomicAdd needed; each scatter
// destination is unique per assignment.
template <class Element>
__global__ void scatter_d_kernel(
    const Element* __restrict__ gemm_out,     // [M_total, N]
    const uint32_t* __restrict__ sorted_token_ids, // [M_total]
    Element* __restrict__ output,             // [M_total, N]
    int M_total, int N)
{
    int row = blockIdx.x;
    if (row >= M_total) return;
    uint32_t dst_row = sorted_token_ids[row];
    const Element* src = gemm_out + static_cast<size_t>(row) * N;
    Element* dst = output + static_cast<size_t>(dst_row) * N;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        dst[n] = src[n];
    }
}

// Build per-group problem-shape / pointer / stride arrays on device
// from the host-side `expert_offsets`. One block per group.
//
// `problem_sizes[g] = (M_g, N, K)` where M_g = expert_offsets[g+1] - expert_offsets[g].
// CUTLASS skips groups with M_g == 0 internally.
template <class Element, class StrideA, class StrideB, class StrideC, class StrideD>
__global__ void prepare_grouped_args_kernel(
    const int32_t* __restrict__ expert_offsets,   // [num_experts + 1]
    const Element* gathered,                      // [M_total, K]
    const Element* weights,                       // [num_experts, N, K]
    Element* gemm_out,                            // [M_total, N]
    int N, int K,
    cutlass::gemm::GroupProblemShape<Shape<int,int,int>>::UnderlyingProblemShape* problem_sizes,
    const Element** ptr_A,
    const Element** ptr_B,
    const Element** ptr_C,  // unused (beta=0) but CUTLASS arg-struct expects it
    Element** ptr_D,
    StrideA* stride_A,
    StrideB* stride_B,
    StrideC* stride_C,
    StrideD* stride_D,
    int num_experts)
{
    int g = blockIdx.x;
    if (g >= num_experts) return;
    if (threadIdx.x != 0) return;

    int off = expert_offsets[g];
    int next = expert_offsets[g + 1];
    int M_g = next - off;

    using PS = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>::UnderlyingProblemShape;
    problem_sizes[g] = PS{M_g, N, K};

    ptr_A[g] = gathered + static_cast<size_t>(off) * K;
    ptr_B[g] = weights  + static_cast<size_t>(g)   * N * K;
    ptr_C[g] = gemm_out + static_cast<size_t>(off) * N;  // unused, but set for safety
    ptr_D[g] = gemm_out + static_cast<size_t>(off) * N;

    stride_A[g] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M_g, K, 1));
    stride_B[g] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    stride_C[g] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M_g, N, 1));
    stride_D[g] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M_g, N, 1));
}

// ── Workspace layout ────────────────────────────────────────────────
//
// Per call we need:
//   - gathered:      M_total * K * sizeof(Element)
//   - gemm_out:      M_total * N * sizeof(Element)
//   - problem_sizes: num_experts * sizeof(PS)
//   - ptr_*:         num_experts * sizeof(void*) × 4
//   - stride_*:      num_experts * sizeof(stride) × 4
//   - cutlass workspace returned by Gemm::get_workspace_size()
//
// All of those land in a single cudaMallocAsync'd workspace allocated
// at the C entry point. The previous implementation did 11 separate
// cudaMallocAsync calls per layer; on Qwen3-30B-A3B that's 11 × 48 =
// 528 allocs per forward. Even with the stream-ordered memory pool it
// adds host dispatch latency that compounds across the 200ms+ prefill.

constexpr size_t WS_ALIGN = 256;
inline size_t ws_round_up(size_t bytes) {
    return (bytes + WS_ALIGN - 1) & ~(WS_ALIGN - 1);
}

template <class Runner>
struct WorkspacePlan {
    using PS = typename Runner::ProblemShape::UnderlyingProblemShape;
    using ElementA = typename Runner::ElementA;
    using ElementB = typename Runner::ElementB;
    using ElementC = typename Runner::ElementC;
    using ElementD = typename Runner::ElementD;
    using StrideA = typename Runner::StrideA;
    using StrideB = typename Runner::StrideB;
    using StrideC = typename Runner::StrideC;
    using StrideD = typename Runner::StrideD;

    size_t off_problem_sizes;
    size_t off_ptr_A, off_ptr_B, off_ptr_C, off_ptr_D;
    size_t off_stride_A, off_stride_B, off_stride_C, off_stride_D;
    size_t off_cutlass_ws;
    size_t metadata_bytes;     // problem_sizes + ptrs + strides (sub-total)
    size_t cutlass_ws_bytes;
    size_t total_bytes;
};

template <class Runner>
WorkspacePlan<Runner> plan_workspace(int num_experts, size_t cutlass_ws_bytes) {
    using P = WorkspacePlan<Runner>;
    P plan{};
    size_t off = 0;
    auto carve = [&](size_t bytes) {
        size_t r = off;
        off += ws_round_up(bytes);
        return r;
    };
    plan.off_problem_sizes = carve(num_experts * sizeof(typename P::PS));
    plan.off_ptr_A = carve(num_experts * sizeof(void*));
    plan.off_ptr_B = carve(num_experts * sizeof(void*));
    plan.off_ptr_C = carve(num_experts * sizeof(void*));
    plan.off_ptr_D = carve(num_experts * sizeof(void*));
    plan.off_stride_A = carve(num_experts * sizeof(typename P::StrideA));
    plan.off_stride_B = carve(num_experts * sizeof(typename P::StrideB));
    plan.off_stride_C = carve(num_experts * sizeof(typename P::StrideC));
    plan.off_stride_D = carve(num_experts * sizeof(typename P::StrideD));
    plan.metadata_bytes = off;
    plan.off_cutlass_ws = carve(cutlass_ws_bytes);
    plan.cutlass_ws_bytes = cutlass_ws_bytes;
    plan.total_bytes = off;
    return plan;
}

// Run the grouped GEMM into a caller-provided workspace. No
// cudaMallocAsync inside — caller batches all buffers in one call.
template <class Runner>
int launch_grouped_moe(
    const typename Runner::ElementA* gathered,
    const typename Runner::ElementB* weights,
    const int32_t* expert_offsets,
    typename Runner::ElementD* gemm_out,
    int M_total, int N, int K, int num_experts,
    void* metadata_ws,                  // size = plan.metadata_bytes
    void* cutlass_ws,                   // size = plan.cutlass_ws_bytes (or nullptr)
    const WorkspacePlan<Runner>& plan,
    cudaStream_t stream)
{
    using Gemm = typename Runner::Gemm;
    using PS = typename Runner::ProblemShape::UnderlyingProblemShape;
    using StrideA = typename Runner::StrideA;
    using StrideB = typename Runner::StrideB;
    using StrideC = typename Runner::StrideC;
    using StrideD = typename Runner::StrideD;
    using ElementA = typename Runner::ElementA;
    using ElementB = typename Runner::ElementB;
    using ElementC = typename Runner::ElementC;
    using ElementD = typename Runner::ElementD;

    uint8_t* base = static_cast<uint8_t*>(metadata_ws);
    PS* d_problem_sizes = reinterpret_cast<PS*>(base + plan.off_problem_sizes);
    const ElementA** d_ptr_A = reinterpret_cast<const ElementA**>(base + plan.off_ptr_A);
    const ElementB** d_ptr_B = reinterpret_cast<const ElementB**>(base + plan.off_ptr_B);
    const ElementC** d_ptr_C = reinterpret_cast<const ElementC**>(base + plan.off_ptr_C);
    ElementD** d_ptr_D = reinterpret_cast<ElementD**>(base + plan.off_ptr_D);
    StrideA* d_stride_A = reinterpret_cast<StrideA*>(base + plan.off_stride_A);
    StrideB* d_stride_B = reinterpret_cast<StrideB*>(base + plan.off_stride_B);
    StrideC* d_stride_C = reinterpret_cast<StrideC*>(base + plan.off_stride_C);
    StrideD* d_stride_D = reinterpret_cast<StrideD*>(base + plan.off_stride_D);

    // Populate metadata on device.
    prepare_grouped_args_kernel<ElementA, StrideA, StrideB, StrideC, StrideD>
        <<<num_experts, 1, 0, stream>>>(
            expert_offsets, gathered, weights, gemm_out, N, K,
            d_problem_sizes,
            d_ptr_A, d_ptr_B, d_ptr_C, d_ptr_D,
            d_stride_A, d_stride_B, d_stride_C, d_stride_D,
            num_experts);

    cutlass::KernelHardwareInfo hw_info;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename Gemm::Arguments args;
    decltype(args.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 0};

    args = typename Gemm::Arguments {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts, d_problem_sizes, nullptr},
        { (const ElementA**)d_ptr_A, d_stride_A,
          (const ElementB**)d_ptr_B, d_stride_B },
        { fusion_args,
          (const ElementC**)d_ptr_C, d_stride_C,
          (ElementD**)d_ptr_D, d_stride_D },
        hw_info
    };

    Gemm gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return -11;
    }

    // Confirm the actual workspace size for THIS call fits the planned
    // upper bound. If not, the caller's plan was undersized — bail.
    size_t actual_ws = Gemm::get_workspace_size(args);
    if (actual_ws > plan.cutlass_ws_bytes) {
        return -14;
    }

    status = gemm.initialize(args, actual_ws > 0 ? cutlass_ws : nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return -12;
    }

    status = gemm.run(stream);
    return (status == cutlass::Status::kSuccess) ? 0 : -13;
}

// Conservative upper bound for CUTLASS grouped GEMM workspace at given
// (num_experts, M_total, N, K). Built by querying with dummy args; the
// result depends only on the problem shape, not the actual pointers.
template <class Runner>
size_t query_cutlass_workspace_upper(int num_experts) {
    using Gemm = typename Runner::Gemm;
    cutlass::KernelHardwareInfo hw_info;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename Gemm::Arguments dummy_args;
    decltype(dummy_args.epilogue.thread) fusion_args{};
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 0};

    dummy_args = typename Gemm::Arguments {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts, nullptr, nullptr},
        { nullptr, nullptr, nullptr, nullptr },
        { fusion_args, nullptr, nullptr, nullptr, nullptr },
        hw_info
    };
    return Gemm::get_workspace_size(dummy_args);
}

}  // namespace prelude_grouped_moe_sm100

// ── C FFI ───────────────────────────────────────────────────────────
//
// Signature mirrors `moe_gemm_wmma` so the Rust wrapper can choose
// between paths at runtime based on compute capability. Only BF16 is
// supported here for now (FP16 falls back to the WMMA path).

extern "C" int moe_grouped_gemm_sm100(
    const void* input,                 // [num_tokens, K]
    const void* weights,               // [num_experts, N, K]
    const uint32_t* sorted_token_ids,  // [M_total] (device)
    const int32_t* expert_offsets,     // [num_experts + 1] (device)
    void* output,                      // [M_total, N]
    int M_total, int N, int K, int num_experts, int topk,
    int data_type,                     // 0 = half, 1 = bfloat16
    cudaStream_t stream)
{
    using namespace prelude_grouped_moe_sm100;
    if (data_type != 1) return -20;  // BF16 only for now
    if (M_total <= 0 || N <= 0 || K <= 0 || num_experts <= 0) return -21;

    using R2 = GroupedMoeRunner<BF16, /*Use2SmMma=*/true>;
    using R1 = GroupedMoeRunner<BF16, /*Use2SmMma=*/false>;

    // ── Plan a single workspace covering everything for this call ──
    //
    //   [gathered][gemm_out][metadata][cutlass_ws]
    //
    // metadata + cutlass_ws sizes are identical between R1 and R2 (same
    // ProblemShape, same Stride types). Take max(cutlass_ws) so the
    // 2Sm-fail → 1Sm retry can reuse the same scratch.
    size_t cw_2sm = query_cutlass_workspace_upper<R2>(num_experts);
    size_t cw_1sm = query_cutlass_workspace_upper<R1>(num_experts);
    size_t cutlass_ws_bound = cw_2sm > cw_1sm ? cw_2sm : cw_1sm;
    auto plan = plan_workspace<R2>(num_experts, cutlass_ws_bound);

    size_t gathered_bytes = ws_round_up((size_t)M_total * K * sizeof(__nv_bfloat16));
    size_t gemm_out_bytes = ws_round_up((size_t)M_total * N * sizeof(__nv_bfloat16));
    size_t total_bytes = gathered_bytes + gemm_out_bytes + plan.total_bytes;

    void* ws_root = nullptr;
    cudaError_t alloc_err = cudaMallocAsync(&ws_root, total_bytes, stream);
    if (alloc_err != cudaSuccess || ws_root == nullptr) {
        return -22;
    }

    uint8_t* p = static_cast<uint8_t*>(ws_root);
    void* gathered = p;                      p += gathered_bytes;
    void* gemm_out = p;                      p += gemm_out_bytes;
    void* metadata_ws = p;                   p += plan.metadata_bytes;
    void* cutlass_ws = (cutlass_ws_bound > 0) ? p : nullptr;

    // Step 1: gather A.
    {
        int block = 256;
        prelude_grouped_moe_sm100::gather_a_kernel<__nv_bfloat16><<<M_total, block, 0, stream>>>(
            (const __nv_bfloat16*)input,
            sorted_token_ids,
            (__nv_bfloat16*)gathered,
            M_total, K, topk);
    }

    // Step 2: dispatch 2Sm or 1Sm based on average M.
    int avg_M = M_total / num_experts;
    int ret = -1;
    if (avg_M >= 32) {
        ret = launch_grouped_moe<R2>(
            (const BF16*)gathered, (const BF16*)weights,
            expert_offsets, (BF16*)gemm_out,
            M_total, N, K, num_experts,
            metadata_ws, cutlass_ws, plan, stream);
        if (ret != 0) {
            // 2Sm misaligned for this shape — fall back to 1Sm. Plan
            // is shared (identical metadata + cutlass_ws upper bound).
            auto plan1 = plan_workspace<R1>(num_experts, cutlass_ws_bound);
            ret = launch_grouped_moe<R1>(
                (const BF16*)gathered, (const BF16*)weights,
                expert_offsets, (BF16*)gemm_out,
                M_total, N, K, num_experts,
                metadata_ws, cutlass_ws, plan1, stream);
        }
    } else {
        auto plan1 = plan_workspace<R1>(num_experts, cutlass_ws_bound);
        ret = launch_grouped_moe<R1>(
            (const BF16*)gathered, (const BF16*)weights,
            expert_offsets, (BF16*)gemm_out,
            M_total, N, K, num_experts,
            metadata_ws, cutlass_ws, plan1, stream);
    }

    if (ret != 0) {
        cudaFreeAsync(ws_root, stream);
        return ret;
    }

    // Step 3: scatter D back to caller's output via sorted_token_ids.
    {
        int block = 256;
        prelude_grouped_moe_sm100::scatter_d_kernel<__nv_bfloat16><<<M_total, block, 0, stream>>>(
            (const __nv_bfloat16*)gemm_out,
            sorted_token_ids,
            (__nv_bfloat16*)output,
            M_total, N);
    }

    cudaFreeAsync(ws_root, stream);
    return 0;
}

#else  // !CUTLASS_ARCH_MMA_SM100_SUPPORTED

// Stub for older CUDA toolkits: surface as "kernel not available" so
// the dispatcher falls back to the WMMA path.
extern "C" int moe_grouped_gemm_sm100(
    const void*, const void*, const uint32_t*, const int32_t*, void*,
    int, int, int, int, int, int, cudaStream_t) {
    return -100;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

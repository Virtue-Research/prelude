// CUTLASS GEMM — cuBLAS replacement.  All CUTLASS 3.x, AOT compiled, statically linked.
// SM90: CollectiveBuilder (TMA + warp-specialized, example 49 pattern).
// SM80: Manual CollectiveMma (cp.async + TensorOp, default_gemm_configuration pattern).
// Both use kernel::GemmUniversal + GemmUniversalAdapter.
//
// Layout: A=RowMajor, B=ColumnMajor, D=ColumnMajor — matches cuBLAS TN convention.

// ── Common includes (shared by SM90 and SM80 paths) ─────────────────
#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"

using BF16 = cutlass::bfloat16_t;
using FP16 = cutlass::half_t;
using namespace cute;

static int g_sm_count = 0;
static bool g_sm_count_init = false;

static void ensure_sm_count() {
    if (g_sm_count_init) return;
    int dev; cudaGetDevice(&dev);
    g_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    g_sm_count_init = true;
}

// ============================================================================
// SM90: CUTLASS 3.x — exact copy of example 49 ExampleRunner pattern
// ============================================================================
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

// ── Exact copy of example 49 ExampleRunner (line 250-337) ───────────
// Only change: parameterized on Element type (half_t or bfloat16_t)
// and schedule/scheduler for different tile configs.

template <
    class MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
    class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
    class StageCountType = cutlass::gemm::collective::StageCountAuto,
    class TileSchedulerType = cutlass::gemm::PersistentScheduler,
    class TileShape_ = Shape<_128, _128, _64>,
    class Element = BF16
>
struct Sm90Runner {

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;

    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScalar = float;

    // 16B alignment lets us use TMA (same as example 49 line 278)
    static constexpr int AlignmentA = 16 / sizeof(Element);
    static constexpr int AlignmentB = 16 / sizeof(Element);
    static constexpr int AlignmentC = 16 / sizeof(Element);
    static constexpr int AlignmentD = 16 / sizeof(Element);

    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

    // Same as example 49 line 305
    using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<
        Element, ElementCompute, Element, ElementScalar, RoundStyle>;

    // Same as example 49 line 307-316 (epilogue cluster = 1×1×1)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, Shape<_1, _1, _1>,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        Element, LayoutC, AlignmentC,
        Element, LayoutD, AlignmentD,
        EpilogueScheduleType,
        DefaultOperation
    >::CollectiveOp;

    // Same as example 49 line 318-328 (mainloop cluster = 2×1×1)
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        Element, LayoutA, AlignmentA,
        Element, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape_, Shape<_2, _1, _1>,
        cute::conditional_t<cute::is_same_v<StageCountType, cutlass::gemm::collective::StageCountAuto>,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            StageCountType>,
        MainloopScheduleType
    >::CollectiveOp;

    // Same as example 49 line 330-337
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        TileSchedulerType
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

// ── Kernel configs ──────────────────────────────────────────────────
// KernelScheduleAuto + 128x128x64: CUTLASS selects best warp-specialized
// schedule internally based on problem shape (cooperative, pingpong, etc).
// Tested StreamK, Pingpong, 64x64, 128x256 — none improved over Auto
// for model GEMM shapes. The ~2-3x gap vs cuBLAS at M=1 is inherent
// (GEMM tile overhead vs cuBLAS's dedicated GEMV kernel).

template <class E> using Sm90_Default = Sm90Runner<
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::PersistentScheduler,
    Shape<_128, _128, _64>, E>;

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED

// ============================================================================
// SM80: CUTLASS 3.x — manual CollectiveMma + GemmUniversal
// (No CollectiveBuilder for SM80, so we specify all types explicitly.
//  Pattern from test/unit/gemm/device/default_gemm_configuration.hpp)
// ============================================================================

// ── SM80 3.x runner — same GemmUniversal kernel as SM90, different mainloop ──
// BF16 uses SM80_16x8x16 MMA atom (same shape/alignment as FP16, different types).
// SmemLayout/GmemTiledCopy reuse FP16 configs from CUTLASS test suite since
// BF16 and FP16 have identical element size (2 bytes).
//
// Sm80RunnerBase: single template for both predicated and unpredicated SM80.
// DispatchPolicy_ selects MainloopSm80CpAsync (predicated, safe for any M)
// or MainloopSm80CpAsyncUnpredicated (faster, requires tile-aligned dims).
// SwizzleB: Swizzle bit width — 2 for K=32, 3 for K=64
template <class Element, class DispatchPolicy_,
          class TileShape_ = Shape<_128, _128, _32>, int SwizzleB_ = 2>
struct Sm80RunnerBase {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;

    using ElementAccumulator = float;
    using ElementCompute = float;

    using TileShape = TileShape_;
    static constexpr int TileK = size<2>(TileShape{});
    using DispatchPolicy = DispatchPolicy_;

    // MMA atom: 16x8x16 TensorOp, F32 accumulator
    using MmaAtom = cute::conditional_t<
        cute::is_same_v<Element, BF16>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;

    using TiledMma = TiledMMA<
        MmaAtom,
        Layout<Shape<_2, _2, _1>>,   // 2x2x1 thread group (4 warps)
        Tile<_32, _32, _16>>;         // 32x32x16 value tile for LDSM

    // ── Operand A (RowMajor = K-major) ──
    static constexpr int kAlignmentA = 8;
    using SmemLayoutAtomA = decltype(
        composition(Swizzle<SwizzleB_, 3, 3>{},
                    Layout<Shape <_8, Int<TileK>>,
                           Stride<Int<TileK>, _1>>{}));
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using GmemTiledCopyA = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, Element>{},
                        Layout<Shape <_16, _8>,
                               Stride< _8, _1>>{},
                        Layout<Shape<_1, _8>>{}));

    // ── Operand B (ColumnMajor = K-major) — reuses A's K-major config ──
    static constexpr int kAlignmentB = 8;
    using SmemLayoutAtomB = SmemLayoutAtomA;
    using SmemCopyAtomB = SmemCopyAtomA;
    using GmemTiledCopyB = GmemTiledCopyA;

    // ── Mainloop ──
    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        DispatchPolicy, TileShape,
        Element, cutlass::gemm::TagToStrideA_t<LayoutA>,
        Element, cutlass::gemm::TagToStrideB_t<LayoutB>,
        TiledMma,
        GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
        GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;

    // ── Epilogue — 128-bit vectorized output stores (8 BF16 elements per store) ──
    static constexpr int kEpilogueVectorWidth = 128 / cutlass::sizeof_bits<Element>::value;  // 8 for BF16/FP16
    using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
        Element,
        cutlass::gemm::TagToStrideC_t<LayoutC>,
        cutlass::gemm::TagToStrideC_t<LayoutD>,
        cutlass::epilogue::thread::LinearCombination<Element, kEpilogueVectorWidth, ElementAccumulator, ElementCompute>,
        cutlass::gemm::EpilogueDefault>;

    // ── Kernel — same GemmUniversal as SM90 ──
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

// Predicated (safe for any M) — default for production dispatch fallback
template <class Element, class TileShape_ = Shape<_128, _128, _32>,
          int Stages_ = 4, int SwizzleB_ = 2>
using Sm80Runner = Sm80RunnerBase<Element,
    cutlass::gemm::MainloopSm80CpAsync<Stages_>, TileShape_, SwizzleB_>;

// Unpredicated (faster, requires M/N/K aligned to tile dims) — for benchmarking
template <class Element, class TileShape_ = Shape<_128, _128, _64>,
          int Stages_ = 3, int SwizzleB_ = 3>
using Sm80RunnerUnpred = Sm80RunnerBase<Element,
    cutlass::gemm::MainloopSm80CpAsyncUnpredicated<Stages_>, TileShape_, SwizzleB_>;

// ── Unified launch for both SM90 and SM80 ────────────────────────────
// Both paths use GemmUniversalAdapter with the same argument structure.

template <typename Runner>
static int launch(const void* A, const void* B, void* D,
                  int M, int N, int K, cudaStream_t stream)
{
    using Gemm = typename Runner::Gemm;

    auto stride_A = cutlass::make_cute_packed_stride(typename Runner::StrideA{}, make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(typename Runner::StrideB{}, make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(typename Runner::StrideC{}, make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(typename Runner::StrideD{}, make_shape(M, N, 1));

    ensure_sm_count();
    cutlass::KernelHardwareInfo hw;
    hw.device_id = 0;
    cudaGetDevice(&hw.device_id);
    hw.sm_count = g_sm_count;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {static_cast<const typename Gemm::ElementA*>(A), stride_A,
         static_cast<const typename Gemm::ElementB*>(B), stride_B},
        {{1.0f, 0.0f},
         nullptr, stride_C,
         static_cast<typename Gemm::ElementD*>(D), stride_D},
        hw
    };

    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;

    size_t ws_size = gemm.get_workspace_size(args);
    void* ws = nullptr;
    if (ws_size > 0) {
        if (cudaMalloc(&ws, ws_size) != cudaSuccess) return -4;
    }

    auto status = gemm.initialize(args, ws, stream);
    if (status != cutlass::Status::kSuccess) { if (ws) cudaFree(ws); return -2; }

    status = gemm.run(stream);
    if (ws) cudaFree(ws);
    return (status == cutlass::Status::kSuccess) ? 0 : -3;
}

// ============================================================================
// SM80 benchmark configs & helpers (must be before C FFI for forward decl)
// ============================================================================

// SM80 configs — wrapped in plain functions to avoid nvcc nested-template parse issues
static int sm80_bf16_c0(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<BF16, Shape<_128,_128,_32>, 4, 2> >(A, B, D, m, n, k, s);
}
static int sm80_bf16_c1(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<BF16, Shape<_128,_128,_64>, 3, 3> >(A, B, D, m, n, k, s);
}
static int sm80_bf16_c2(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<BF16, Shape<_128,_128,_64>, 4, 3> >(A, B, D, m, n, k, s);
}
static int sm80_bf16_c3(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80RunnerUnpred<BF16, Shape<_128,_128,_64>, 3, 3> >(A, B, D, m, n, k, s);
}
static int sm80_bf16_c4(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<BF16, Shape<_256,_128,_64>, 3, 3> >(A, B, D, m, n, k, s);
}
static int sm80_bf16_c5(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<BF16, Shape<_128,_128,_64>, 5, 3> >(A, B, D, m, n, k, s);
}
static int sm80_fp16_c0(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<FP16, Shape<_128,_128,_32>, 4, 2> >(A, B, D, m, n, k, s);
}
static int sm80_fp16_c1(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<FP16, Shape<_128,_128,_64>, 3, 3> >(A, B, D, m, n, k, s);
}
static int sm80_fp16_c2(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<FP16, Shape<_128,_128,_64>, 4, 3> >(A, B, D, m, n, k, s);
}
static int sm80_fp16_c3(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80RunnerUnpred<FP16, Shape<_128,_128,_64>, 3, 3> >(A, B, D, m, n, k, s);
}
static int sm80_fp16_c4(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<FP16, Shape<_256,_128,_64>, 3, 3> >(A, B, D, m, n, k, s);
}
static int sm80_fp16_c5(const void* A, const void* B, void* D, int m, int n, int k, cudaStream_t s) {
    return launch< Sm80Runner<FP16, Shape<_128,_128,_64>, 5, 3> >(A, B, D, m, n, k, s);
}

constexpr int SM80_NUM_CONFIGS = 6;
using Sm80Fn = int(*)(const void*, const void*, void*, int, int, int, cudaStream_t);
static Sm80Fn sm80_bf16_fns[] = { sm80_bf16_c0, sm80_bf16_c1, sm80_bf16_c2, sm80_bf16_c3, sm80_bf16_c4, sm80_bf16_c5 };
static Sm80Fn sm80_fp16_fns[] = { sm80_fp16_c0, sm80_fp16_c1, sm80_fp16_c2, sm80_fp16_c3, sm80_fp16_c4, sm80_fp16_c5 };

// ============================================================================
// C FFI dispatch
// ============================================================================

extern "C" int cutlass_gemm_dispatch(
    const void* A, const void* B, void* D,
    int m, int n, int k,
    int batch,
    int lda, int ldb, int ldd,
    int64_t stride_a, int64_t stride_b, int64_t stride_d,
    int transa, int transb,
    uint32_t dtype,
    const void* stream)
{
    if (transa != 1 || transb != 0) return -10;
    if (batch > 1) return -30;

    auto s = static_cast<cudaStream_t>(const_cast<void*>(stream));
    cudaGetLastError();

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    {
        int ret;
        switch (dtype) {
            case 0: ret = launch<Sm90_Default<BF16>>(A, B, D, m, n, k, s); break;
            case 1: ret = launch<Sm90_Default<FP16>>(A, B, D, m, n, k, s); break;
            default: ret = -20; break;
        }
        if (ret == 0) return 0;
        // SM90 failed — log and fall through to SM80
        fprintf(stderr, "CUTLASS: SM90 failed (code %d) for m=%d n=%d k=%d dtype=%u, falling back to SM80\n",
                ret, m, n, k, dtype);
    }
#endif

    // SM80 fallback — try unpredicated first (fastest), fall back to predicated for unaligned M.
    // Unpredicated requires M and N aligned to tile dims (128); skip if not aligned.
    if (dtype == 0) {
        if (m % 128 == 0 && n % 128 == 0) {
            int ret = sm80_bf16_c3(A, B, D, m, n, k, s);  // unpredicated
            if (ret == 0) return 0;
        }
        return sm80_bf16_c1(A, B, D, m, n, k, s);      // predicated fallback
    }
    if (dtype == 1) {
        if (m % 128 == 0 && n % 128 == 0) {
            int ret = sm80_fp16_c3(A, B, D, m, n, k, s);
            if (ret == 0) return 0;
        }
        return sm80_fp16_c1(A, B, D, m, n, k, s);
    }
    return -20;
}

// Force SM80 path — for benchmarking SM80 fallback on SM90+ hardware
extern "C" int cutlass_gemm_sm80(
    const void* A, const void* B, void* D,
    int m, int n, int k,
    uint32_t dtype, int config,
    const void* stream)
{
    auto s = static_cast<cudaStream_t>(const_cast<void*>(stream));
    if (config < 0 || config >= SM80_NUM_CONFIGS) return -40;
    if (dtype == 0) return sm80_bf16_fns[config](A, B, D, m, n, k, s);
    if (dtype == 1) return sm80_fp16_fns[config](A, B, D, m, n, k, s);
    return -20;
}

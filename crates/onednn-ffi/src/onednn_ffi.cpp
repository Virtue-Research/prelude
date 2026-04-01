// oneDNN FFI wrapper — thin C interface over oneDNN's matmul primitive.
//
// Design:
//   - Global engine + stream created once via onednn_init()
//   - Primitive + memory desc cache keyed by (M, K, N, transpose_B, dtype)
//   - Memory objects created/destroyed per call (lightweight, ~50µs for 3 pairs)
//   - All calls serialized from Rust side (one model forward at a time)
//   - Threading: DNNL_CPU_RUNTIME=THREADPOOL — oneDNN calls our RayonThreadPool
//     adapter which dispatches parallel work to Rust's rayon pool. No OpenMP.
//
// Weight packing:
//   - onednn_{bf16,f32}_pack_weights() reorders weights from user format to
//     oneDNN's optimal blocked format (format_tag_any). Done once at model load.
//   - onednn_{bf16,f32}_linear_packed() uses pre-packed weights.
//   - Packed primitives are cached keyed by (M, K, N, dtype).

#include "onednn_ffi.h"
#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_threadpool.h"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <functional>

// Per-GEMM phase timing (enabled by ONEDNN_PROFILE=1 env var)
static bool g_profile = false;
static int g_profile_call_id = 0;

// ── Rayon callbacks (defined in Rust, resolved at link time) ────────────
// These symbols are exported by Rust via #[no_mangle] extern "C" functions.
extern "C" {
    void rayon_parallel_for(int n, void(*body)(int, int, void*), void* context);
    int rayon_get_num_threads(void);
    int rayon_get_in_parallel(void);
}

// ── RayonThreadPool: bridges oneDNN's threadpool interface to Rust's rayon ──
class RayonThreadPool : public dnnl::threadpool_interop::threadpool_iface {
public:
    int get_num_threads() const override {
        return rayon_get_num_threads();
    }

    bool get_in_parallel() const override {
        return rayon_get_in_parallel() != 0;
    }

    void parallel_for(int n, const std::function<void(int, int)>& fn) override {
        // Wrap std::function into a C-compatible trampoline.
        // The fn reference is valid for the duration of this call (synchronous).
        struct Context {
            const std::function<void(int, int)>* fn;
            int n;
        };
        Context ctx{&fn, n};
        rayon_parallel_for(n, [](int i, int n, void* raw) {
            auto* c = static_cast<Context*>(raw);
            (*c->fn)(i, c->n);
        }, &ctx);
    }

    uint64_t get_flags() const override {
        return 0; // synchronous — parallel_for returns only after all work is done
    }

    void wait() override {
        // no-op for synchronous threadpool
    }
};

#define CHECK_DNNL(f) do {                                                \
    dnnl_status_t s_ = (f);                                               \
    if (s_ != dnnl_success) {                                             \
        fprintf(stderr, "oneDNN error: %s returned %d at %s:%d\n",       \
                #f, (int)s_, __FILE__, __LINE__);                         \
        abort();                                                          \
    }                                                                     \
} while(0)

static dnnl_engine_t g_engine = nullptr;
static dnnl_stream_t g_stream = nullptr;
static RayonThreadPool g_threadpool;

// ── Unpacked path cache ──────────────────────────────────────────────────

struct CacheKey {
    int64_t m, k, n;
    bool transpose_b;
    dnnl_data_type_t dtype;
    bool operator==(const CacheKey& o) const {
        return m == o.m && k == o.k && n == o.n && transpose_b == o.transpose_b && dtype == o.dtype;
    }
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& key) const {
        size_t h = std::hash<int64_t>()(key.m);
        h ^= std::hash<int64_t>()(key.k) * 2654435761ULL;
        h ^= std::hash<int64_t>()(key.n) * 40503ULL;
        h ^= std::hash<bool>()(key.transpose_b) * 97ULL;
        h ^= std::hash<int>()(key.dtype) * 131ULL;
        return h;
    }
};

struct CachedPrimitive {
    dnnl_primitive_t prim;
    dnnl_memory_desc_t src_md, weights_md, dst_md;
};

static thread_local std::unordered_map<CacheKey, CachedPrimitive, CacheKeyHash> g_cache;

static const CachedPrimitive& get_or_create(int64_t m, int64_t k, int64_t n,
                                             bool transpose_b, dnnl_data_type_t dt) {
    CacheKey key{m, k, n, transpose_b, dt};
    auto it = g_cache.find(key);
    if (it != g_cache.end()) return it->second;

    CachedPrimitive entry{};
    dnnl_dims_t src_dims = {m, k};
    dnnl_dims_t dst_dims = {m, n};

    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&entry.src_md, 2, src_dims, dt, dnnl_ab));
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&entry.dst_md, 2, dst_dims, dt, dnnl_ab));

    dnnl_dims_t w_dims = {k, n};
    if (transpose_b) {
        CHECK_DNNL(dnnl_memory_desc_create_with_tag(&entry.weights_md, 2, w_dims, dt, dnnl_ba));
    } else {
        CHECK_DNNL(dnnl_memory_desc_create_with_tag(&entry.weights_md, 2, w_dims, dt, dnnl_ab));
    }

    dnnl_primitive_desc_t pd = nullptr;
    CHECK_DNNL(dnnl_matmul_primitive_desc_create(&pd, g_engine,
        entry.src_md, entry.weights_md, nullptr, entry.dst_md, nullptr));
    CHECK_DNNL(dnnl_primitive_create(&entry.prim, pd));
    dnnl_primitive_desc_destroy(pd);

    auto [inserted, _] = g_cache.emplace(key, entry);
    return inserted->second;
}

static void execute_matmul(const CachedPrimitive& cached,
                           const void* src, const void* weights, void* dst) {
    dnnl_memory_t src_mem, weights_mem, dst_mem;
    CHECK_DNNL(dnnl_memory_create(&src_mem, cached.src_md, g_engine, const_cast<void*>(src)));
    CHECK_DNNL(dnnl_memory_create(&weights_mem, cached.weights_md, g_engine, const_cast<void*>(weights)));
    CHECK_DNNL(dnnl_memory_create(&dst_mem, cached.dst_md, g_engine, dst));

    dnnl_exec_arg_t args[3] = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem},
    };
    CHECK_DNNL(dnnl_primitive_execute(cached.prim, g_stream, 3, args));
    CHECK_DNNL(dnnl_stream_wait(g_stream));

    dnnl_memory_destroy(dst_mem);
    dnnl_memory_destroy(weights_mem);
    dnnl_memory_destroy(src_mem);
}

// ── Packed weight path ──────────────────────────────────────────────────

struct onednn_packed_weights {
    void* data;                  // packed weight buffer (oneDNN blocked format)
    size_t size;                 // buffer size in bytes
    int64_t k, n;                // original dimensions
    dnnl_data_type_t dtype;      // data type (dnnl_bf16 or dnnl_f32)
    dnnl_memory_desc_t packed_md; // the blocked memory desc
};

struct PackedCacheKey {
    int64_t m, k, n;
    dnnl_data_type_t dtype;
    bool operator==(const PackedCacheKey& o) const {
        return m == o.m && k == o.k && n == o.n && dtype == o.dtype;
    }
};

struct PackedCacheKeyHash {
    size_t operator()(const PackedCacheKey& key) const {
        size_t h = std::hash<int64_t>()(key.m);
        h ^= std::hash<int64_t>()(key.k) * 2654435761ULL;
        h ^= std::hash<int64_t>()(key.n) * 40503ULL;
        h ^= std::hash<int>()(key.dtype) * 131ULL;
        return h;
    }
};

struct PackedCachedPrimitive {
    dnnl_primitive_t prim;
    dnnl_memory_desc_t src_md, dst_md;
    dnnl_memory_desc_t weights_md; // = the packed blocked md
};

static std::unordered_map<PackedCacheKey, PackedCachedPrimitive, PackedCacheKeyHash> g_packed_cache;

// ── Internal helpers for pack/linear_packed ──────────────────────────────

static onednn_packed_weights_t pack_weights_impl(
    const void* weight, int64_t k, int64_t n, int64_t ref_m, dnnl_data_type_t dt)
{
    dnnl_dims_t src_dims = {ref_m, k};
    dnnl_dims_t w_dims = {k, n};
    dnnl_dims_t dst_dims = {ref_m, n};

    dnnl_memory_desc_t src_md, w_user_md, w_any_md, dst_md;
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&src_md, 2, src_dims, dt, dnnl_ab));
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&dst_md, 2, dst_dims, dt, dnnl_ab));
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&w_user_md, 2, w_dims, dt, dnnl_ba));
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&w_any_md, 2, w_dims, dt, dnnl_format_tag_any));

    dnnl_primitive_desc_t pd = nullptr;
    CHECK_DNNL(dnnl_matmul_primitive_desc_create(&pd, g_engine,
        src_md, w_any_md, nullptr, dst_md, nullptr));

    const_dnnl_memory_desc_t queried_w_md = dnnl_primitive_desc_query_md(pd, dnnl_query_weights_md, 0);

    size_t packed_size = dnnl_memory_desc_get_size(queried_w_md);
    void* packed_data = malloc(packed_size);
    if (!packed_data) {
        fprintf(stderr, "oneDNN pack_weights: malloc(%zu) failed\n", packed_size);
        abort();
    }

    dnnl_memory_t user_mem, packed_mem;
    CHECK_DNNL(dnnl_memory_create(&user_mem, w_user_md, g_engine, const_cast<void*>(weight)));
    CHECK_DNNL(dnnl_memory_create(&packed_mem, queried_w_md, g_engine, packed_data));

    dnnl_primitive_desc_t reorder_pd = nullptr;
    CHECK_DNNL(dnnl_reorder_primitive_desc_create(&reorder_pd,
        w_user_md, g_engine, queried_w_md, g_engine, nullptr));

    dnnl_primitive_t reorder_prim = nullptr;
    CHECK_DNNL(dnnl_primitive_create(&reorder_prim, reorder_pd));

    dnnl_exec_arg_t reorder_args[2] = {
        {DNNL_ARG_SRC, user_mem},
        {DNNL_ARG_DST, packed_mem},
    };
    CHECK_DNNL(dnnl_primitive_execute(reorder_prim, g_stream, 2, reorder_args));
    CHECK_DNNL(dnnl_stream_wait(g_stream));

    dnnl_primitive_destroy(reorder_prim);
    dnnl_primitive_desc_destroy(reorder_pd);
    dnnl_memory_destroy(packed_mem);
    dnnl_memory_destroy(user_mem);

    dnnl_memory_desc_t cloned_md = nullptr;
    CHECK_DNNL(dnnl_memory_desc_clone(&cloned_md, queried_w_md));

    dnnl_primitive_desc_destroy(pd);
    dnnl_memory_desc_destroy(w_any_md);
    dnnl_memory_desc_destroy(w_user_md);
    dnnl_memory_desc_destroy(dst_md);
    dnnl_memory_desc_destroy(src_md);

    auto* pw = new onednn_packed_weights();
    pw->data = packed_data;
    pw->size = packed_size;
    pw->k = k;
    pw->n = n;
    pw->dtype = dt;
    pw->packed_md = cloned_md;
    return pw;
}

static void linear_packed_impl(
    const void* input, onednn_packed_weights_t pw, void* output, int64_t m)
{
    if (m == 0 || pw->k == 0 || pw->n == 0) return;
    int64_t k = pw->k;
    int64_t n = pw->n;
    dnnl_data_type_t dt = pw->dtype;

    auto t0 = std::chrono::steady_clock::now();

    PackedCacheKey key{m, k, n, dt};
    auto it = g_packed_cache.find(key);
    bool cache_miss = (it == g_packed_cache.end());

    if (cache_miss) {
        PackedCachedPrimitive entry{};
        dnnl_dims_t src_dims = {m, k};
        dnnl_dims_t dst_dims = {m, n};

        CHECK_DNNL(dnnl_memory_desc_create_with_tag(&entry.src_md, 2, src_dims, dt, dnnl_ab));
        CHECK_DNNL(dnnl_memory_desc_create_with_tag(&entry.dst_md, 2, dst_dims, dt, dnnl_ab));
        entry.weights_md = pw->packed_md;

        dnnl_primitive_desc_t pd = nullptr;
        CHECK_DNNL(dnnl_matmul_primitive_desc_create(&pd, g_engine,
            entry.src_md, pw->packed_md, nullptr, entry.dst_md, nullptr));
        CHECK_DNNL(dnnl_primitive_create(&entry.prim, pd));
        dnnl_primitive_desc_destroy(pd);

        auto [inserted, _] = g_packed_cache.emplace(key, entry);
        it = inserted;
    }

    auto t1 = std::chrono::steady_clock::now();

    const auto& cached = it->second;

    dnnl_memory_t src_mem, weights_mem, dst_mem;
    CHECK_DNNL(dnnl_memory_create(&src_mem, cached.src_md, g_engine, const_cast<void*>(input)));
    CHECK_DNNL(dnnl_memory_create(&weights_mem, pw->packed_md, g_engine, pw->data));
    CHECK_DNNL(dnnl_memory_create(&dst_mem, cached.dst_md, g_engine, output));

    auto t2 = std::chrono::steady_clock::now();

    dnnl_exec_arg_t args[3] = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem},
    };
    CHECK_DNNL(dnnl_primitive_execute(cached.prim, g_stream, 3, args));
    CHECK_DNNL(dnnl_stream_wait(g_stream));

    auto t3 = std::chrono::steady_clock::now();

    dnnl_memory_destroy(dst_mem);
    dnnl_memory_destroy(weights_mem);
    dnnl_memory_destroy(src_mem);

    auto t4 = std::chrono::steady_clock::now();

    if (g_profile) {
        auto us = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
        };
        int id = g_profile_call_id++;
        fprintf(stderr, "onednn_gemm #%d M=%ld K=%ld N=%ld cache=%s "
                "lookup=%ldus mem_create=%ldus execute=%ldus mem_destroy=%ldus total=%ldus\n",
                id, m, k, n, cache_miss ? "MISS" : "hit",
                us(t0, t1), us(t1, t2), us(t2, t3), us(t3, t4), us(t0, t4));
    }
}

// ── Public API ──────────────────────────────────────────────────────────

void onednn_init(void) {
    if (g_engine) return;
    CHECK_DNNL(dnnl_engine_create(&g_engine, dnnl_cpu, 0));
    // Create stream with our rayon-backed threadpool
    CHECK_DNNL(dnnl_threadpool_interop_stream_create(
        &g_stream, g_engine, &g_threadpool));
    // Set max concurrency so oneDNN's internal scratchpad sizing is correct
    CHECK_DNNL(dnnl_threadpool_interop_set_max_concurrency(
        g_threadpool.get_num_threads()));
    // Enable per-GEMM profiling if ONEDNN_PROFILE=1
    const char* p = getenv("ONEDNN_PROFILE");
    g_profile = (p && p[0] == '1');
}

void onednn_cleanup(void) {
    for (auto& [key, entry] : g_cache) {
        dnnl_primitive_destroy(entry.prim);
        dnnl_memory_desc_destroy(entry.src_md);
        dnnl_memory_desc_destroy(entry.weights_md);
        dnnl_memory_desc_destroy(entry.dst_md);
    }
    g_cache.clear();
    for (auto& [key, entry] : g_packed_cache) {
        dnnl_primitive_destroy(entry.prim);
        dnnl_memory_desc_destroy(entry.src_md);
        dnnl_memory_desc_destroy(entry.dst_md);
    }
    g_packed_cache.clear();
    if (g_stream) { dnnl_stream_destroy(g_stream); g_stream = nullptr; }
    if (g_engine) { dnnl_engine_destroy(g_engine); g_engine = nullptr; }
}

void onednn_set_num_threads(int num_threads) {
    // With THREADPOOL runtime, thread count is controlled by rayon pool size.
    // This is a no-op but kept for API compatibility.
    (void)num_threads;
}

void onednn_bind_threads(const int* cpu_ids, int num_threads) {
    // With THREADPOOL runtime, thread affinity is controlled by rayon/NUMA init.
    // This is a no-op but kept for API compatibility.
    (void)cpu_ids;
    (void)num_threads;
}

// ── BF16 API ────────────────────────────────────────────────────────────

void onednn_bf16_linear(const void* input, const void* weight, void* output,
                        int64_t m, int64_t k, int64_t n) {
    if (m == 0 || k == 0 || n == 0) return;
    const auto& cached = get_or_create(m, k, n, true, dnnl_bf16);
    execute_matmul(cached, input, weight, output);
}

void onednn_bf16_matmul(const void* a, const void* b, void* c,
                        int64_t m, int64_t k, int64_t n) {
    if (m == 0 || k == 0 || n == 0) return;
    const auto& cached = get_or_create(m, k, n, false, dnnl_bf16);
    execute_matmul(cached, a, b, c);
}

onednn_packed_weights_t onednn_bf16_pack_weights(
    const void* weight, int64_t k, int64_t n, int64_t ref_m) {
    return pack_weights_impl(weight, k, n, ref_m, dnnl_bf16);
}

void onednn_bf16_linear_packed(
    const void* input, onednn_packed_weights_t pw, void* output, int64_t m) {
    linear_packed_impl(input, pw, output, m);
}

// ── F32 API ─────────────────────────────────────────────────────────────

void onednn_f32_linear(const void* input, const void* weight, void* output,
                       int64_t m, int64_t k, int64_t n) {
    if (m == 0 || k == 0 || n == 0) return;
    const auto& cached = get_or_create(m, k, n, true, dnnl_f32);
    execute_matmul(cached, input, weight, output);
}

void onednn_f32_matmul(const void* a, const void* b, void* c,
                       int64_t m, int64_t k, int64_t n) {
    if (m == 0 || k == 0 || n == 0) return;
    const auto& cached = get_or_create(m, k, n, false, dnnl_f32);
    execute_matmul(cached, a, b, c);
}

onednn_packed_weights_t onednn_f32_pack_weights(
    const void* weight, int64_t k, int64_t n, int64_t ref_m) {
    return pack_weights_impl(weight, k, n, ref_m, dnnl_f32);
}

void onednn_f32_linear_packed(
    const void* input, onednn_packed_weights_t pw, void* output, int64_t m) {
    linear_packed_impl(input, pw, output, m);
}

// ── Common ──────────────────────────────────────────────────────────────

void onednn_packed_weights_destroy(onednn_packed_weights_t pw) {
    if (!pw) return;
    free(pw->data);
    dnnl_memory_desc_destroy(pw->packed_md);
    delete pw;
}

// ── BRGeMM micro-kernel path ────────────────────────────────────────────
// Uses oneDNN's brgemm ukernel API for near-zero-overhead BF16 GEMM.
// Weight is pre-packed in VNNI block format. JIT'd kernels are cached
// per-thread. Caller handles parallelism (rayon from Rust side).

#ifdef DNNL_EXPERIMENTAL_UKERNEL
#include "oneapi/dnnl/dnnl_ukernel.h"

#define BRGEMM_BLOCK_N 32

struct brgemm_packed_b {
    void* data;           // packed weight buffer, 64-byte aligned
    int64_t k, n;         // original dimensions
    int64_t block_n;      // = BRGEMM_BLOCK_N
    int64_t n_blocks;     // = ceil(n / block_n)
};

// Thread-local brgemm kernel cache
struct BrgemmKey {
    int64_t m, n, k;
    bool operator==(const BrgemmKey& o) const { return m==o.m && n==o.n && k==o.k; }
};

struct BrgemmKeyHash {
    size_t operator()(const BrgemmKey& key) const {
        return (size_t)(key.m * 1000003 + key.n * 1009 + key.k);
    }
};

struct BrgemmKernel {
    dnnl_brgemm_t brgemm = nullptr;
    std::vector<uint8_t> scratchpad;
};

static thread_local std::unordered_map<BrgemmKey, BrgemmKernel, BrgemmKeyHash> tls_brgemm;
static thread_local const BrgemmKernel* tls_current = nullptr;

static BrgemmKernel& get_brgemm_kernel(int64_t m, int64_t n, int64_t k) {
    BrgemmKey key{m, n, k};
    auto it = tls_brgemm.find(key);
    if (it != tls_brgemm.end()) return it->second;

    BrgemmKernel kern{};
    CHECK_DNNL(dnnl_brgemm_create(&kern.brgemm, m, n, k,
        /*batch_size*/1, /*lda*/k, /*ldb*/n, /*ldc*/n,
        dnnl_bf16, dnnl_bf16, dnnl_f32));
    CHECK_DNNL(dnnl_brgemm_set_add_C(kern.brgemm, 0));
    CHECK_DNNL(dnnl_brgemm_finalize(kern.brgemm));
    CHECK_DNNL(dnnl_brgemm_generate(kern.brgemm));

    size_t sz = 0;
    CHECK_DNNL(dnnl_brgemm_get_scratchpad_size(kern.brgemm, &sz));
    kern.scratchpad.resize(sz);

    auto [inserted, _] = tls_brgemm.emplace(key, std::move(kern));
    return inserted->second;
}

int brgemm_available(void) {
    // Try creating a small brgemm to check if the platform supports it
    static int cached = -1;
    if (cached >= 0) return cached;

    dnnl_brgemm_t brg = nullptr;
    dnnl_status_t s = dnnl_brgemm_create(&brg, 1, 32, 64,
        1, 64, 32, 32, dnnl_bf16, dnnl_bf16, dnnl_f32);
    if (s == dnnl_success && brg) {
        s = dnnl_brgemm_finalize(brg);
        if (s == dnnl_success) {
            s = dnnl_brgemm_generate(brg);
        }
        dnnl_brgemm_destroy(brg);
        cached = (s == dnnl_success) ? 1 : 0;
    } else {
        if (brg) dnnl_brgemm_destroy(brg);
        cached = 0;
    }
    return cached;
}

brgemm_packed_b_t brgemm_bf16_pack(const void* weight_ptr, int64_t k, int64_t n) {
    if (!brgemm_available()) return nullptr;

    const uint16_t* w = (const uint16_t*)weight_ptr;
    int64_t block_n = BRGEMM_BLOCK_N;
    int64_t n_blocks = (n + block_n - 1) / block_n;
    int64_t vnni = 2;  // BF16 VNNI factor (pack32)

    // Each block: [K/vnni, block_n, vnni] = K * block_n BF16 elements
    size_t block_elems = (size_t)(k * block_n);
    size_t total_bytes = n_blocks * block_elems * 2;

    uint16_t* packed = (uint16_t*)aligned_alloc(64, total_bytes);
    if (!packed) return nullptr;
    memset(packed, 0, total_bytes);

    // Pack each N-block into VNNI format
    // weight[N, K] row-major → packed[NB, K/2, BLOCK_N, 2]
    for (int64_t nb = 0; nb < n_blocks; nb++) {
        uint16_t* blk = packed + nb * block_elems;
        int64_t nc = nb * block_n;
        int64_t actual_n = (nc + block_n <= n) ? block_n : (n - nc);

        for (int64_t kp = 0; kp < k / vnni; kp++) {
            for (int64_t j = 0; j < actual_n; j++) {
                for (int64_t d = 0; d < vnni; d++) {
                    blk[kp * block_n * vnni + j * vnni + d] =
                        w[(nc + j) * k + kp * vnni + d];
                }
            }
        }
        // Tail K element if K is odd
        if (k % vnni != 0) {
            int64_t kp = k / vnni;
            for (int64_t j = 0; j < actual_n; j++) {
                blk[kp * block_n * vnni + j * vnni] =
                    w[(nc + j) * k + kp * vnni];
            }
        }
    }

    auto* pw = new brgemm_packed_b();
    pw->data = packed;
    pw->k = k;
    pw->n = n;
    pw->block_n = block_n;
    pw->n_blocks = n_blocks;
    return pw;
}

void brgemm_bf16_pack_destroy(brgemm_packed_b_t pw) {
    if (!pw) return;
    free(pw->data);
    delete pw;
}

static inline uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    bits = bits + 0x7FFFu + ((bits >> 16) & 1);
    return (uint16_t)(bits >> 16);
}

static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

void brgemm_bf16_linear(
    const void* input,
    brgemm_packed_b_t pw,
    void* output,
    int64_t m,
    int64_t n_total,
    int64_t n_start, int64_t n_end)
{
    if (m <= 0 || n_start >= n_end || !pw) return;

    const uint16_t* A = (const uint16_t*)input;
    uint16_t* out = (uint16_t*)output;
    const uint16_t* packed = (const uint16_t*)pw->data;
    int64_t k = pw->k;
    int64_t block_n = pw->block_n;
    int64_t block_elems = k * block_n;

    // Thread-local accumulator (grows-only, no per-call heap alloc)
    static thread_local std::vector<float> tls_acc;
    size_t acc_needed = (size_t)(m * BRGEMM_BLOCK_N);
    if (tls_acc.size() < acc_needed) tls_acc.resize(acc_needed);
    float* acc = tls_acc.data();
    int64_t offsets[2] = {0, 0};

    int64_t nb_start = n_start / block_n;
    int64_t nb_end = (n_end + block_n - 1) / block_n;

    // Process all N-blocks in this thread's range.
    // AMX tile config is set once (on first block or kernel change)
    // and released once at the end — NOT per-block.
    for (int64_t nb = nb_start; nb < nb_end; nb++) {
        int64_t nc = nb * block_n;
        int64_t actual_n = (nc + block_n <= pw->n) ? block_n : (pw->n - nc);

        BrgemmKernel& kern = get_brgemm_kernel(m, actual_n, k);
        if (&kern != tls_current) {
            dnnl_brgemm_set_hw_context(kern.brgemm);
            tls_current = &kern;
        }

        const void* B = packed + nb * block_elems;
        dnnl_brgemm_execute(kern.brgemm, A, B, offsets,
            acc, kern.scratchpad.data());

        // Convert F32 → BF16 and write to output
        for (int64_t r = 0; r < m; r++) {
            for (int64_t c = 0; c < actual_n; c++) {
                out[r * n_total + nc + c] = f32_to_bf16(acc[r * actual_n + c]);
            }
        }
    }

    // Release AMX tile config once after all blocks processed
    dnnl_brgemm_release_hw_context();
    tls_current = nullptr;
}

static inline float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

void brgemm_bf16_linear_fused_silu_mul(
    const void* input,
    brgemm_packed_b_t pw,
    void* output,
    int64_t m,
    int64_t dim,
    int64_t n_start, int64_t n_end)
{
    if (m <= 0 || n_start >= n_end || !pw) return;

    const uint16_t* A = (const uint16_t*)input;
    uint16_t* out = (uint16_t*)output;
    const uint16_t* packed = (const uint16_t*)pw->data;
    int64_t k = pw->k;
    int64_t block_n = pw->block_n;
    int64_t block_elems = k * block_n;

    // Two accumulators: one for gate columns, one for up columns
    static thread_local std::vector<float> tls_gate_acc;
    static thread_local std::vector<float> tls_up_acc;
    size_t acc_needed = (size_t)(m * BRGEMM_BLOCK_N);
    if (tls_gate_acc.size() < acc_needed) tls_gate_acc.resize(acc_needed);
    if (tls_up_acc.size() < acc_needed) tls_up_acc.resize(acc_needed);
    float* gate_acc = tls_gate_acc.data();
    float* up_acc = tls_up_acc.data();
    int64_t offsets[2] = {0, 0};

    int64_t nb_start = n_start / block_n;
    int64_t nb_end = (n_end + block_n - 1) / block_n;

    for (int64_t nb = nb_start; nb < nb_end; nb++) {
        int64_t nc = nb * block_n;
        int64_t actual_n = (nc + block_n <= dim) ? block_n : (dim - nc);
        if (actual_n <= 0) break;

        BrgemmKernel& kern = get_brgemm_kernel(m, actual_n, k);
        if (&kern != tls_current) {
            dnnl_brgemm_set_hw_context(kern.brgemm);
            tls_current = &kern;
        }

        // Compute gate block: columns [nc, nc+actual_n) from first half
        const void* B_gate = packed + nb * block_elems;
        dnnl_brgemm_execute(kern.brgemm, A, B_gate, offsets,
            gate_acc, kern.scratchpad.data());

        // Compute corresponding up block: columns [nc+dim, nc+dim+actual_n) from second half
        int64_t up_nb = (nc + dim) / block_n;
        const void* B_up = packed + up_nb * block_elems;
        dnnl_brgemm_execute(kern.brgemm, A, B_up, offsets,
            up_acc, kern.scratchpad.data());

        // Fused SiLU(gate) * up → BF16 output (data hot in L1/L2)
        for (int64_t r = 0; r < m; r++) {
            for (int64_t c = 0; c < actual_n; c++) {
                float g = gate_acc[r * actual_n + c];
                float u = up_acc[r * actual_n + c];
                out[r * dim + nc + c] = f32_to_bf16(silu_f32(g) * u);
            }
        }
    }

    dnnl_brgemm_release_hw_context();
    tls_current = nullptr;
}

// ── AVX-512 VNNI packing helpers ─────────────────────────────────────

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>

// Transpose 2×32 BF16 values: interleave pairs for VNNI format.
// Input:  row0 = [a0..a31], row1 = [b0..b31]
// Output: lo = [a0,b0,a1,b1,...,a15,b15], hi = [a16,b16,...,a31,b31]
static inline void transpose_2x32_16bit(__m512i r0, __m512i r1,
                                         __m512i& lo, __m512i& hi) {
    __m512i t0 = _mm512_unpacklo_epi16(r0, r1);  // interleave low 16-bit
    __m512i t1 = _mm512_unpackhi_epi16(r0, r1);  // interleave high 16-bit
    // Shuffle 64-bit lanes to get final order
    lo = _mm512_permutex2var_epi64(t0, _mm512_setr_epi64(0,1,8,9,2,3,10,11), t1);
    hi = _mm512_permutex2var_epi64(t0, _mm512_setr_epi64(4,5,12,13,6,7,14,15), t1);
}

// Pack [K, N] row-major BF16 to VNNI [K/2, N, 2] format using AVX-512.
// Processes 32 columns at a time. Handles K padding (zero-fill odd K).
// This is the hot path for attention V packing and K^T packing.
static void pack_vnni_avx512(
    uint16_t* __restrict__ dst,
    const uint16_t* __restrict__ src,
    int64_t K, int64_t N, int64_t ld_src, int64_t ld_dst)
{
    int64_t padded_K = (K + 1) & ~1;  // round up to even
    int64_t NB = N / 32;
    int64_t N_rem = N - NB * 32;

    for (int64_t kp = 0; kp < padded_K / 2; kp++) {
        int64_t k0 = kp * 2;
        int64_t k1 = kp * 2 + 1;

        for (int64_t nb = 0; nb < NB; nb++) {
            __m512i r0 = (k0 < K)
                ? _mm512_loadu_si512(src + k0 * ld_src + nb * 32)
                : _mm512_setzero_si512();
            __m512i r1 = (k1 < K)
                ? _mm512_loadu_si512(src + k1 * ld_src + nb * 32)
                : _mm512_setzero_si512();
            __m512i lo, hi;
            transpose_2x32_16bit(r0, r1, lo, hi);
            _mm512_storeu_si512(dst + kp * ld_dst * 2 + nb * 64, lo);
            _mm512_storeu_si512(dst + kp * ld_dst * 2 + nb * 64 + 32, hi);
        }
        // Remainder columns (scalar)
        if (N_rem > 0) {
            for (int64_t j = NB * 32; j < N; j++) {
                uint16_t v0 = (k0 < K) ? src[k0 * ld_src + j] : 0;
                uint16_t v1 = (k1 < K) ? src[k1 * ld_src + j] : 0;
                dst[kp * ld_dst * 2 + j * 2 + 0] = v0;
                dst[kp * ld_dst * 2 + j * 2 + 1] = v1;
            }
        }
    }
}

// Vectorized F32 → BF16 conversion + zero-pad in one pass.
// Converts scores_f32[m × lda] (first k columns) to BF16 [m × padded_k], padding with 0.
static void convert_f32_bf16_pad_avx512(
    uint16_t* __restrict__ dst,
    const float* __restrict__ src,
    int64_t m, int64_t k, int64_t padded_k, int64_t lda)
{
    for (int64_t i = 0; i < m; i++) {
        const float* row_src = src + i * lda;
        uint16_t* row_dst = dst + i * padded_k;
        int64_t j = 0;
        // 16 floats → 16 BF16 per iteration
        for (; j + 16 <= k; j += 16) {
            __m512 v = _mm512_loadu_ps(row_src + j);
            __m256i bf = _mm512_cvtneps_pbh(v);
            _mm256_storeu_si256((__m256i*)(row_dst + j), bf);
        }
        // Scalar remainder
        for (; j < k; j++) {
            row_dst[j] = f32_to_bf16(row_src[j]);
        }
        // Zero-pad
        if (padded_k > k) {
            memset(row_dst + k, 0, (padded_k - k) * sizeof(uint16_t));
        }
    }
}
// 16×16 transpose of 32-bit elements in 16 zmm registers.
// Standard 4-stage algorithm: unpack32 → unpack64 → shuffle128 → shuffle256.
static inline void transpose_16x16_epi32(__m512i r[16]) {
    __m512i a[16];
    for (int i = 0; i < 16; i += 2) {
        a[i]   = _mm512_unpacklo_epi32(r[i], r[i+1]);
        a[i+1] = _mm512_unpackhi_epi32(r[i], r[i+1]);
    }
    __m512i b[16];
    for (int i = 0; i < 16; i += 4) {
        b[i]   = _mm512_unpacklo_epi64(a[i],   a[i+2]);
        b[i+1] = _mm512_unpacklo_epi64(a[i+1], a[i+3]);
        b[i+2] = _mm512_unpackhi_epi64(a[i],   a[i+2]);
        b[i+3] = _mm512_unpackhi_epi64(a[i+1], a[i+3]);
    }
    __m512i c[16];
    for (int i = 0; i < 16; i += 8) {
        c[i]   = _mm512_shuffle_i32x4(b[i],   b[i+4], 0x88);
        c[i+1] = _mm512_shuffle_i32x4(b[i+1], b[i+5], 0x88);
        c[i+2] = _mm512_shuffle_i32x4(b[i+2], b[i+6], 0x88);
        c[i+3] = _mm512_shuffle_i32x4(b[i+3], b[i+7], 0x88);
        c[i+4] = _mm512_shuffle_i32x4(b[i],   b[i+4], 0xDD);
        c[i+5] = _mm512_shuffle_i32x4(b[i+1], b[i+5], 0xDD);
        c[i+6] = _mm512_shuffle_i32x4(b[i+2], b[i+6], 0xDD);
        c[i+7] = _mm512_shuffle_i32x4(b[i+3], b[i+7], 0xDD);
    }
    r[0]  = _mm512_shuffle_i32x4(c[0],  c[8],  0x88);
    r[1]  = _mm512_shuffle_i32x4(c[1],  c[9],  0x88);
    r[2]  = _mm512_shuffle_i32x4(c[2],  c[10], 0x88);
    r[3]  = _mm512_shuffle_i32x4(c[3],  c[11], 0x88);
    r[4]  = _mm512_shuffle_i32x4(c[0],  c[8],  0xDD);
    r[5]  = _mm512_shuffle_i32x4(c[1],  c[9],  0xDD);
    r[6]  = _mm512_shuffle_i32x4(c[2],  c[10], 0xDD);
    r[7]  = _mm512_shuffle_i32x4(c[3],  c[11], 0xDD);
    r[8]  = _mm512_shuffle_i32x4(c[4],  c[12], 0x88);
    r[9]  = _mm512_shuffle_i32x4(c[5],  c[13], 0x88);
    r[10] = _mm512_shuffle_i32x4(c[6],  c[14], 0x88);
    r[11] = _mm512_shuffle_i32x4(c[7],  c[15], 0x88);
    r[12] = _mm512_shuffle_i32x4(c[4],  c[12], 0xDD);
    r[13] = _mm512_shuffle_i32x4(c[5],  c[13], 0xDD);
    r[14] = _mm512_shuffle_i32x4(c[6],  c[14], 0xDD);
    r[15] = _mm512_shuffle_i32x4(c[7],  c[15], 0xDD);
}

// Pack K^T to VNNI using load+transpose (cache-friendly).
// K[n rows, stride k_stride, head_dim cols] → K^T_vnni[head_dim/2, n, 2]
// Processes 16-row × 32-column blocks: each row loads a full cache line (64B).
static void pack_kt_vnni_transpose(
    uint16_t* __restrict__ dst,
    const uint16_t* __restrict__ K_bf16,
    int64_t n, int64_t head_dim, int64_t k_stride, int64_t ld_dst)
{
    int64_t KB = head_dim / 32;
    int64_t K_rem = head_dim - KB * 32;

    for (int64_t nb = 0; nb < n; nb += 16) {
        int64_t n_size = std::min((int64_t)16, n - nb);

        for (int64_t kb = 0; kb < KB; kb++) {
            int64_t d_start = kb * 32;
            __m512i regs[16];

            // Load n_size rows of 32 BF16 each (= 16 × i32 VNNI pairs)
            for (int64_t j = 0; j < n_size; j++) {
                regs[j] = _mm512_loadu_si512(
                    K_bf16 + (nb + j) * k_stride + d_start);
            }
            for (int64_t j = n_size; j < 16; j++) {
                regs[j] = _mm512_setzero_si512();
            }

            // 16×16 32-bit transpose: regs[k] now has 16 VNNI pairs
            // from 16 rows at head_dim position (d_start + 2k, d_start + 2k + 1)
            transpose_16x16_epi32(regs);

            // Store with mask for partial row blocks
            __mmask16 mask = (n_size >= 16) ? (__mmask16)0xFFFF
                : (__mmask16)((1u << n_size) - 1);
            for (int64_t k = 0; k < 16; k++) {
                int64_t kp = d_start / 2 + k;
                _mm512_mask_storeu_epi32(
                    dst + kp * ld_dst * 2 + nb * 2, mask, regs[k]);
            }
        }

        // Remainder columns (head_dim % 32 != 0)
        if (K_rem > 0) {
            int64_t d_start = KB * 32;
            int64_t K2 = K_rem / 2;
            __mmask16 load_mask = (K2 >= 16) ? (__mmask16)0xFFFF
                : (__mmask16)((1u << K2) - 1);
            __m512i regs[16];
            for (int64_t j = 0; j < n_size; j++) {
                regs[j] = _mm512_maskz_loadu_epi32(
                    load_mask, K_bf16 + (nb + j) * k_stride + d_start);
            }
            for (int64_t j = n_size; j < 16; j++) {
                regs[j] = _mm512_setzero_si512();
            }
            transpose_16x16_epi32(regs);
            __mmask16 store_mask = (n_size >= 16) ? (__mmask16)0xFFFF
                : (__mmask16)((1u << n_size) - 1);
            for (int64_t k = 0; k < K2; k++) {
                int64_t kp = d_start / 2 + k;
                _mm512_mask_storeu_epi32(
                    dst + kp * ld_dst * 2 + nb * 2, store_mask, regs[k]);
            }
        }
    }
}
#endif // __AVX512F__ && __AVX512BW__

// Scalar fallback for VNNI packing (non-AVX512 or remainder)
static void pack_vnni_scalar(
    uint16_t* __restrict__ dst,
    const uint16_t* __restrict__ src,
    int64_t K, int64_t N, int64_t ld_src, int64_t ld_dst)
{
    int64_t padded_K = (K + 1) & ~1;
    for (int64_t kp = 0; kp < padded_K / 2; kp++) {
        int64_t k0 = kp * 2, k1 = kp * 2 + 1;
        for (int64_t j = 0; j < N; j++) {
            dst[kp * ld_dst * 2 + j * 2 + 0] = (k0 < K) ? src[k0 * ld_src + j] : 0;
            dst[kp * ld_dst * 2 + j * 2 + 1] = (k1 < K) ? src[k1 * ld_src + j] : 0;
        }
    }
}

// Dispatch to AVX-512 or scalar VNNI packing
static inline void pack_vnni(
    uint16_t* dst, const uint16_t* src,
    int64_t K, int64_t N, int64_t ld_src, int64_t ld_dst)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    pack_vnni_avx512(dst, src, K, N, ld_src, ld_dst);
#else
    pack_vnni_scalar(dst, src, K, N, ld_src, ld_dst);
#endif
}

// ── Score @ V accumulation for attention ──────────────────────────────
//
// Computes C += A @ B where:
//   A: BF16 [m × k] row-major (softmax scores, F32→BF16 converted + padded)
//   B: BF16 [k × n] VNNI-packed (V matrix, on-the-fly packed)
//   C: F32 [m × n] accumulator (add to existing values)
//
// Uses a separate kernel cache with add_C=1 (beta=1).

static thread_local std::unordered_map<BrgemmKey, BrgemmKernel, BrgemmKeyHash> tls_brgemm_addc;

static BrgemmKernel& get_brgemm_kernel_addc(int64_t m, int64_t n, int64_t k) {
    BrgemmKey key{m, n, k};
    auto it = tls_brgemm_addc.find(key);
    if (it != tls_brgemm_addc.end()) return it->second;

    BrgemmKernel kern{};
    CHECK_DNNL(dnnl_brgemm_create(&kern.brgemm, m, n, k,
        /*batch_size*/1, /*lda*/k, /*ldb*/n, /*ldc*/n,
        dnnl_bf16, dnnl_bf16, dnnl_f32));
    CHECK_DNNL(dnnl_brgemm_set_add_C(kern.brgemm, 1));  // beta=1: C += A @ B
    CHECK_DNNL(dnnl_brgemm_finalize(kern.brgemm));
    CHECK_DNNL(dnnl_brgemm_generate(kern.brgemm));

    size_t sz = 0;
    CHECK_DNNL(dnnl_brgemm_get_scratchpad_size(kern.brgemm, &sz));
    kern.scratchpad.resize(sz);

    auto [inserted, _] = tls_brgemm_addc.emplace(key, std::move(kern));
    return inserted->second;
}

// Score @ V accumulation: C_f32 += scores_bf16 @ V_vnni
//
// scores_f32: [m × lda] F32 softmax weights (only first k columns used)
// V_bf16:     [k × n] BF16 row-major V data
// C_f32:      [m × n] F32 accumulator (add in-place)
// m: number of query rows
// k: number of key rows (actual, before padding)
// n: head_dim (V column dimension)
// lda: leading dimension of scores (= block_n from caller)
void brgemm_score_v_accum(
    const float* scores_f32,
    const uint16_t* V_bf16,
    float* C_f32,
    int64_t m, int64_t k, int64_t n, int64_t lda,
    int64_t v_stride)
{
    if (m <= 0 || k <= 0 || n <= 0) return;

    // Pad k to TILE_K boundary (32 elements for AMX BF16)
    const int64_t TILE_K = 32;
    int64_t padded_k = (k + TILE_K - 1) / TILE_K * TILE_K;

    // Thread-local buffers (grow-only, no per-call alloc)
    static thread_local std::vector<uint16_t> tls_scores_bf16;
    static thread_local std::vector<uint16_t> tls_v_vnni;

    size_t scores_sz = (size_t)(m * padded_k);
    size_t v_vnni_sz = (size_t)(padded_k * n);
    if (tls_scores_bf16.size() < scores_sz) tls_scores_bf16.resize(scores_sz);
    if (tls_v_vnni.size() < v_vnni_sz) tls_v_vnni.resize(v_vnni_sz);

    // Convert F32 scores → BF16 + pad (AVX-512 vectorized)
#if defined(__AVX512F__) && defined(__AVX512BW__)
    convert_f32_bf16_pad_avx512(tls_scores_bf16.data(), scores_f32, m, k, padded_k, lda);
#else
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < k; j++)
            tls_scores_bf16[i * padded_k + j] = f32_to_bf16(scores_f32[i * lda + j]);
        for (int64_t j = k; j < padded_k; j++)
            tls_scores_bf16[i * padded_k + j] = 0;
    }
#endif

    // Pack V to VNNI: [k rows, stride v_stride, n cols] → [k/2, n, 2]
    pack_vnni(tls_v_vnni.data(), V_bf16, k, n, v_stride, n);

    // Use beta=1 kernel (add_C=1) to accumulate directly into C_f32,
    // eliminating temp buffer + manual addition loop.
    BrgemmKernel& kern = get_brgemm_kernel_addc(m, n, padded_k);
    int64_t offsets[2] = {0, 0};

    if (&kern != tls_current) {
        dnnl_brgemm_set_hw_context(kern.brgemm);
        tls_current = &kern;
    }
    dnnl_brgemm_execute(kern.brgemm,
        tls_scores_bf16.data(), tls_v_vnni.data(),
        offsets, C_f32, kern.scratchpad.data());
}

// ── QK^T GEMM for attention ──────────────────────────────────────────
//
// Computes scores = Q @ K^T * sm_scale where:
//   Q: BF16 [m × head_dim] with row stride q_stride
//   K: BF16 [n × head_dim] contiguous (gathered)
//   scores: F32 [m × ldc] (overwrite, beta=0)
//
void brgemm_qk_gemm(
    const uint16_t* Q_bf16,
    const uint16_t* K_bf16,
    float* scores_f32,
    int64_t m, int64_t n, int64_t head_dim,
    int64_t q_stride,
    int64_t k_stride,
    int64_t ldc,
    float sm_scale)
{
    if (m <= 0 || n <= 0 || head_dim <= 0) return;

    // Pad head_dim to TILE_K=32 for AMX alignment
    const int64_t TILE_K_PAD = 32;
    int64_t padded_d = (head_dim + TILE_K_PAD - 1) / TILE_K_PAD * TILE_K_PAD;

    // Thread-local buffers
    static thread_local std::vector<uint16_t> tls_q_gather;
    static thread_local std::vector<uint16_t> tls_kt_vnni;

    size_t q_sz = (size_t)(m * padded_d);
    size_t vnni_sz = (size_t)(padded_d * n);
    if (tls_q_gather.size() < q_sz) tls_q_gather.resize(q_sz, 0);
    if (tls_kt_vnni.size() < vnni_sz) tls_kt_vnni.resize(vnni_sz, 0);

    // Gather Q to contiguous [m × padded_d] only if strided or needs padding
    const uint16_t* q_data;
    if (q_stride == head_dim && padded_d == head_dim) {
        // Already contiguous and no padding needed — skip gather
        q_data = Q_bf16;
    } else {
        for (int64_t i = 0; i < m; i++) {
            const uint16_t* src = Q_bf16 + i * q_stride;
            uint16_t* dst = tls_q_gather.data() + i * padded_d;
            memcpy(dst, src, head_dim * sizeof(uint16_t));
            if (padded_d > head_dim)
                memset(dst + head_dim, 0, (padded_d - head_dim) * sizeof(uint16_t));
        }
        q_data = tls_q_gather.data();
    }

    // Pack K^T to VNNI: K[n rows, stride k_stride] → K^T_vnni[d/2, n, 2]
    // Uses 16×16 32-bit transpose (load full cache lines per row, 100% utilization)
    // instead of gather (which wastes 93% of each cache line).
#if defined(__AVX512F__) && defined(__AVX512BW__)
    pack_kt_vnni_transpose(tls_kt_vnni.data(), K_bf16, n, head_dim, k_stride, n);
    // Zero-pad if padded_d > head_dim
    if (padded_d > head_dim) {
        for (int64_t kp = head_dim / 2; kp < padded_d / 2; kp++) {
            memset(tls_kt_vnni.data() + kp * n * 2, 0, n * 2 * sizeof(uint16_t));
        }
    }
#else
    {
        int64_t padded_K = padded_d;
        for (int64_t kp = 0; kp < padded_K / 2; kp++) {
            int64_t d0 = kp * 2, d1 = kp * 2 + 1;
            for (int64_t j = 0; j < n; j++) {
                tls_kt_vnni[kp * n * 2 + j * 2 + 0] = (d0 < head_dim) ? K_bf16[j * k_stride + d0] : 0;
                tls_kt_vnni[kp * n * 2 + j * 2 + 1] = (d1 < head_dim) ? K_bf16[j * k_stride + d1] : 0;
            }
        }
    }
#endif

    // brgemm: scores[m × n] = Q[m × d] @ K^T_vnni[d × n]
    // Use lda=padded_d, ldb=n, ldc=ldc
    // Need a kernel with these specific dimensions
    BrgemmKernel& kern = get_brgemm_kernel(m, n, padded_d);

    int64_t offsets[2] = {0, 0};
    if (&kern != tls_current) {
        dnnl_brgemm_set_hw_context(kern.brgemm);
        tls_current = &kern;
    }

    // If ldc == n, write directly to scores; otherwise use temp buffer
    if (ldc == n) {
        dnnl_brgemm_execute(kern.brgemm,
            q_data, tls_kt_vnni.data(),
            offsets, scores_f32, kern.scratchpad.data());
    } else {
        static thread_local std::vector<float> tls_scores_tmp;
        if (tls_scores_tmp.size() < (size_t)(m * n))
            tls_scores_tmp.resize(m * n);
        dnnl_brgemm_execute(kern.brgemm,
            q_data, tls_kt_vnni.data(),
            offsets, tls_scores_tmp.data(), kern.scratchpad.data());
        for (int64_t i = 0; i < m; i++)
            memcpy(scores_f32 + i * ldc, tls_scores_tmp.data() + i * n, n * sizeof(float));
    }

    // sm_scale is applied in the caller's softmax (fused with exp computation)
    (void)sm_scale;
}

void brgemm_attn_release() {
    dnnl_brgemm_release_hw_context();
    tls_current = nullptr;
}

// ── INT8 BRGeMM (W8A8) ─────────────────────────────────────────────────
// U8 × S8 → F32 accumulation → compensation → scale → BF16 output.
//
// The brgemm ukernel API does NOT handle s8→u8 compensation internally
// (see upstream: brgemm_desc_.req_s8s8_compensation = false, with comment
// "Users must add compensation on their own as a binary post-op").
//
// AVX-512 VNNI only has VPDPBUSD (u8×s8). For W8A8 we:
//   1. Quantize BF16 input to u8 (= clamp(round(x/scale)) + 128)
//   2. Run brgemm u8×s8 → F32 (native VPDPBUSD, no compensation needed inside)
//   3. Subtract compensation: C -= 128 × Σ_k(weight_s8[j,k]) per column
//   4. Apply scales: D = C × a_scale × b_scale[j] → BF16
//
// Steps 3-4 are fused via post-ops: binary_add(comp) + A/B scales + BF16 output.

struct brgemm_s8_packed_b {
    void* data;           // packed weight buffer, 64-byte aligned
    float* scales;        // per-channel F32 scales [N]
    float* col_comp;      // per-column compensation: -128 * Σ_k(w_s8[j,k]) as F32 [N]
    int64_t k, n;
    int64_t block_n;
    int64_t n_blocks;
};

// Thread-local INT8 brgemm kernel cache (u8×s8 with compensation post-op)
struct BrgemmS8Key {
    int64_t m, n, k, ldd;
    bool operator==(const BrgemmS8Key& o) const {
        return m==o.m && n==o.n && k==o.k && ldd==o.ldd;
    }
};
struct BrgemmS8KeyHash {
    size_t operator()(const BrgemmS8Key& key) const {
        return (size_t)(key.m * 1000003 + key.n * 1009 + key.k * 31 + key.ldd);
    }
};
struct BrgemmS8Kernel {
    dnnl_brgemm_t brgemm = nullptr;
    std::vector<uint8_t> scratchpad;
};
static thread_local std::unordered_map<BrgemmS8Key, BrgemmS8Kernel, BrgemmS8KeyHash> tls_brgemm_s8;

static BrgemmS8Kernel& get_brgemm_s8_kernel(int64_t m, int64_t n, int64_t k, int64_t ldd) {
    BrgemmS8Key key{m, n, k, ldd};
    auto it = tls_brgemm_s8.find(key);
    if (it != tls_brgemm_s8.end()) return it->second;

    BrgemmS8Kernel kern{};
    // u8×s8 → F32, with A/B scales and binary_add post-op for compensation.
    //
    // Execution order in execute_postops:
    //   1. C = GEMM(u8_A, s8_B)           — native VPDPBUSD
    //   2. C *= a_scale * b_scale[j]       — dequantization
    //   3. D = C + comp_scaled[j]          — s8→u8 compensation (binary_add)
    //   4. Convert D → BF16               — set via d_dt
    //
    // comp_scaled[j] = -128 * Σ_k(w[j,k]) * a_scale * b_scale[j]
    // Precomputed at runtime since a_scale is dynamic.
    CHECK_DNNL(dnnl_brgemm_create(&kern.brgemm, m, n, k,
        /*batch_size*/1, /*lda*/k, /*ldb*/n, /*ldc*/n,
        dnnl_u8, dnnl_s8, dnnl_f32));
    CHECK_DNNL(dnnl_brgemm_set_add_C(kern.brgemm, 0));
    CHECK_DNNL(dnnl_brgemm_set_A_scales(kern.brgemm, 0));  // per-tensor
    CHECK_DNNL(dnnl_brgemm_set_B_scales(kern.brgemm, 2));  // per-channel

    // Binary_add post-op for compensation [1, N] F32
    dnnl_post_ops_t post_ops = nullptr;
    CHECK_DNNL(dnnl_post_ops_create(&post_ops));
    dnnl_dims_t comp_dims = {1, n};
    dnnl_memory_desc_t comp_md = nullptr;
    CHECK_DNNL(dnnl_memory_desc_create_with_tag(&comp_md, 2, comp_dims, dnnl_f32, dnnl_ab));
    CHECK_DNNL(dnnl_post_ops_append_binary(post_ops, dnnl_binary_add, comp_md));
    dnnl_memory_desc_destroy(comp_md);

    CHECK_DNNL(dnnl_brgemm_set_post_ops(kern.brgemm, ldd, dnnl_bf16, post_ops));
    dnnl_post_ops_destroy(post_ops);

    CHECK_DNNL(dnnl_brgemm_finalize(kern.brgemm));
    CHECK_DNNL(dnnl_brgemm_generate(kern.brgemm));

    size_t sz = 0;
    CHECK_DNNL(dnnl_brgemm_get_scratchpad_size(kern.brgemm, &sz));
    kern.scratchpad.resize(sz);

    auto [inserted, _] = tls_brgemm_s8.emplace(key, std::move(kern));
    return inserted->second;
}

int brgemm_s8_available(void) {
    static int cached = -1;
    if (cached >= 0) return cached;

    dnnl_brgemm_t brg = nullptr;
    dnnl_status_t s = dnnl_brgemm_create(&brg, 1, 32, 64,
        1, 64, 32, 32, dnnl_u8, dnnl_s8, dnnl_f32);
    if (s == dnnl_success && brg) {
        s = dnnl_brgemm_finalize(brg);
        if (s == dnnl_success) s = dnnl_brgemm_generate(brg);
        dnnl_brgemm_destroy(brg);
        cached = (s == dnnl_success) ? 1 : 0;
    } else {
        if (brg) dnnl_brgemm_destroy(brg);
        cached = 0;
    }
    return cached;
}

brgemm_s8_packed_b_t brgemm_s8_pack(
    const int8_t* weight, const float* scales, int64_t k, int64_t n)
{
    if (!brgemm_s8_available()) return nullptr;

    int64_t block_n = BRGEMM_BLOCK_N;
    int64_t n_blocks = (n + block_n - 1) / block_n;
    int64_t vnni = 4;  // INT8 VNNI factor: 4 bytes = 32 bits

    size_t block_elems = (size_t)(k * block_n);
    size_t total_bytes = n_blocks * block_elems;

    int8_t* packed = (int8_t*)aligned_alloc(64, total_bytes);
    if (!packed) return nullptr;
    memset(packed, 0, total_bytes);

    // Pack: weight[N, K] row-major → packed[NB, K/4, BLOCK_N, 4]
    for (int64_t nb = 0; nb < n_blocks; nb++) {
        int8_t* blk = packed + nb * block_elems;
        int64_t nc = nb * block_n;
        int64_t actual_n = (nc + block_n <= n) ? block_n : (n - nc);

        int64_t k_groups = k / vnni;
        for (int64_t kp = 0; kp < k_groups; kp++) {
            for (int64_t j = 0; j < actual_n; j++) {
                for (int64_t d = 0; d < vnni; d++) {
                    blk[kp * block_n * vnni + j * vnni + d] =
                        weight[(nc + j) * k + kp * vnni + d];
                }
            }
        }
        int64_t k_tail = k % vnni;
        if (k_tail > 0) {
            for (int64_t j = 0; j < actual_n; j++) {
                for (int64_t d = 0; d < k_tail; d++) {
                    blk[k_groups * block_n * vnni + j * vnni + d] =
                        weight[(nc + j) * k + k_groups * vnni + d];
                }
            }
        }
    }

    float* scales_copy = (float*)malloc(n * sizeof(float));
    memcpy(scales_copy, scales, n * sizeof(float));

    // Precompute per-column compensation: -128 * Σ_k(weight_s8[j,k])
    float* col_comp = (float*)malloc(n * sizeof(float));
    for (int64_t j = 0; j < n; j++) {
        int64_t sum = 0;
        for (int64_t c = 0; c < k; c++) {
            sum += weight[j * k + c];
        }
        col_comp[j] = -128.0f * (float)sum;
    }

    auto* pw = new brgemm_s8_packed_b();
    pw->data = packed;
    pw->scales = scales_copy;
    pw->col_comp = col_comp;
    pw->k = k;
    pw->n = n;
    pw->block_n = block_n;
    pw->n_blocks = n_blocks;
    return pw;
}

void brgemm_s8_pack_destroy(brgemm_s8_packed_b_t pw) {
    if (!pw) return;
    free(pw->data);
    free(pw->scales);
    free(pw->col_comp);
    delete pw;
}

float brgemm_quantize_bf16_s8(
    const void* input_bf16, int8_t* out_s8, int64_t m, int64_t k)
{
    const uint16_t* in = (const uint16_t*)input_bf16;
    int64_t n = m * k;

    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float abs_v = fabsf(bf16_to_f32(in[i]));
        if (abs_v > max_abs) max_abs = abs_v;
    }

    float scale = max_abs / 127.0f;
    if (max_abs == 0.0f) {
        // Output u8 zero-point = 128
        memset(out_s8, (int8_t)(uint8_t)128, n);
        return scale;
    }

    float inv_scale = 127.0f / max_abs;
    for (int64_t i = 0; i < n; i++) {
        float v = bf16_to_f32(in[i]);
        int32_t q = (int32_t)roundf(v * inv_scale);
        q = std::max(-128, std::min(127, q));
        // Store as u8 = s8 + 128 (reinterpreted as int8_t for FFI convenience)
        out_s8[i] = (int8_t)(uint8_t)(q + 128);
    }
    return scale;
}

void brgemm_s8_linear(
    const int8_t* input_s8, float a_scale,
    brgemm_s8_packed_b_t pw,
    void* output_bf16,
    int64_t m, int64_t n_total,
    int64_t n_start, int64_t n_end)
{
    if (m <= 0 || n_start >= n_end || !pw) return;

    uint16_t* out = (uint16_t*)output_bf16;
    const int8_t* packed = (const int8_t*)pw->data;
    int64_t k = pw->k;
    int64_t block_n = pw->block_n;
    int64_t block_elems = k * block_n;

    static thread_local std::vector<float> tls_s8_acc;
    size_t acc_needed = (size_t)(m * BRGEMM_BLOCK_N);
    if (tls_s8_acc.size() < acc_needed) tls_s8_acc.resize(acc_needed);
    float* acc = tls_s8_acc.data();
    int64_t offsets[2] = {0, 0};

    int64_t nb_start = n_start / block_n;
    int64_t nb_end = (n_end + block_n - 1) / block_n;

    // Thread-local buffer for runtime-scaled compensation
    static thread_local std::vector<float> tls_comp_scaled;
    if (tls_comp_scaled.size() < (size_t)BRGEMM_BLOCK_N)
        tls_comp_scaled.resize(BRGEMM_BLOCK_N);

    dnnl_ukernel_attr_params_t params = nullptr;
    CHECK_DNNL(dnnl_ukernel_attr_params_create(&params));
    CHECK_DNNL(dnnl_ukernel_attr_params_set_A_scales(params, &a_scale));

    const BrgemmS8Kernel* current_kern = nullptr;

    for (int64_t nb = nb_start; nb < nb_end; nb++) {
        int64_t nc = nb * block_n;
        int64_t actual_n = (nc + block_n <= pw->n) ? block_n : (pw->n - nc);

        BrgemmS8Kernel& kern = get_brgemm_s8_kernel(m, actual_n, k, n_total);
        if (&kern != current_kern) {
            dnnl_brgemm_set_hw_context(kern.brgemm);
            current_kern = &kern;
        }

        const void* B = packed + nb * block_elems;

        CHECK_DNNL(dnnl_ukernel_attr_params_set_B_scales(params, pw->scales + nc));

        // Compute runtime-scaled compensation:
        // comp_scaled[j] = col_comp[nc+j] * a_scale * b_scale[nc+j]
        // where col_comp = -128 * Σ_k(w[j,k])
        for (int64_t j = 0; j < actual_n; j++) {
            tls_comp_scaled[j] = pw->col_comp[nc + j] * a_scale * pw->scales[nc + j];
        }
        const void* po_args[1] = { tls_comp_scaled.data() };
        CHECK_DNNL(dnnl_ukernel_attr_params_set_post_ops_args(params, po_args));

        uint16_t* D = out + nc;

        CHECK_DNNL(dnnl_brgemm_execute_postops(kern.brgemm,
            input_s8, B, offsets,
            acc, D, kern.scratchpad.data(), params));
    }

    dnnl_brgemm_release_hw_context();
    tls_current = nullptr;
    dnnl_ukernel_attr_params_destroy(params);
}

// ── FP8 BRGeMM ─────────────────────────────────────────────────────────
// FP8 (E4M3) × FP8 (E4M3) → F32 → BF16.
// Requires AVX10.2 AMX-2 hardware. Gracefully returns 0/NULL if unavailable.

struct brgemm_f8_packed_b {
    void* data;
    float* scales;
    int64_t k, n;
    int64_t block_n;
    int64_t n_blocks;
};

// Thread-local FP8 brgemm kernel cache
static thread_local std::unordered_map<BrgemmS8Key, BrgemmS8Kernel, BrgemmS8KeyHash> tls_brgemm_f8;

static BrgemmS8Kernel& get_brgemm_f8_kernel(int64_t m, int64_t n, int64_t k, int64_t ldd) {
    BrgemmS8Key key{m, n, k, ldd};
    auto it = tls_brgemm_f8.find(key);
    if (it != tls_brgemm_f8.end()) return it->second;

    BrgemmS8Kernel kern{};
    CHECK_DNNL(dnnl_brgemm_create(&kern.brgemm, m, n, k,
        1, k, n, n,
        dnnl_f8_e4m3, dnnl_f8_e4m3, dnnl_f32));
    CHECK_DNNL(dnnl_brgemm_set_add_C(kern.brgemm, 0));
    CHECK_DNNL(dnnl_brgemm_set_A_scales(kern.brgemm, 0));
    CHECK_DNNL(dnnl_brgemm_set_B_scales(kern.brgemm, 2));

    dnnl_post_ops_t post_ops = nullptr;
    CHECK_DNNL(dnnl_post_ops_create(&post_ops));
    CHECK_DNNL(dnnl_brgemm_set_post_ops(kern.brgemm, ldd, dnnl_bf16, post_ops));
    dnnl_post_ops_destroy(post_ops);

    CHECK_DNNL(dnnl_brgemm_finalize(kern.brgemm));
    CHECK_DNNL(dnnl_brgemm_generate(kern.brgemm));

    size_t sz = 0;
    CHECK_DNNL(dnnl_brgemm_get_scratchpad_size(kern.brgemm, &sz));
    kern.scratchpad.resize(sz);

    auto [inserted, _] = tls_brgemm_f8.emplace(key, std::move(kern));
    return inserted->second;
}

int brgemm_f8_available(void) {
    static int cached = -1;
    if (cached >= 0) return cached;

    dnnl_brgemm_t brg = nullptr;
    dnnl_status_t s = dnnl_brgemm_create(&brg, 1, 32, 64,
        1, 64, 32, 32, dnnl_f8_e4m3, dnnl_f8_e4m3, dnnl_f32);
    if (s == dnnl_success && brg) {
        s = dnnl_brgemm_finalize(brg);
        if (s == dnnl_success) s = dnnl_brgemm_generate(brg);
        dnnl_brgemm_destroy(brg);
        cached = (s == dnnl_success) ? 1 : 0;
    } else {
        if (brg) dnnl_brgemm_destroy(brg);
        cached = 0;
    }
    return cached;
}

// F8E4M3 conversion helpers
// E4M3 format: 1 sign, 4 exponent (bias=7), 3 mantissa. No inf, max=448.
static inline uint8_t f32_to_f8e4m3(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp32 = ((int32_t)((bits >> 23) & 0xFF)) - 127;
    uint32_t mant = bits & 0x7FFFFF;

    // Handle special cases
    if ((bits & 0x7FFFFFFF) == 0) return (uint8_t)(sign << 7); // ±0
    if (exp32 > 8) return (uint8_t)((sign << 7) | 0x7E);       // overflow → max

    int32_t e4 = exp32 + 7; // E4M3 bias = 7
    if (e4 <= 0) {
        // Subnormal: shift mantissa
        int shift = 1 - e4 + 20; // 20 = 23 - 3 mantissa bits
        uint32_t full_mant = (0x800000 | mant);
        uint32_t m3 = (shift < 32) ? (full_mant >> shift) : 0;
        return (uint8_t)((sign << 7) | (m3 & 0x7));
    }
    if (e4 > 15) return (uint8_t)((sign << 7) | 0x7E); // overflow

    // Round mantissa to 3 bits (round to nearest even)
    uint32_t round_bit = (mant >> 19) & 1;
    uint32_t sticky = mant & ((1 << 19) - 1);
    uint32_t m3 = (mant >> 20) & 0x7;
    if (round_bit && (sticky || (m3 & 1))) {
        m3++;
        if (m3 > 7) { m3 = 0; e4++; }
        if (e4 > 15) return (uint8_t)((sign << 7) | 0x7E);
    }
    return (uint8_t)((sign << 7) | ((e4 & 0xF) << 3) | (m3 & 0x7));
}

brgemm_f8_packed_b_t brgemm_f8e4m3_pack(
    const void* weight_f8, const float* scales, int64_t k, int64_t n)
{
    if (!brgemm_f8_available()) return nullptr;

    int64_t block_n = BRGEMM_BLOCK_N;
    int64_t n_blocks = (n + block_n - 1) / block_n;
    int64_t vnni = 4;  // FP8 VNNI factor: 4 bytes = 32 bits

    size_t block_elems = (size_t)(k * block_n);
    size_t total_bytes = n_blocks * block_elems;

    uint8_t* packed = (uint8_t*)aligned_alloc(64, total_bytes);
    if (!packed) return nullptr;
    memset(packed, 0, total_bytes);

    const uint8_t* w = (const uint8_t*)weight_f8;

    // Pack: weight[N, K] → packed[NB, K/4, BLOCK_N, 4]
    for (int64_t nb = 0; nb < n_blocks; nb++) {
        uint8_t* blk = packed + nb * block_elems;
        int64_t nc = nb * block_n;
        int64_t actual_n = (nc + block_n <= n) ? block_n : (n - nc);

        int64_t k_groups = k / vnni;
        for (int64_t kp = 0; kp < k_groups; kp++) {
            for (int64_t j = 0; j < actual_n; j++) {
                for (int64_t d = 0; d < vnni; d++) {
                    blk[kp * block_n * vnni + j * vnni + d] =
                        w[(nc + j) * k + kp * vnni + d];
                }
            }
        }
        int64_t k_tail = k % vnni;
        if (k_tail > 0) {
            for (int64_t j = 0; j < actual_n; j++) {
                for (int64_t d = 0; d < k_tail; d++) {
                    blk[k_groups * block_n * vnni + j * vnni + d] =
                        w[(nc + j) * k + k_groups * vnni + d];
                }
            }
        }
    }

    float* scales_copy = (float*)malloc(n * sizeof(float));
    memcpy(scales_copy, scales, n * sizeof(float));

    auto* pw = new brgemm_f8_packed_b();
    pw->data = packed;
    pw->scales = scales_copy;
    pw->k = k;
    pw->n = n;
    pw->block_n = block_n;
    pw->n_blocks = n_blocks;
    return pw;
}

void brgemm_f8_pack_destroy(brgemm_f8_packed_b_t pw) {
    if (!pw) return;
    free(pw->data);
    free(pw->scales);
    delete pw;
}

float brgemm_quantize_bf16_f8e4m3(
    const void* input_bf16, void* out_f8, int64_t m, int64_t k)
{
    const uint16_t* in = (const uint16_t*)input_bf16;
    uint8_t* out = (uint8_t*)out_f8;
    int64_t n = m * k;

    // E4M3 max representable value = 448.0
    const float f8_max = 448.0f;

    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float abs_v = fabsf(bf16_to_f32(in[i]));
        if (abs_v > max_abs) max_abs = abs_v;
    }

    float scale = max_abs / f8_max;
    if (max_abs == 0.0f) {
        memset(out, 0, n);
        return scale;
    }

    float inv_scale = f8_max / max_abs;
    for (int64_t i = 0; i < n; i++) {
        float v = bf16_to_f32(in[i]) * inv_scale;
        out[i] = f32_to_f8e4m3(v);
    }
    return scale;
}

void brgemm_f8e4m3_linear(
    const void* input_f8, float a_scale,
    brgemm_f8_packed_b_t pw,
    void* output_bf16,
    int64_t m, int64_t n_total,
    int64_t n_start, int64_t n_end)
{
    if (m <= 0 || n_start >= n_end || !pw) return;

    uint16_t* out = (uint16_t*)output_bf16;
    const uint8_t* packed = (const uint8_t*)pw->data;
    int64_t k = pw->k;
    int64_t block_n = pw->block_n;
    int64_t block_elems = k * block_n;

    static thread_local std::vector<float> tls_f8_acc;
    size_t acc_needed = (size_t)(m * BRGEMM_BLOCK_N);
    if (tls_f8_acc.size() < acc_needed) tls_f8_acc.resize(acc_needed);
    float* acc = tls_f8_acc.data();
    int64_t offsets[2] = {0, 0};

    int64_t nb_start = n_start / block_n;
    int64_t nb_end = (n_end + block_n - 1) / block_n;

    dnnl_ukernel_attr_params_t params = nullptr;
    CHECK_DNNL(dnnl_ukernel_attr_params_create(&params));
    CHECK_DNNL(dnnl_ukernel_attr_params_set_A_scales(params, &a_scale));

    const BrgemmS8Kernel* current_kern = nullptr;

    for (int64_t nb = nb_start; nb < nb_end; nb++) {
        int64_t nc = nb * block_n;
        int64_t actual_n = (nc + block_n <= pw->n) ? block_n : (pw->n - nc);

        BrgemmS8Kernel& kern = get_brgemm_f8_kernel(m, actual_n, k, n_total);
        if (&kern != current_kern) {
            dnnl_brgemm_set_hw_context(kern.brgemm);
            current_kern = &kern;
        }

        const void* B = packed + nb * block_elems;
        CHECK_DNNL(dnnl_ukernel_attr_params_set_B_scales(params, pw->scales + nc));

        uint16_t* D = out + nc;

        CHECK_DNNL(dnnl_brgemm_execute_postops(kern.brgemm,
            input_f8, B, offsets,
            acc, D, kern.scratchpad.data(), params));
    }

    dnnl_brgemm_release_hw_context();
    tls_current = nullptr;
    dnnl_ukernel_attr_params_destroy(params);
}

// ── BRGeMM post-ops (BF16 with fused bias/GELU/ReLU) ───────────────────

struct BrgemmPostopsKey {
    int64_t m, n, k, ldd;
    int flags;
    bool operator==(const BrgemmPostopsKey& o) const {
        return m==o.m && n==o.n && k==o.k && ldd==o.ldd && flags==o.flags;
    }
};
struct BrgemmPostopsKeyHash {
    size_t operator()(const BrgemmPostopsKey& key) const {
        return (size_t)(key.m * 1000003 + key.n * 1009 + key.k * 31 + key.ldd + key.flags * 7);
    }
};
struct BrgemmPostopsKernel {
    dnnl_brgemm_t brgemm = nullptr;
    std::vector<uint8_t> scratchpad;
    int num_binary_args;  // number of post-op args needing external data
};
static thread_local std::unordered_map<BrgemmPostopsKey, BrgemmPostopsKernel, BrgemmPostopsKeyHash> tls_brgemm_postops;

static BrgemmPostopsKernel& get_brgemm_postops_kernel(
    int64_t m, int64_t n, int64_t k, int64_t ldd, int flags)
{
    BrgemmPostopsKey key{m, n, k, ldd, flags};
    auto it = tls_brgemm_postops.find(key);
    if (it != tls_brgemm_postops.end()) return it->second;

    BrgemmPostopsKernel kern{};
    CHECK_DNNL(dnnl_brgemm_create(&kern.brgemm, m, n, k,
        1, k, n, n,
        dnnl_bf16, dnnl_bf16, dnnl_f32));
    CHECK_DNNL(dnnl_brgemm_set_add_C(kern.brgemm, 0));

    dnnl_post_ops_t post_ops = nullptr;
    CHECK_DNNL(dnnl_post_ops_create(&post_ops));
    kern.num_binary_args = 0;

    if (flags & BRGEMM_POSTOP_BIAS) {
        dnnl_dims_t bias_dims = {1, n};
        dnnl_memory_desc_t bias_md = nullptr;
        CHECK_DNNL(dnnl_memory_desc_create_with_tag(&bias_md, 2, bias_dims, dnnl_bf16, dnnl_ab));
        CHECK_DNNL(dnnl_post_ops_append_binary(post_ops, dnnl_binary_add, bias_md));
        dnnl_memory_desc_destroy(bias_md);
        kern.num_binary_args++;
    }
    if (flags & BRGEMM_POSTOP_GELU_TANH) {
        CHECK_DNNL(dnnl_post_ops_append_eltwise(post_ops, dnnl_eltwise_gelu_tanh, 0.0f, 0.0f));
    }
    if (flags & BRGEMM_POSTOP_GELU_ERF) {
        CHECK_DNNL(dnnl_post_ops_append_eltwise(post_ops, dnnl_eltwise_gelu_erf, 0.0f, 0.0f));
    }
    if (flags & BRGEMM_POSTOP_RELU) {
        CHECK_DNNL(dnnl_post_ops_append_eltwise(post_ops, dnnl_eltwise_relu, 0.0f, 0.0f));
    }

    CHECK_DNNL(dnnl_brgemm_set_post_ops(kern.brgemm, ldd, dnnl_bf16, post_ops));
    dnnl_post_ops_destroy(post_ops);

    CHECK_DNNL(dnnl_brgemm_finalize(kern.brgemm));
    CHECK_DNNL(dnnl_brgemm_generate(kern.brgemm));

    size_t sz = 0;
    CHECK_DNNL(dnnl_brgemm_get_scratchpad_size(kern.brgemm, &sz));
    kern.scratchpad.resize(sz);

    auto [inserted, _] = tls_brgemm_postops.emplace(key, std::move(kern));
    return inserted->second;
}

void brgemm_bf16_linear_postops(
    const void* input, brgemm_packed_b_t pw,
    void* output,
    const void* bias_bf16,
    int postop_flags,
    int64_t m, int64_t n_total,
    int64_t n_start, int64_t n_end)
{
    if (m <= 0 || n_start >= n_end || !pw) return;

    const uint16_t* A = (const uint16_t*)input;
    uint16_t* out = (uint16_t*)output;
    const uint16_t* packed = (const uint16_t*)pw->data;
    int64_t k = pw->k;
    int64_t block_n = pw->block_n;
    int64_t block_elems = k * block_n;

    // Thread-local F32 accumulator
    static thread_local std::vector<float> tls_postops_acc;
    size_t acc_needed = (size_t)(m * BRGEMM_BLOCK_N);
    if (tls_postops_acc.size() < acc_needed) tls_postops_acc.resize(acc_needed);
    float* acc = tls_postops_acc.data();
    int64_t offsets[2] = {0, 0};

    int64_t nb_start = n_start / block_n;
    int64_t nb_end = (n_end + block_n - 1) / block_n;

    dnnl_ukernel_attr_params_t params = nullptr;
    CHECK_DNNL(dnnl_ukernel_attr_params_create(&params));

    const BrgemmPostopsKernel* current_kern = nullptr;

    for (int64_t nb = nb_start; nb < nb_end; nb++) {
        int64_t nc = nb * block_n;
        int64_t actual_n = (nc + block_n <= pw->n) ? block_n : (pw->n - nc);

        BrgemmPostopsKernel& kern = get_brgemm_postops_kernel(
            m, actual_n, k, n_total, postop_flags);
        if (&kern != current_kern) {
            dnnl_brgemm_set_hw_context(kern.brgemm);
            current_kern = &kern;
        }

        const void* B = packed + nb * block_elems;

        // Set bias pointer for this block's columns
        if (kern.num_binary_args > 0 && bias_bf16) {
            const void* args[1] = { (const uint16_t*)bias_bf16 + nc };
            CHECK_DNNL(dnnl_ukernel_attr_params_set_post_ops_args(params, args));
        }

        // D_ptr at column nc with stride n_total
        uint16_t* D = out + nc;

        CHECK_DNNL(dnnl_brgemm_execute_postops(kern.brgemm,
            A, B, offsets,
            acc, D, kern.scratchpad.data(), params));
    }

    dnnl_brgemm_release_hw_context();
    tls_current = nullptr;
    dnnl_ukernel_attr_params_destroy(params);
}

#else
// Stubs when brgemm ukernel API is not available
int brgemm_available(void) { return 0; }
brgemm_packed_b_t brgemm_bf16_pack(const void* w, int64_t k, int64_t n) {
    (void)w; (void)k; (void)n; return NULL;
}
void brgemm_bf16_pack_destroy(brgemm_packed_b_t pw) { (void)pw; }
void brgemm_bf16_linear(const void* in, brgemm_packed_b_t pw, void* out,
    int64_t m, int64_t n, int64_t ns, int64_t ne)
{
    (void)in;(void)pw;(void)out;(void)m;(void)n;(void)ns;(void)ne;
}
void brgemm_bf16_linear_fused_silu_mul(const void* in, brgemm_packed_b_t pw,
    void* out, int64_t m, int64_t dim, int64_t ns, int64_t ne)
{
    (void)in;(void)pw;(void)out;(void)m;(void)dim;(void)ns;(void)ne;
}
void brgemm_score_v_accum(const float* s, const uint16_t* v, float* c,
    int64_t m, int64_t k, int64_t n, int64_t lda, int64_t vs)
{
    (void)s;(void)v;(void)c;(void)m;(void)k;(void)n;(void)lda;(void)vs;
}
void brgemm_qk_gemm(const uint16_t* q, const uint16_t* k, float* c,
    int64_t m, int64_t n, int64_t d, int64_t qs, int64_t ks, int64_t ldc, float sm)
{
    (void)q;(void)k;(void)c;(void)m;(void)n;(void)d;(void)qs;(void)ks;(void)ldc;(void)sm;
}
void brgemm_attn_release() {}
// INT8 stubs
int brgemm_s8_available(void) { return 0; }
brgemm_s8_packed_b_t brgemm_s8_pack(const int8_t* w, const float* s, int64_t k, int64_t n) {
    (void)w;(void)s;(void)k;(void)n; return NULL;
}
void brgemm_s8_pack_destroy(brgemm_s8_packed_b_t pw) { (void)pw; }
float brgemm_quantize_bf16_s8(const void* in, int8_t* out, int64_t m, int64_t k) {
    (void)in;(void)out;(void)m;(void)k; return 0.0f;
}
void brgemm_s8_linear(const int8_t* in, float as, brgemm_s8_packed_b_t pw,
    void* out, int64_t m, int64_t nt, int64_t ns, int64_t ne)
{
    (void)in;(void)as;(void)pw;(void)out;(void)m;(void)nt;(void)ns;(void)ne;
}
// FP8 stubs
int brgemm_f8_available(void) { return 0; }
brgemm_f8_packed_b_t brgemm_f8e4m3_pack(const void* w, const float* s, int64_t k, int64_t n) {
    (void)w;(void)s;(void)k;(void)n; return NULL;
}
void brgemm_f8_pack_destroy(brgemm_f8_packed_b_t pw) { (void)pw; }
float brgemm_quantize_bf16_f8e4m3(const void* in, void* out, int64_t m, int64_t k) {
    (void)in;(void)out;(void)m;(void)k; return 0.0f;
}
void brgemm_f8e4m3_linear(const void* in, float as, brgemm_f8_packed_b_t pw,
    void* out, int64_t m, int64_t nt, int64_t ns, int64_t ne)
{
    (void)in;(void)as;(void)pw;(void)out;(void)m;(void)nt;(void)ns;(void)ne;
}
// Post-ops stubs
void brgemm_bf16_linear_postops(const void* in, brgemm_packed_b_t pw,
    void* out, const void* bias, int flags,
    int64_t m, int64_t nt, int64_t ns, int64_t ne)
{
    (void)in;(void)pw;(void)out;(void)bias;(void)flags;(void)m;(void)nt;(void)ns;(void)ne;
}
#endif

// Standalone ablation: call run_mha_v3 with q/k/v/o allocated by PLAIN cudaMalloc (not
// cudarc's cuMemAllocAsync mempool that candle uses). Tests the "allocator page-property /
// alignment" hypothesis for the uniform ~15% kernel gap. Same uniform 4x2048 batch.
// Build: nvcc probe_rawbuf.cu libcandle_fa3_v3.a -lcudart -lcuda -o probe_rawbuf
//
// Optional argv[1] = path to a cubin (e.g. extracted from vLLM's _vllm_fa3_C.abi3.so):
// runs an A/B transplant experiment — same host code, same params, but the main kernel is
// loaded from the cubin by mangled name (see fa3_transplant_launch). Verifies bit-identical
// output, then times both. Requires V3_PACKGQA=1 when the cubin only has PackGQA kernels.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

extern "C" void run_mha_v3(
    void*,void*,void*,void*,void*, const int*,const int*, int*,
    unsigned,unsigned,unsigned,unsigned, unsigned,unsigned,unsigned,unsigned,
    unsigned,unsigned,unsigned,unsigned, unsigned,unsigned, unsigned,unsigned, float,int,int);

// cheap per-element pseudo-random bf16 fill (varied across seqlen so online-softmax rescales)
__global__ void fill_bf16(unsigned short* p, long n, unsigned seed) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned)(i * 2654435761u + seed * 40503u);
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    unsigned y = x * 0x27d4eb2du; y ^= y >> 15;
    float u1 = ((x & 0xffffff) + 1.0f) / 16777217.0f;
    float u2 = ((y & 0xffffff)) / 16777216.0f;
    float f = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);  // N(0,1) Box-Muller
    // f32 -> bf16 (truncate)
    unsigned fb = __float_as_uint(f);
    p[i] = (unsigned short)(fb >> 16);
}

int main(int argc, char** argv) {
    int B = 4, S = 2048, HQ = 32, HKV = 4, D = 128;
    int WARMUP = 20, ITERS = 200;
    long total = (long)B * S;
    float scale = 1.0f / 11.3137085f; // 1/sqrt(128)

    auto cm = [](void** p, size_t b){ if (cudaMalloc(p, b) != cudaSuccess) { printf("malloc fail\n"); exit(1);} };
    void *q,*k,*v,*o,*lse; int *cu,*sched;
    cm(&q, total*HQ*D*2); cm(&k, total*HKV*D*2); cm(&v, total*HKV*D*2); cm(&o, total*HQ*D*2);
    cm(&lse, (size_t)HQ*total*4);
    int b_rounded = (B+3)/4*4;
    cm((void**)&sched, (1 + b_rounded*2)*4); cudaMemset(sched, 0, (1+b_rounded*2)*4);
    cm((void**)&cu, (B+1)*4);
    std::vector<int> cuh(B+1); for (int i=0;i<=B;i++) cuh[i]=i*S;
    cudaMemcpy(cu, cuh.data(), (B+1)*4, cudaMemcpyHostToDevice);
    long qn = total*HQ*D, kn = total*HKV*D;
    fill_bf16<<<(qn+255)/256,256>>>((unsigned short*)q, qn, 1);
    fill_bf16<<<(kn+255)/256,256>>>((unsigned short*)k, kn, 2);
    fill_bf16<<<(kn+255)/256,256>>>((unsigned short*)v, kn, 3);
    cudaDeviceSynchronize();

    unsigned qrs=HQ*D, krs=HKV*D, ors=HQ*D, qhs=D, khs=D, ohs=D;
    auto call = [&](){ run_mha_v3(q,k,v,o,lse, cu,cu, sched,
        qrs,krs,krs,ors, qhs,khs,khs,ohs, (unsigned)B,HQ,HKV,D,(unsigned)S,(unsigned)S,
        (unsigned)total,(unsigned)total, scale, 1, 1); };

    const char* cubin = argc > 1 ? argv[1] : nullptr;
    unsetenv("FA3_CUBIN");

    // V3_SYNC_PER_ITER=1 mimics torch-style benches (event pair + sync each iter, brief
    // host gaps) vs default back-to-back launching; isolates GPU clock/boost effects.
    bool sync_per_iter = getenv("V3_SYNC_PER_ITER") && getenv("V3_SYNC_PER_ITER")[0]=='1';
    auto timeit = [&](const char* label){
        for (int i=0;i<WARMUP;i++) call();
        cudaDeviceSynchronize();
        float us = 0;
        if (sync_per_iter) {
            std::vector<float> ts(ITERS);
            for (int i=0;i<ITERS;i++) {
                cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
                cudaEventRecord(s); call(); cudaEventRecord(e); cudaEventSynchronize(e);
                float ms=0; cudaEventElapsedTime(&ms,s,e); ts[i]=ms*1000.0f;
                cudaEventDestroy(s); cudaEventDestroy(e);
            }
            std::sort(ts.begin(), ts.end()); us = ts[ITERS/2];
            printf("%s uniform 4x2048: %.1f us/call (median of %d, sync/iter)\n", label, us, ITERS);
        } else {
            cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
            cudaEventRecord(s);
            for (int i=0;i<ITERS;i++) call();
            cudaEventRecord(e); cudaEventSynchronize(e);
            float ms=0; cudaEventElapsedTime(&ms,s,e);
            printf("%s uniform 4x2048: %.1f us/call (mean of %d)\n", label, ms*1000.0/ITERS, ITERS);
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("%s: ASYNC ERROR: %s\n", label, cudaGetErrorString(err)); exit(1); }
    };

    size_t obytes = (size_t)total*HQ*D*2, lbytes = (size_t)HQ*total*4;
    std::vector<unsigned char> o_own(obytes), lse_own(lbytes), o_tx(obytes), lse_tx(lbytes);

    // own-build kernel
    call(); cudaDeviceSynchronize();
    cudaMemcpy(o_own.data(), o, obytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(lse_own.data(), lse, lbytes, cudaMemcpyDeviceToHost);
    timeit("OWN   (static-linked)");

    if (cubin) {
        setenv("FA3_CUBIN", cubin, 1);
        cudaMemset(o, 0, obytes); cudaMemset(lse, 0, lbytes);  // avoid stale-buffer false positive
        call();
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("transplant run failed: %s\n", cudaGetErrorString(err)); return 1; }
        cudaMemcpy(o_tx.data(), o, obytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(lse_tx.data(), lse, lbytes, cudaMemcpyDeviceToHost);
        printf("TRANSPLANT correctness: o %s, lse %s\n",
               memcmp(o_own.data(), o_tx.data(), obytes) == 0 ? "bit-identical" : "MISMATCH",
               memcmp(lse_own.data(), lse_tx.data(), lbytes) == 0 ? "bit-identical" : "MISMATCH");
        timeit("VLLM-CUBIN (transplant)");
        // re-verify AFTER the timed loop: if the kernel started no-op'ing, this catches it
        cudaMemset(o, 0, obytes);
        call(); cudaDeviceSynchronize();
        cudaMemcpy(o_tx.data(), o, obytes, cudaMemcpyDeviceToHost);
        printf("TRANSPLANT post-loop correctness: o %s\n",
               memcmp(o_own.data(), o_tx.data(), obytes) == 0 ? "bit-identical" : "MISMATCH");
    }
    return 0;
}

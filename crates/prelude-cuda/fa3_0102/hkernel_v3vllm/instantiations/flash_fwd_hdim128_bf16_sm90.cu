// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#ifdef FA3_CLOCK_PROBE
// Per-CTA clock64 probe globals, defined BEFORE the kernel header so operator() can use them
// (single-instantiation probe build only). A host reader is at the bottom.
#include <cuda_runtime.h>
__device__ unsigned long long fa3_probe_start[256];
__device__ unsigned long long fa3_probe_end[256];
__device__ unsigned fa3_probe_sm[256];
#endif

#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false>(Flash_fwd_params &params, cudaStream_t stream);
#endif

#ifdef FA3_CLOCK_PROBE
extern "C" void fa3_read_probe(unsigned long long* s, unsigned long long* e, unsigned* m) {
    cudaMemcpyFromSymbol(s, fa3_probe_start, sizeof(unsigned long long) * 256);
    cudaMemcpyFromSymbol(e, fa3_probe_end, sizeof(unsigned long long) * 256);
    cudaMemcpyFromSymbol(m, fa3_probe_sm, sizeof(unsigned) * 256);
}
#endif

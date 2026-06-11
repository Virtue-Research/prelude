// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
//
// PagedKVNonTMA=true + PackGQA=true: used by the paged serving path when
// pagedkv_tma is false (decode / small-seqlen_q batches where
// seqlen_q*(h/h_k) <= kBlockM). Matches vLLM's run_mha_fwd dispatch, which
// forces PackGQA=true whenever PagedKVNonTMA is set.

#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, true, false, true>(Flash_fwd_params &params, cudaStream_t stream);
#endif

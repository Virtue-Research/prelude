//! Microbenchmark: FA4 vs FA3 kernel latency (non-paged AND paged).
//!
//! Directly calls both kernels with identical input, measures GPU time via cudaEvent.
//! No server/candle overhead — pure kernel + FFI cost.
//!
//! Run:
//!   CUDA_VISIBLE_DEVICES=1 cargo bench -p prelude-flash-attn-v4 --bench fa4_vs_fa3

use half::bf16;
use prelude_flash_attn_v4::{KernelDtype, KernelKey, KernelRegistry};
use std::ffi::c_void;

// ── CUDA FFI ────────────────────────────────────────────────────────

unsafe extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
}

// vLLM paged attention C FFI (linked from candle-paged-attn)
#[link(name = "pagedattention", kind = "static")]
unsafe extern "C" {
    fn paged_attention_v1(
        out: *const c_void, query: *const c_void,
        key_cache: *const c_void, value_cache: *const c_void,
        num_kv_heads: i32, scale: f32,
        block_tables: *const i32, context_lens: *const i32,
        block_size: i32, max_context_len: i32,
        num_seqs: i32, num_heads: i32, head_size: i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32, kv_block_stride: i32, kv_head_stride: i32,
        dtype: u32, softcapping: f32, stream: i64,
    );

    fn paged_attention_v2(
        out: *const c_void, exp_sums: *const f32, max_logits: *const f32, tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void, value_cache: *const c_void,
        num_kv_heads: i32, scale: f32,
        block_tables: *const i32, context_lens: *const i32,
        block_size: i32, max_context_len: i32,
        num_seqs: i32, num_heads: i32, head_size: i32,
        max_num_blocks_per_seq: i32,
        q_stride: i32, kv_block_stride: i32, kv_head_stride: i32,
        dtype: u32, softcapping: f32, stream: i64,
    );
}

// FA3 C FFI (linked from candle-flash-attn-v3)
#[link(name = "flashattentionv3", kind = "static")]
#[link(name = "cuda")]
unsafe extern "C" {
    fn run_mha(
        q_ptr: *const c_void, k_ptr: *const c_void, v_ptr: *const c_void,
        o_ptr: *const c_void, softmax_lse_ptr: *const c_void, alibi_slopes_ptr: *const c_void,
        cu_seqlens_q_ptr: *const i32, cu_seqlens_k_ptr: *const i32,
        q_batch_stride: u32, k_batch_stride: u32, v_batch_stride: u32,
        o_batch_stride: u32, alibi_slopes_batch_stride: u32,
        q_row_stride: u32, k_row_stride: u32, v_row_stride: u32, o_row_stride: u32,
        q_head_stride: u32, k_head_stride: u32, v_head_stride: u32, o_head_stride: u32,
        b: u32, h: u32, h_k: u32, d: u32, d_rounded: u32, softmax_scale: f32,
        seqlen_q: u32, seqlen_k: u32, seqlen_q_rounded: u32, seqlen_k_rounded: u32,
        is_bf16: i32, is_causal: i32, unpadded_lse: i32, use_gqa_packing: i32,
        window_size_left: i32, window_size_right: i32,
        total_q: u32, total_k: u32, stream: *mut c_void,
    );

    fn run_mha_paged(
        q_ptr: *const c_void, k_ptr: *const c_void, v_ptr: *const c_void,
        o_ptr: *const c_void, softmax_lse_ptr: *const c_void, alibi_slopes_ptr: *const c_void,
        cu_seqlens_q_ptr: *const i32, cu_seqlens_k_ptr: *const i32,
        block_table_ptr: *const i32,
        q_row_stride: u32, q_head_stride: u32,
        k_batch_stride: u32, k_row_stride: u32, k_head_stride: u32,
        v_batch_stride: u32, v_row_stride: u32, v_head_stride: u32,
        o_row_stride: u32, o_head_stride: u32,
        block_table_batch_stride: u32,
        b: u32, b_k: u32, h: u32, h_k: u32, d: u32, d_rounded: u32,
        softmax_scale: f32,
        seqlen_q: u32, seqlen_k: u32, seqlen_q_rounded: u32, seqlen_k_rounded: u32,
        page_block_size: u32, page_num_blocks: u32,
        is_bf16: i32, is_causal: i32, unpadded_lse: i32, use_gqa_packing: i32,
        window_size_left: i32, window_size_right: i32,
        total_q: u32, total_k: u32, stream: *mut c_void,
    );
}

const CUDA_MEMCPY_H2D: i32 = 1;

// ── GPU buffer ──────────────────────────────────────────────────────

struct GpuBuf { ptr: *mut c_void, #[allow(dead_code)] bytes: usize }

impl GpuBuf {
    fn alloc(bytes: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&mut ptr, bytes) }, 0, "cudaMalloc failed");
        Self { ptr, bytes }
    }
    fn upload<T: Copy>(&self, data: &[T]) {
        let bytes = data.len() * std::mem::size_of::<T>();
        assert_eq!(
            unsafe { cudaMemcpy(self.ptr, data.as_ptr() as _, bytes, CUDA_MEMCPY_H2D) },
            0, "cudaMemcpy H2D failed",
        );
    }
}

impl Drop for GpuBuf {
    fn drop(&mut self) { unsafe { cudaFree(self.ptr) }; }
}

// ── Timing ──────────────────────────────────────────────────────────

struct CudaTimer { start: *mut c_void, stop: *mut c_void }

impl CudaTimer {
    fn new() -> Self {
        let (mut start, mut stop) = (std::ptr::null_mut(), std::ptr::null_mut());
        unsafe { cudaEventCreate(&mut start); cudaEventCreate(&mut stop); }
        Self { start, stop }
    }
    fn record_start(&self, stream: *mut c_void) { unsafe { cudaEventRecord(self.start, stream) }; }
    fn record_stop(&self, stream: *mut c_void) { unsafe { cudaEventRecord(self.stop, stream) }; }
    fn elapsed_ms(&self) -> f32 {
        let mut ms: f32 = 0.0;
        unsafe { cudaEventSynchronize(self.stop); cudaEventElapsedTime(&mut ms, self.start, self.stop); }
        ms
    }
}

impl Drop for CudaTimer {
    fn drop(&mut self) { unsafe { cudaEventDestroy(self.start); cudaEventDestroy(self.stop); } }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn round_multiple(x: usize, m: usize) -> usize { (x + m - 1) / m * m }

fn generate_bf16(count: usize, seed: u32) -> Vec<bf16> {
    (0..count)
        .map(|i| bf16::from_f32(((i as f32 + seed as f32 * 0.1) * 0.01).sin() * 0.5))
        .collect()
}

fn report(label: &str, fa3_ms: f32, fa4_ms: Option<f32>) {
    match fa4_ms {
        Some(fa4) => {
            let delta = (fa4 - fa3_ms) / fa3_ms * 100.0;
            let sign = if delta > 0.0 { "+" } else { "" };
            eprintln!("  {label}\n    FA3: {fa3_ms:.4}ms  FA4: {fa4:.4}ms  ({sign}{delta:.1}%)");
        }
        None => eprintln!("  {label}\n    FA3: {fa3_ms:.4}ms  FA4: N/A"),
    }
}

/// Measure N iterations of a closure, return ms/iter.
fn bench_loop(
    stream: *mut c_void, warmup: usize, iters: usize,
    timer: &CudaTimer, mut f: impl FnMut(),
) -> f32 {
    for _ in 0..warmup { f(); }
    unsafe { cudaDeviceSynchronize() };
    timer.record_start(stream);
    for _ in 0..iters { f(); }
    timer.record_stop(stream);
    timer.elapsed_ms() / iters as f32
}

// ── Non-paged varlen benchmark ──────────────────────────────────────

fn bench_varlen(
    total_tokens: usize, num_heads_q: usize, num_heads_k: usize,
    head_dim: usize, batch_size: usize, warmup: usize, iters: usize,
) {
    let registry = KernelRegistry::new();
    let gqa_ratio = num_heads_q / num_heads_k;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let seq_len = total_tokens / batch_size;

    let key = KernelKey::new(head_dim as u32, gqa_ratio as u32, true, false);
    let fa4_func = registry.get(&key);

    let mut stream: *mut c_void = std::ptr::null_mut();
    unsafe { cudaStreamCreate(&mut stream) };

    let q_elems = total_tokens * num_heads_q * head_dim;
    let k_elems = total_tokens * num_heads_k * head_dim;
    let lse_elems = num_heads_q * total_tokens;

    let q_gpu = GpuBuf::alloc(q_elems * 2);
    let k_gpu = GpuBuf::alloc(k_elems * 2);
    let v_gpu = GpuBuf::alloc(k_elems * 2);
    let o_gpu = GpuBuf::alloc(q_elems * 2);
    let cu_gpu = GpuBuf::alloc((batch_size + 1) * 4);
    let lse_gpu = GpuBuf::alloc(lse_elems * 4);

    q_gpu.upload(&generate_bf16(q_elems, 1));
    k_gpu.upload(&generate_bf16(k_elems, 2));
    v_gpu.upload(&generate_bf16(k_elems, 3));
    let mut cu_seqlens = vec![0i32; batch_size + 1];
    for i in 0..batch_size { cu_seqlens[i + 1] = cu_seqlens[i] + seq_len as i32; }
    cu_gpu.upload(&cu_seqlens);
    o_gpu.upload(&vec![bf16::ZERO; q_elems]);
    lse_gpu.upload(&vec![0.0f32; lse_elems]);

    let timer = CudaTimer::new();
    let head_size_rounded = round_multiple(head_dim, 32);
    let seqlen_q_rounded = round_multiple(seq_len, 128);
    let seqlen_k_rounded = round_multiple(seq_len, 128);
    let use_gqa_packing = if gqa_ratio >= 2 { 1i32 } else { 0 };

    let fa3_ms = bench_loop(stream, warmup, iters, &timer, || unsafe {
        run_mha(
            q_gpu.ptr, k_gpu.ptr, v_gpu.ptr, o_gpu.ptr,
            lse_gpu.ptr, std::ptr::null(),
            cu_gpu.ptr as _, cu_gpu.ptr as _,
            0, 0, 0, 0, 0,
            (num_heads_q * head_dim) as u32, (num_heads_k * head_dim) as u32,
            (num_heads_k * head_dim) as u32, (num_heads_q * head_dim) as u32,
            head_dim as u32, head_dim as u32, head_dim as u32, head_dim as u32,
            batch_size as u32, num_heads_q as u32, num_heads_k as u32,
            head_dim as u32, head_size_rounded as u32, softmax_scale,
            seq_len as u32, seq_len as u32, seqlen_q_rounded as u32, seqlen_k_rounded as u32,
            1, 1, 1, use_gqa_packing, -1, 0,
            total_tokens as u32, total_tokens as u32, stream,
        );
    });

    let fa4_ms = fa4_func.map(|func| {
        let q_shape: [i64; 3] = [total_tokens as _, num_heads_q as _, head_dim as _];
        let k_shape: [i64; 3] = [total_tokens as _, num_heads_k as _, head_dim as _];
        let o_shape = q_shape;
        let lse_shape: [i64; 2] = [num_heads_q as _, total_tokens as _];
        let cu_shape: [i64; 1] = [(batch_size + 1) as _];

        bench_loop(stream, warmup, iters, &timer, || unsafe {
            prelude_flash_attn_v4::fa4_varlen_fwd(
                &registry, func,
                q_gpu.ptr, k_gpu.ptr, v_gpu.ptr, o_gpu.ptr,
                std::ptr::null_mut(), softmax_scale, stream,
                cu_gpu.ptr, cu_gpu.ptr,
                &q_shape, &k_shape, &o_shape, &lse_shape, &cu_shape,
                0, None, None, None, None,
                KernelDtype::BF16,
            ).expect("FA4 kernel failed");
        })
    });

    report(
        &format!("varlen b={batch_size} seq={seq_len} hq={num_heads_q} hk={num_heads_k} d={head_dim}"),
        fa3_ms, fa4_ms,
    );
}

// ── Paged varlen benchmark ──────────────────────────────────────────

fn bench_paged(
    total_q: usize, total_k: usize,
    num_heads_q: usize, num_heads_k: usize,
    head_dim: usize, batch_size: usize,
    warmup: usize, iters: usize,
) {
    let registry = KernelRegistry::new();
    let gqa_ratio = num_heads_q / num_heads_k;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let seq_q = total_q / batch_size;
    let seq_k = total_k / batch_size;

    // Block size for paged KV. FA4 TMA needs block_size == tile_n.
    let block_size: usize = match head_dim {
        d if d <= 128 => 128,
        d if d <= 192 => 112,
        _ => 80,
    };

    // FA4 paged kernel
    let fa4_key = KernelKey::new(head_dim as u32, gqa_ratio as u32, true, false)
        .with_paged(true);
    let fa4_func = registry.get(&fa4_key);

    let mut stream: *mut c_void = std::ptr::null_mut();
    unsafe { cudaStreamCreate(&mut stream) };

    // Allocate Q: [total_q, num_heads_q, head_dim]
    let q_elems = total_q * num_heads_q * head_dim;
    let q_gpu = GpuBuf::alloc(q_elems * 2);
    q_gpu.upload(&generate_bf16(q_elems, 1));

    // Allocate paged KV cache: [num_blocks, block_size, num_heads_k, head_dim]
    let blocks_per_seq = (seq_k + block_size - 1) / block_size;
    let max_blocks_per_seq = blocks_per_seq;
    let total_blocks = blocks_per_seq * batch_size;
    let cache_elems = total_blocks * block_size * num_heads_k * head_dim;

    let k_cache_gpu = GpuBuf::alloc(cache_elems * 2);
    let v_cache_gpu = GpuBuf::alloc(cache_elems * 2);
    k_cache_gpu.upload(&generate_bf16(cache_elems, 2));
    v_cache_gpu.upload(&generate_bf16(cache_elems, 3));

    // Output
    let o_gpu = GpuBuf::alloc(q_elems * 2);
    o_gpu.upload(&vec![bf16::ZERO; q_elems]);

    // LSE
    let lse_elems = num_heads_q * total_q;
    let lse_gpu = GpuBuf::alloc(lse_elems * 4);
    lse_gpu.upload(&vec![0.0f32; lse_elems]);

    // cu_seqlens_q
    let mut cu_seqlens_q = vec![0i32; batch_size + 1];
    for i in 0..batch_size { cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seq_q as i32; }
    let cu_q_gpu = GpuBuf::alloc((batch_size + 1) * 4);
    cu_q_gpu.upload(&cu_seqlens_q);

    // cu_seqlens_k (for FA3)
    let mut cu_seqlens_k = vec![0i32; batch_size + 1];
    for i in 0..batch_size { cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seq_k as i32; }
    let cu_k_gpu = GpuBuf::alloc((batch_size + 1) * 4);
    cu_k_gpu.upload(&cu_seqlens_k);

    // seqused_k (for FA4): per-seq K lengths
    let seqused_k: Vec<i32> = vec![seq_k as i32; batch_size];
    let seqused_k_gpu = GpuBuf::alloc(batch_size * 4);
    seqused_k_gpu.upload(&seqused_k);

    // Page table: sequential block assignment [batch_size, max_blocks_per_seq]
    let mut page_table = vec![0i32; batch_size * max_blocks_per_seq];
    for seq in 0..batch_size {
        for blk in 0..blocks_per_seq {
            page_table[seq * max_blocks_per_seq + blk] = (seq * blocks_per_seq + blk) as i32;
        }
    }
    let pt_gpu = GpuBuf::alloc(batch_size * max_blocks_per_seq * 4);
    pt_gpu.upload(&page_table);

    let timer = CudaTimer::new();
    let head_size_rounded = round_multiple(head_dim, 32);
    let seqlen_q_rounded = round_multiple(seq_q, 128);
    let seqlen_k_rounded = round_multiple(seq_k, 128);
    let use_gqa_packing = if gqa_ratio >= 2 { 1i32 } else { 0 };

    // ── FA3 paged ───────────────────────────────────────────────────
    // FA3 paged: K/V are [num_blocks, block_size, num_heads_k, head_dim] (flash layout)
    let k_batch_stride = (block_size * num_heads_k * head_dim) as u32;
    let k_row_stride = (num_heads_k * head_dim) as u32;
    let k_head_stride = head_dim as u32;

    let fa3_ms = bench_loop(stream, warmup, iters, &timer, || unsafe {
        run_mha_paged(
            q_gpu.ptr, k_cache_gpu.ptr, v_cache_gpu.ptr, o_gpu.ptr,
            lse_gpu.ptr, std::ptr::null(),
            cu_q_gpu.ptr as _, cu_k_gpu.ptr as _, pt_gpu.ptr as _,
            (num_heads_q * head_dim) as u32, head_dim as u32, // q strides
            k_batch_stride, k_row_stride, k_head_stride, // k strides
            k_batch_stride, k_row_stride, k_head_stride, // v strides (same)
            (num_heads_q * head_dim) as u32, head_dim as u32, // o strides
            max_blocks_per_seq as u32, // block_table_batch_stride
            batch_size as u32, // b
            total_blocks as u32, // b_k (num_blocks)
            num_heads_q as u32, num_heads_k as u32,
            head_dim as u32, head_size_rounded as u32, softmax_scale,
            seq_q as u32, seq_k as u32, seqlen_q_rounded as u32, seqlen_k_rounded as u32,
            block_size as u32, total_blocks as u32,
            1, 1, 1, use_gqa_packing, -1, 0,
            total_q as u32, total_k as u32, stream,
        );
    });

    // ── FA4 paged ───────────────────────────────────────────────────
    let fa4_ms = fa4_func.map(|func| {
        let q_shape: [i64; 3] = [total_q as _, num_heads_q as _, head_dim as _];
        let k_shape: [i64; 4] = [total_blocks as _, block_size as _, num_heads_k as _, head_dim as _];
        let o_shape = q_shape;
        let lse_shape: [i64; 2] = [num_heads_q as _, total_q as _];
        let cu_q_shape: [i64; 1] = [(batch_size + 1) as _];
        let seqused_k_shape: [i64; 1] = [batch_size as _];
        let pt_shape: [i64; 2] = [batch_size as _, max_blocks_per_seq as _];

        bench_loop(stream, warmup, iters, &timer, || unsafe {
            prelude_flash_attn_v4::fa4_varlen_paged_fwd(
                &registry, func,
                q_gpu.ptr, k_cache_gpu.ptr, v_cache_gpu.ptr, o_gpu.ptr,
                std::ptr::null_mut(), softmax_scale, stream,
                cu_q_gpu.ptr, seqused_k_gpu.ptr, pt_gpu.ptr,
                &q_shape, &k_shape, &o_shape, &lse_shape,
                &cu_q_shape, &seqused_k_shape, &pt_shape,
                0, None, None,
                KernelDtype::BF16,
            ).expect("FA4 paged kernel failed");
        })
    });

    report(
        &format!("paged  b={batch_size} qlen={seq_q} klen={seq_k} hq={num_heads_q} hk={num_heads_k} d={head_dim} bs={block_size}"),
        fa3_ms, fa4_ms,
    );
}

// ── Decode benchmark (Q=1): FA3 paged vs vLLM paged_attention ────────

fn bench_decode(
    batch_size: usize, seq_k: usize,
    num_heads_q: usize, num_heads_k: usize,
    head_dim: usize,
    warmup: usize, iters: usize,
) {
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // FA3 paged requires block_size >= tile_n (128 for hdim<=128).
    // vLLM paged_attention works with any block_size (commonly 16).
    let vllm_block_size: usize = 16;
    let fa3_block_size: usize = 128;

    let vllm_blocks_per_seq = (seq_k + vllm_block_size - 1) / vllm_block_size;
    let vllm_total_blocks = vllm_blocks_per_seq * batch_size;

    let fa3_blocks_per_seq = (seq_k + fa3_block_size - 1) / fa3_block_size;
    let fa3_total_blocks = fa3_blocks_per_seq * batch_size;

    let mut stream: *mut c_void = std::ptr::null_mut();
    unsafe { cudaStreamCreate(&mut stream) };
    let stream_i64 = stream as i64;

    // Q: [batch_size, num_heads_q, head_dim] (Q=1 per seq)
    let q_elems = batch_size * num_heads_q * head_dim;
    let q_gpu = GpuBuf::alloc(q_elems * 2);
    q_gpu.upload(&generate_bf16(q_elems, 1));

    // Output
    let o_gpu = GpuBuf::alloc(q_elems * 2);
    o_gpu.upload(&vec![bf16::ZERO; q_elems]);

    // Context lengths: [batch_size]
    let context_lens: Vec<i32> = vec![seq_k as i32; batch_size];
    let cl_gpu = GpuBuf::alloc(batch_size * 4);
    cl_gpu.upload(&context_lens);

    let timer = CudaTimer::new();

    // ── vLLM paged_attention (v1 cache layout, block_size=16) ───────
    // key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    // value_cache: [num_blocks, num_kv_heads, head_size, block_size]
    // x = 16 / sizeof(bf16) = 8
    let x = 8usize;
    let kc_v1_elems = vllm_total_blocks * num_heads_k * (head_dim / x) * vllm_block_size * x;
    let vc_v1_elems = vllm_total_blocks * num_heads_k * head_dim * vllm_block_size;
    let kc_v1_gpu = GpuBuf::alloc(kc_v1_elems * 2);
    let vc_v1_gpu = GpuBuf::alloc(vc_v1_elems * 2);
    kc_v1_gpu.upload(&generate_bf16(kc_v1_elems, 2));
    vc_v1_gpu.upload(&generate_bf16(vc_v1_elems, 3));

    // vLLM page table
    let mut vllm_pt = vec![0i32; batch_size * vllm_blocks_per_seq];
    for seq in 0..batch_size {
        for blk in 0..vllm_blocks_per_seq {
            vllm_pt[seq * vllm_blocks_per_seq + blk] = (seq * vllm_blocks_per_seq + blk) as i32;
        }
    }
    let vllm_pt_gpu = GpuBuf::alloc(vllm_pt.len() * 4);
    vllm_pt_gpu.upload(&vllm_pt);

    let kv_block_stride = (num_heads_k * head_dim * vllm_block_size) as i32;
    let kv_head_stride = (head_dim * vllm_block_size) as i32;
    let q_stride = (num_heads_q * head_dim) as i32;

    let partition_size = 512usize;
    let max_num_partitions = (seq_k + partition_size - 1) / partition_size;
    let use_v1 = (max_num_partitions == 1 || batch_size * num_heads_q > 512)
        && partition_size % vllm_block_size == 0;

    let vllm_ms = if use_v1 {
        bench_loop(stream, warmup, iters, &timer, || unsafe {
            paged_attention_v1(
                o_gpu.ptr, q_gpu.ptr, kc_v1_gpu.ptr, vc_v1_gpu.ptr,
                num_heads_k as i32, softmax_scale,
                vllm_pt_gpu.ptr as _, cl_gpu.ptr as _,
                vllm_block_size as i32, seq_k as i32,
                batch_size as i32, num_heads_q as i32, head_dim as i32,
                vllm_blocks_per_seq as i32,
                q_stride, kv_block_stride, kv_head_stride,
                1, 1.0, stream_i64,
            );
        })
    } else {
        let tmp_gpu = GpuBuf::alloc(batch_size * num_heads_q * max_num_partitions * head_dim * 2);
        let exp_gpu = GpuBuf::alloc(batch_size * num_heads_q * max_num_partitions * 4);
        let max_gpu = GpuBuf::alloc(batch_size * num_heads_q * max_num_partitions * 4);

        bench_loop(stream, warmup, iters, &timer, || unsafe {
            paged_attention_v2(
                o_gpu.ptr, exp_gpu.ptr as _, max_gpu.ptr as _, tmp_gpu.ptr,
                q_gpu.ptr, kc_v1_gpu.ptr, vc_v1_gpu.ptr,
                num_heads_k as i32, softmax_scale,
                vllm_pt_gpu.ptr as _, cl_gpu.ptr as _,
                vllm_block_size as i32, seq_k as i32,
                batch_size as i32, num_heads_q as i32, head_dim as i32,
                vllm_blocks_per_seq as i32,
                q_stride, kv_block_stride, kv_head_stride,
                1, 1.0, stream_i64,
            );
        })
    };

    // ── FA3 paged decode (flash layout, block_size=128, Q=1) ────────
    let cache_flash_elems = fa3_total_blocks * fa3_block_size * num_heads_k * head_dim;
    let kc_flash_gpu = GpuBuf::alloc(cache_flash_elems * 2);
    let vc_flash_gpu = GpuBuf::alloc(cache_flash_elems * 2);
    kc_flash_gpu.upload(&generate_bf16(cache_flash_elems, 2));
    vc_flash_gpu.upload(&generate_bf16(cache_flash_elems, 3));

    // FA3 page table
    let mut fa3_pt = vec![0i32; batch_size * fa3_blocks_per_seq];
    for seq in 0..batch_size {
        for blk in 0..fa3_blocks_per_seq {
            fa3_pt[seq * fa3_blocks_per_seq + blk] = (seq * fa3_blocks_per_seq + blk) as i32;
        }
    }
    let fa3_pt_gpu = GpuBuf::alloc(fa3_pt.len() * 4);
    fa3_pt_gpu.upload(&fa3_pt);

    let lse_elems = num_heads_q * batch_size;
    let lse_gpu = GpuBuf::alloc(lse_elems * 4);
    lse_gpu.upload(&vec![0.0f32; lse_elems]);

    let mut cu_q = vec![0i32; batch_size + 1];
    let mut cu_k = vec![0i32; batch_size + 1];
    for i in 0..batch_size {
        cu_q[i + 1] = cu_q[i] + 1;
        cu_k[i + 1] = cu_k[i] + seq_k as i32;
    }
    let cu_q_gpu = GpuBuf::alloc((batch_size + 1) * 4);
    let cu_k_gpu = GpuBuf::alloc((batch_size + 1) * 4);
    cu_q_gpu.upload(&cu_q);
    cu_k_gpu.upload(&cu_k);

    let head_size_rounded = round_multiple(head_dim, 32);
    let seqlen_k_rounded = round_multiple(seq_k, 128);
    let gqa_ratio = num_heads_q / num_heads_k;
    let use_gqa_packing = if gqa_ratio >= 2 { 1i32 } else { 0 };
    let fa3_k_batch_stride = (fa3_block_size * num_heads_k * head_dim) as u32;
    let fa3_k_row_stride = (num_heads_k * head_dim) as u32;
    let fa3_k_head_stride = head_dim as u32;

    let fa3_ms = bench_loop(stream, warmup, iters, &timer, || unsafe {
        run_mha_paged(
            q_gpu.ptr, kc_flash_gpu.ptr, vc_flash_gpu.ptr, o_gpu.ptr,
            lse_gpu.ptr, std::ptr::null(),
            cu_q_gpu.ptr as _, cu_k_gpu.ptr as _, fa3_pt_gpu.ptr as _,
            (num_heads_q * head_dim) as u32, head_dim as u32,
            fa3_k_batch_stride, fa3_k_row_stride, fa3_k_head_stride,
            fa3_k_batch_stride, fa3_k_row_stride, fa3_k_head_stride,
            (num_heads_q * head_dim) as u32, head_dim as u32,
            fa3_blocks_per_seq as u32,
            batch_size as u32, fa3_total_blocks as u32,
            num_heads_q as u32, num_heads_k as u32,
            head_dim as u32, head_size_rounded as u32, softmax_scale,
            1u32, seq_k as u32, 128u32, seqlen_k_rounded as u32,
            fa3_block_size as u32, fa3_total_blocks as u32,
            1, 1, 1, use_gqa_packing, -1, 0,
            batch_size as u32, (batch_size * seq_k) as u32, stream,
        );
    });

    let v_label = if use_v1 { "v1" } else { "v2" };
    eprintln!(
        "  decode b={batch_size} klen={seq_k} hq={num_heads_q} hk={num_heads_k} d={head_dim}\n    \
         FA3(bs={fa3_block_size}): {fa3_ms:.4}ms  vLLM-{v_label}(bs={vllm_block_size}): {vllm_ms:.4}ms  \
         (vLLM {delta:+.1}%)",
        delta = (vllm_ms - fa3_ms) / fa3_ms * 100.0,
    );
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    eprintln!("=== FA4 vs FA3 Kernel Microbenchmark ===\n");

    let warmup = 10;
    let iters = 100;

    // Qwen3-0.6B dims: head_dim=128, num_heads_q=16, num_heads_kv=8
    eprintln!("--- Non-paged varlen (Qwen3-0.6B: hdim=128, hq=16, hk=8, gqa=2) ---");
    for &(total, batch) in &[(128, 1), (512, 1), (128, 4), (512, 4)] {
        bench_varlen(total, 16, 8, 128, batch, warmup, iters);
    }

    eprintln!("\n--- Paged varlen (Qwen3-0.6B: hdim=128, hq=16, hk=8, gqa=2) ---");
    // (total_q, total_k, batch) — simulates prefill with prefix cache
    for &(tq, tk, batch) in &[
        (128, 128, 1),  // no prefix
        (128, 256, 1),  // half prefix
        (128, 512, 1),  // large prefix
        (128, 128, 4),  // batch=4, no prefix
        (128, 512, 4),  // batch=4, large prefix
        (512, 512, 1),  // long prefill
        (512, 512, 4),  // long prefill, batched
    ] {
        bench_paged(tq, tk, 16, 8, 128, batch, warmup, iters);
    }

    eprintln!("\n--- Paged varlen (hdim=128, gqa=8, Llama-like) ---");
    for &(tq, tk, batch) in &[(128, 512, 1), (128, 512, 4)] {
        bench_paged(tq, tk, 64, 8, 128, batch, warmup, iters);
    }

    eprintln!("\n--- Decode Q=1: FA3-paged vs vLLM paged_attention (Qwen3-0.6B) ---");
    for &(batch, klen) in &[
        (1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048),
        (4, 128), (4, 512), (4, 2048),
        (16, 512), (32, 512),
    ] {
        bench_decode(batch, klen, 16, 8, 128, warmup, iters);
    }

    eprintln!("\n--- Decode Q=1: FA3 vs vLLM (hdim=128, gqa=8, Llama-like) ---");
    for &(batch, klen) in &[(1, 512), (4, 512), (16, 512)] {
        bench_decode(batch, klen, 64, 8, 128, warmup, iters);
    }
}

//! Paged attention CUDA kernels for candle.
//!
//! Provides two operations:
//! - `paged_attention`: Compute attention from query against a paged KV cache (decode step)
//! - `reshape_and_cache`: Scatter-write K/V tokens into a paged KV cache pool

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use core::ffi::{c_int, c_long, c_void};

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn paged_attention_v1(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        dtype: u32,
        softcapping: f32,
        stream: i64,
    );

    fn paged_attention_v2(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,
        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        dtype: u32,
        softcapping: f32,
        stream: i64,
    );

    fn call_reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        slot_mapping: *const c_long,
        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        dtype: u32,
        stream: i64,
    );

    fn call_reshape_and_cache_flash(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        slot_mapping: *const c_long,
        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        key_stride: c_int,
        value_stride: c_int,
        dtype: u32,
        stream: i64,
    );
}

// ---------------------------------------------------------------------------
// dtype → integer code used by the CUDA kernels
// ---------------------------------------------------------------------------

fn dtype_to_internal(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        dt => candle_core::bail!("paged-attention: unsupported dtype {dt:?}"),
    }
}

// ---------------------------------------------------------------------------
// PagedAttention (CustomOp1)
// ---------------------------------------------------------------------------

struct PagedAttention {
    softmax_scale: f32,
    softcapping: f32,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    max_context_len: usize,
}

impl PagedAttention {
    fn cuda_fwd_t<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dtype = q.dtype();
        let internal_type = dtype_to_internal(dtype)?;
        let dev = q.device();
        let stream = dev.cuda_stream();
        let out_shape = q_l.shape().clone();

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Cuda(kc) => kc,
            _ => candle_core::bail!("key_cache must be a cuda tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Cuda(vc) => vc,
            _ => candle_core::bail!("value_cache must be a cuda tensor"),
        };

        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = match &*bt {
            Storage::Cuda(bt) => bt,
            _ => candle_core::bail!("block_tables must be a cuda tensor"),
        };

        let (cl, cl_l) = self.context_lens.storage_and_layout();
        let cl = match &*cl {
            Storage::Cuda(cl) => cl,
            _ => candle_core::bail!("context_lens must be a cuda tensor"),
        };

        // Validate ranks
        if q_l.stride().len() != 3 {
            candle_core::bail!("paged-attention: q must be rank 3, got {q_l:?}");
        }
        if kc_l.stride().len() != 5 {
            candle_core::bail!("paged-attention: key_cache must be rank 5, got {kc_l:?}");
        }
        if vc_l.stride().len() != 4 {
            candle_core::bail!("paged-attention: value_cache must be rank 4, got {vc_l:?}");
        }

        // Get cuda slices and apply offsets
        let q_s = q.as_cuda_slice::<T>()?;
        let kc_s = kc.as_cuda_slice::<T>()?;
        let vc_s = vc.as_cuda_slice::<T>()?;
        let bt_s = bt.as_cuda_slice::<u32>()?;
        let cl_s = cl.as_cuda_slice::<u32>()?;

        let q_v = q_s.slice(q_l.start_offset()..);
        let kc_v = kc_s.slice(kc_l.start_offset()..);
        let vc_v = vc_s.slice(vc_l.start_offset()..);
        let bt_v = bt_s.slice(bt_l.start_offset()..);
        let cl_v = cl_s.slice(cl_l.start_offset()..);

        let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;
        let (_, max_num_blocks_per_seq) = bt_l.shape().dims2()?;
        let (_num_blocks, num_kv_heads, _head_size_kc, block_size, _x) = kc_l.shape().dims5()?;

        let q_stride = q_l.stride()[0];
        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];

        let partition_size = 512;
        let max_num_partitions = self.max_context_len.div_ceil(partition_size);
        let use_v1 = (max_num_partitions == 1 || num_seqs * num_heads > 512)
            && partition_size % block_size == 0;

        let elem_count = out_shape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }?;

        let stream_raw = stream.cu_stream() as i64;

        // Scope the device_ptr guards so they drop before we move `out`.
        // All kernel launches are on the same CUDA stream, so the pointers
        // remain valid for the duration of the kernel execution.
        {
            let (out_ptr, _g_out) = out.device_ptr(&stream);
            let (q_ptr, _g_q) = q_v.device_ptr(&stream);
            let (kc_ptr, _g_kc) = kc_v.device_ptr(&stream);
            let (vc_ptr, _g_vc) = vc_v.device_ptr(&stream);
            let (bt_ptr, _g_bt) = bt_v.device_ptr(&stream);
            let (cl_ptr, _g_cl) = cl_v.device_ptr(&stream);

            if use_v1 {
                unsafe {
                    paged_attention_v1(
                        out_ptr as *const c_void,
                        q_ptr as *const c_void,
                        kc_ptr as *const c_void,
                        vc_ptr as *const c_void,
                        num_kv_heads as c_int,
                        self.softmax_scale,
                        bt_ptr as *const c_int,
                        cl_ptr as *const c_int,
                        block_size as c_int,
                        self.max_context_len as c_int,
                        num_seqs as c_int,
                        num_heads as c_int,
                        head_size as c_int,
                        max_num_blocks_per_seq as c_int,
                        q_stride as c_int,
                        kv_block_stride as c_int,
                        kv_head_stride as c_int,
                        internal_type,
                        self.softcapping,
                        stream_raw,
                    )
                }
            } else {
                let tmp_out = unsafe {
                    dev.alloc::<T>(num_seqs * num_heads * max_num_partitions * head_size)
                }?;
                let exp_sums =
                    unsafe { dev.alloc::<f32>(num_seqs * num_heads * max_num_partitions) }?;
                let max_logits =
                    unsafe { dev.alloc::<f32>(num_seqs * num_heads * max_num_partitions) }?;

                let (tmp_out_ptr, _g_tmp) = tmp_out.device_ptr(&stream);
                let (exp_sums_ptr, _g_es) = exp_sums.device_ptr(&stream);
                let (max_logits_ptr, _g_ml) = max_logits.device_ptr(&stream);

                unsafe {
                    paged_attention_v2(
                        out_ptr as *const c_void,
                        exp_sums_ptr as *const f32,
                        max_logits_ptr as *const f32,
                        tmp_out_ptr as *const c_void,
                        q_ptr as *const c_void,
                        kc_ptr as *const c_void,
                        vc_ptr as *const c_void,
                        num_kv_heads as c_int,
                        self.softmax_scale,
                        bt_ptr as *const c_int,
                        cl_ptr as *const c_int,
                        block_size as c_int,
                        self.max_context_len as c_int,
                        num_seqs as c_int,
                        num_heads as c_int,
                        head_size as c_int,
                        max_num_blocks_per_seq as c_int,
                        q_stride as c_int,
                        kv_block_stride as c_int,
                        kv_head_stride as c_int,
                        internal_type,
                        self.softcapping,
                        stream_raw,
                    )
                }
            }
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, out_shape))
    }
}

impl candle_core::CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for paged-attention")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            DType::F16 => self.cuda_fwd_t::<half::f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_t::<half::bf16>(q, q_l),
            dt => candle_core::bail!("paged-attention: unsupported dtype {dt:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ReshapeCache (InplaceOp1)
// ---------------------------------------------------------------------------

struct ReshapeCache {
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
}

impl ReshapeCache {
    fn cuda_fwd_t<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        k: &CudaStorage,
        k_l: &Layout,
    ) -> Result<()> {
        let dtype = k.dtype();
        let dev = k.device();
        let stream = dev.cuda_stream();
        let internal_type = dtype_to_internal(dtype)?;

        let (v, v_l) = self.value.storage_and_layout();
        let v = match &*v {
            Storage::Cuda(v) => v,
            _ => candle_core::bail!("value must be a cuda tensor"),
        };

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Cuda(kc) => kc,
            _ => candle_core::bail!("key_cache must be a cuda tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Cuda(vc) => vc,
            _ => candle_core::bail!("value_cache must be a cuda tensor"),
        };

        let (s, s_l) = self.slot_mapping.storage_and_layout();
        let s = match &*s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("slot_mapping must be a cuda tensor"),
        };

        // Validate ranks
        if k_l.stride().len() != 3 || v_l.stride().len() != 3 {
            candle_core::bail!(
                "reshape-cache: key and value must be rank 3 (k: {k_l:?}, v: {v_l:?})"
            );
        }
        if kc_l.stride().len() != 5 {
            candle_core::bail!("reshape-cache: key_cache must be rank 5, got {kc_l:?}");
        }
        if vc_l.stride().len() != 4 {
            candle_core::bail!("reshape-cache: value_cache must be rank 4, got {vc_l:?}");
        }

        // Get cuda slices and apply offsets
        let k_s = k.as_cuda_slice::<T>()?;
        let v_s = v.as_cuda_slice::<T>()?;
        let kc_s = kc.as_cuda_slice::<T>()?;
        let vc_s = vc.as_cuda_slice::<T>()?;
        let s_s = s.as_cuda_slice::<i64>()?;

        let k_v = k_s.slice(k_l.start_offset()..);
        let v_v = v_s.slice(v_l.start_offset()..);
        let kc_v = kc_s.slice(kc_l.start_offset()..);
        let vc_v = vc_s.slice(vc_l.start_offset()..);
        let s_v = s_s.slice(s_l.start_offset()..);

        let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
        let (_, _, _, block_size, x) = kc_l.shape().dims5()?;

        let key_stride = k_l.stride()[0] as c_int;
        let value_stride = v_l.stride()[0] as c_int;

        // Get raw pointers with stream guards
        let (k_ptr, _g_k) = k_v.device_ptr(&stream);
        let (v_ptr, _g_v) = v_v.device_ptr(&stream);
        let (kc_ptr, _g_kc) = kc_v.device_ptr(&stream);
        let (vc_ptr, _g_vc) = vc_v.device_ptr(&stream);
        let (s_ptr, _g_s) = s_v.device_ptr(&stream);

        let stream_raw = stream.cu_stream() as i64;

        unsafe {
            call_reshape_and_cache(
                k_ptr as *const c_void,
                v_ptr as *const c_void,
                kc_ptr as *const c_void,
                vc_ptr as *const c_void,
                s_ptr as *const c_long,
                num_tokens as c_int,
                num_heads as c_int,
                head_size as c_int,
                block_size as c_int,
                x as c_int,
                key_stride,
                value_stride,
                internal_type,
                stream_raw,
            )
        }
        Ok(())
    }
}

impl candle_core::InplaceOp1 for ReshapeCache {
    fn name(&self) -> &'static str {
        "reshape-cache"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("no cpu support for reshape-cache")
    }

    fn cuda_fwd(&self, k: &mut CudaStorage, k_l: &Layout) -> Result<()> {
        match k.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(k, k_l),
            DType::F16 => self.cuda_fwd_t::<half::f16>(k, k_l),
            DType::BF16 => self.cuda_fwd_t::<half::bf16>(k, k_l),
            dt => candle_core::bail!("reshape-cache: unsupported dtype {dt:?}"),
        }
    }
}

struct ReshapeCacheFlash {
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
}

impl ReshapeCacheFlash {
    fn cuda_fwd_t<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        k: &CudaStorage,
        k_l: &Layout,
    ) -> Result<()> {
        let dtype = k.dtype();
        let dev = k.device();
        let stream = dev.cuda_stream();
        let internal_type = dtype_to_internal(dtype)?;

        let (v, v_l) = self.value.storage_and_layout();
        let v = match &*v {
            Storage::Cuda(v) => v,
            _ => candle_core::bail!("value must be a cuda tensor"),
        };

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Cuda(kc) => kc,
            _ => candle_core::bail!("key_cache must be a cuda tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Cuda(vc) => vc,
            _ => candle_core::bail!("value_cache must be a cuda tensor"),
        };

        let (s, s_l) = self.slot_mapping.storage_and_layout();
        let s = match &*s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("slot_mapping must be a cuda tensor"),
        };

        if k_l.stride().len() != 3 || v_l.stride().len() != 3 {
            candle_core::bail!(
                "reshape-cache-flash: key and value must be rank 3 (k: {k_l:?}, v: {v_l:?})"
            );
        }
        if kc_l.stride().len() != 4 || vc_l.stride().len() != 4 {
            candle_core::bail!(
                "reshape-cache-flash: key_cache/value_cache must be rank 4 (k: {kc_l:?}, v: {vc_l:?})"
            );
        }

        let (_num_blocks_k, block_size_k, num_heads_k, head_size_k) = kc_l.shape().dims4()?;
        let (_num_blocks_v, block_size_v, num_heads_v, head_size_v) = vc_l.shape().dims4()?;
        if block_size_k != block_size_v || num_heads_k != num_heads_v || head_size_k != head_size_v
        {
            candle_core::bail!(
                "reshape-cache-flash: key/value cache shape mismatch (k: {:?}, v: {:?})",
                kc_l.shape(),
                vc_l.shape()
            );
        }

        let k_s = k.as_cuda_slice::<T>()?;
        let v_s = v.as_cuda_slice::<T>()?;
        let kc_s = kc.as_cuda_slice::<T>()?;
        let vc_s = vc.as_cuda_slice::<T>()?;
        let s_s = s.as_cuda_slice::<i64>()?;

        let k_v = k_s.slice(k_l.start_offset()..);
        let v_v = v_s.slice(v_l.start_offset()..);
        let kc_v = kc_s.slice(kc_l.start_offset()..);
        let vc_v = vc_s.slice(vc_l.start_offset()..);
        let s_v = s_s.slice(s_l.start_offset()..);

        let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
        if num_heads != num_heads_k || head_size != head_size_k {
            candle_core::bail!(
                "reshape-cache-flash: input/cache head dims mismatch (input: [{num_heads},{head_size}] cache: [{num_heads_k},{head_size_k}])"
            );
        }

        let key_stride = k_l.stride()[0] as c_int;
        let value_stride = v_l.stride()[0] as c_int;

        let (k_ptr, _g_k) = k_v.device_ptr(&stream);
        let (v_ptr, _g_v) = v_v.device_ptr(&stream);
        let (kc_ptr, _g_kc) = kc_v.device_ptr(&stream);
        let (vc_ptr, _g_vc) = vc_v.device_ptr(&stream);
        let (s_ptr, _g_s) = s_v.device_ptr(&stream);
        let stream_raw = stream.cu_stream() as i64;

        unsafe {
            call_reshape_and_cache_flash(
                k_ptr as *const c_void,
                v_ptr as *const c_void,
                kc_ptr as *const c_void,
                vc_ptr as *const c_void,
                s_ptr as *const c_long,
                num_tokens as c_int,
                num_heads as c_int,
                head_size as c_int,
                block_size_k as c_int,
                key_stride,
                value_stride,
                internal_type,
                stream_raw,
            )
        }
        Ok(())
    }
}

impl candle_core::InplaceOp1 for ReshapeCacheFlash {
    fn name(&self) -> &'static str {
        "reshape-cache-flash"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("no cpu support for reshape-cache-flash")
    }

    fn cuda_fwd(&self, k: &mut CudaStorage, k_l: &Layout) -> Result<()> {
        match k.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(k, k_l),
            DType::F16 => self.cuda_fwd_t::<half::f16>(k, k_l),
            DType::BF16 => self.cuda_fwd_t::<half::bf16>(k, k_l),
            dt => candle_core::bail!("reshape-cache-flash: unsupported dtype {dt:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute paged attention: `softmax(Q @ K^T * scale) @ V` reading K/V from a paged cache.
///
/// # Arguments
/// * `q` - Query tensor `[num_seqs, num_heads, head_size]`
/// * `key_cache` - `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
/// * `value_cache` - `[num_blocks, num_kv_heads, head_size, block_size]`
/// * `block_tables` - `[num_seqs, max_blocks_per_seq]` (u32)
/// * `context_lens` - `[num_seqs]` (u32)
/// * `max_context_len` - maximum context length across all sequences
/// * `softmax_scale` - typically `1/sqrt(head_size)`
///
/// Returns tensor of shape `[num_seqs, num_heads, head_size]`.
pub fn paged_attention(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    max_context_len: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let op = PagedAttention {
        softmax_scale,
        softcapping: 1.0, // 1.0 disables softcapping (kernel checks != 1.0)
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        context_lens: context_lens.clone(),
        max_context_len,
    };
    q.apply_op1(op)
}

/// Scatter-write key and value tokens into a paged KV cache.
///
/// # Arguments
/// * `key` - `[num_tokens, num_heads, head_size]`
/// * `value` - `[num_tokens, num_heads, head_size]`
/// * `key_cache` - `[num_blocks, num_heads, head_size/x, block_size, x]`
/// * `value_cache` - `[num_blocks, num_heads, head_size, block_size]`
/// * `slot_mapping` - `[num_tokens]` (i64) — slot index for each token
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let op = ReshapeCache {
        value: value.clone(),
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        slot_mapping: slot_mapping.clone(),
    };
    key.inplace_op1(&op)
}

/// Scatter-write key and value tokens into a flash-friendly paged KV cache.
///
/// # Arguments
/// * `key` - `[num_tokens, num_heads, head_size]`
/// * `value` - `[num_tokens, num_heads, head_size]`
/// * `key_cache` - `[num_blocks, block_size, num_heads, head_size]`
/// * `value_cache` - `[num_blocks, block_size, num_heads, head_size]`
/// * `slot_mapping` - `[num_tokens]` (i64) — slot index for each token
pub fn reshape_and_cache_flash(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let op = ReshapeCacheFlash {
        value: value.clone(),
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        slot_mapping: slot_mapping.clone(),
    };
    key.inplace_op1(&op)
}

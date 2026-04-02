use super::{MOD_ADD, MOD_SILU_MUL, PTX_ADD, PTX_SILU_MUL};
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, DType, Layout, Result, Shape, Tensor};

// ── Fused SiLU(gate) * up ──────────────────────────────────────────
struct FusedSiluMul;

impl candle_core::CustomOp2 for FusedSiluMul {
    fn name(&self) -> &'static str {
        "fused_silu_mul"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused_silu_mul: no CPU impl")
    }

    fn cuda_fwd(
        &self,
        gate_s: &candle_core::CudaStorage,
        gate_l: &Layout,
        up_s: &candle_core::CudaStorage,
        up_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = gate_s.device();
        let n = gate_l.shape().elem_count();
        if n != up_l.shape().elem_count() {
            candle_core::bail!("fused_silu_mul: shape mismatch");
        }
        if gate_s.dtype() != DType::BF16 || up_s.dtype() != DType::BF16 {
            candle_core::bail!("fused_silu_mul: requires BF16");
        }

        let gate = gate_s.as_cuda_slice::<half::bf16>()?;
        let up = up_s.as_cuda_slice::<half::bf16>()?;
        let gate = gate.slice(gate_l.start_offset()..);
        let up = up.slice(up_l.start_offset()..);

        let out = unsafe { dev.alloc::<half::bf16>(n) }?;

        let threads = 256u32;
        let elems_per_thread = 8u32;
        let blocks = (n as u32 + threads * elems_per_thread - 1) / (threads * elems_per_thread);

        let func =
            dev.get_or_load_custom_func("fused_silu_mul_bf16", MOD_SILU_MUL, PTX_SILU_MUL)?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&gate);
        builder.arg(&up);
        builder.arg(&out);
        let n_val = n as u32;
        builder.arg(&n_val);
        unsafe { builder.launch(cfg) }.w()?;

        let out_shape = gate_l.shape().clone();
        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out_storage, out_shape))
    }
}

// ── Vectorized BF16 Add ────────────────────────────────────────────
struct VectorizedAdd;

impl candle_core::CustomOp2 for VectorizedAdd {
    fn name(&self) -> &'static str {
        "vectorized_add"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("vectorized_add: no CPU impl")
    }

    fn cuda_fwd(
        &self,
        a_s: &candle_core::CudaStorage,
        a_l: &Layout,
        b_s: &candle_core::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = a_s.device();
        let n = a_l.shape().elem_count();
        if n != b_l.shape().elem_count() {
            candle_core::bail!("vectorized_add: shape mismatch");
        }
        if a_s.dtype() != DType::BF16 || b_s.dtype() != DType::BF16 {
            candle_core::bail!("vectorized_add: requires BF16");
        }

        let a = a_s.as_cuda_slice::<half::bf16>()?;
        let b = b_s.as_cuda_slice::<half::bf16>()?;
        let a = a.slice(a_l.start_offset()..);
        let b = b.slice(b_l.start_offset()..);

        let out = unsafe { dev.alloc::<half::bf16>(n) }?;

        let threads = 256u32;
        let elems_per_thread = 8u32;
        let blocks = (n as u32 + threads * elems_per_thread - 1) / (threads * elems_per_thread);

        let func = dev.get_or_load_custom_func("vectorized_add_bf16", MOD_ADD, PTX_ADD)?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&a);
        builder.arg(&b);
        builder.arg(&out);
        let n_val = n as u32;
        builder.arg(&n_val);
        unsafe { builder.launch(cfg) }.w()?;

        let out_shape = a_l.shape().clone();
        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out_storage, out_shape))
    }
}

/// Fused SiLU(gate) * up. Both inputs must be contiguous BF16 CUDA tensors.
pub fn fused_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.apply_op2(up, FusedSiluMul)
}

/// Vectorized BF16 addition. Both inputs must be contiguous BF16 CUDA tensors.
pub fn vectorized_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.apply_op2(b, VectorizedAdd)
}

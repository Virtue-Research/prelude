//! CUTLASS Fused MoE runner — wraps FlashInfer's CUTLASS backend.
//!
//! Provides a Rust-callable MoE forward pass that fuses:
//!   routing (permute) → GEMM1 (gate_up) → Swiglu → GEMM2 (down) → unpermute
//! into a single optimized CUTLASS kernel pipeline.
//!
//! Usage:
//!   let runner = FusedMoeRunner::new()?;
//!   runner.run_moe(&output, &input, &expert_ids, &expert_weights,
//!                  &w1, &w2, num_experts, top_k, ...)?;

use crate::loader::KernelRegistry;
use crate::types::*;
use std::sync::OnceLock;
use tvm_static_ffi::TVMModuleHandle;

/// CUTLASS-based fused MoE runner.
/// Wraps a TVM Module that holds the CUTLASS MoeFCRunner internally.
pub struct FusedMoeRunner {
    module: TVMModuleHandle,
}

/// Swiglu activation type (matches CUTLASS MoE ActivationType enum).
const ACTIVATION_SWIGLU: i64 = 3;

impl FusedMoeRunner {
    /// Create a new FusedMoeRunner for BF16 computation.
    ///
    /// Looks up the `init` TVM-FFI symbol from the kernel registry and
    /// creates the CUTLASS MoeFCRunner with BF16 activation/weight/output types.
    pub fn new() -> Result<Self, String> {
        // Register CUDA allocator for TVM-FFI internal workspace allocation
        // (needed by CUTLASS MoE runner for temporary GPU buffers).
        static ALLOCATOR_INIT: std::sync::Once = std::sync::Once::new();
        ALLOCATOR_INIT.call_once(|| {
            let rc = unsafe { tvm_static_ffi::tvm_static_ffi_register_cuda_allocator() };
            if rc != 0 {
                tracing::error!("Failed to register CUDA allocator for TVM-FFI (rc={rc})");
            }
        });

        let registry = get_registry();
        let init_fn = registry
            .get_utility("init")
            .ok_or("CUTLASS fused MoE kernel not compiled (missing 'init' symbol)")?;

        // init(activation_dtype, weight_dtype, output_dtype,
        //      use_deepseek_fp8, use_w4_group_scaling, use_mxfp8_act_scaling, use_packed_weights)
        // DLDataType for BF16: code=4 (kDLBfloat), bits=16, lanes=1
        let bf16_dtype = TVMFFIAny::dlpack_dtype(DLDataType {
            code: KDLBFLOAT,
            bits: 16,
            lanes: 1,
        });

        let args = [
            bf16_dtype,                 // activation_dtype
            bf16_dtype,                 // weight_dtype
            bf16_dtype,                 // output_dtype
            TVMFFIAny::bool_val(false), // use_deepseek_fp8_block_scale
            TVMFFIAny::bool_val(false), // use_w4_group_scaling
            TVMFFIAny::bool_val(false), // use_mxfp8_act_scaling
            TVMFFIAny::bool_val(false), // use_packed_weights
        ];

        let module = unsafe { tvm_static_ffi::call_tvm_ffi_module(init_fn, &args) }?;
        tracing::info!("CUTLASS fused MoE runner created (BF16)");
        Ok(Self { module })
    }

    /// Run fused MoE forward pass.
    ///
    /// Performs: routing + permute → GEMM1 (gate_up) → Swiglu → GEMM2 (down) → unpermute
    /// in a fused CUTLASS kernel pipeline.
    ///
    /// # Arguments
    /// - `output`: [num_tokens, hidden_size] BF16, pre-allocated
    /// - `input`: [num_tokens, hidden_size] BF16
    /// - `token_selected_experts`: [num_tokens, top_k] I32 (expert IDs per token)
    /// - `token_final_scales`: [num_tokens, top_k] F32 (routing weights)
    /// - `w1`: [num_experts, 2*intermediate_size, hidden_size] BF16 (gate_up fused)
    /// - `w2`: [num_experts, hidden_size, intermediate_size] BF16 (down)
    ///
    /// # Safety
    /// All tensor data pointers must be valid CUDA device pointers on the current device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_moe(
        &self,
        output: &DLTensor,
        input: &DLTensor,
        token_selected_experts: &DLTensor,
        token_final_scales: &DLTensor,
        w1: &DLTensor,
        w2: &DLTensor,
        tp_size: i64,
        tp_rank: i64,
        ep_size: i64,
        ep_rank: i64,
    ) -> Result<(), String> {
        let args = [
            TVMFFIAny::dltensor(output),                 // output
            TVMFFIAny::dltensor(input),                  // input
            TVMFFIAny::dltensor(token_selected_experts), // token_selected_experts
            TVMFFIAny::dltensor(token_final_scales),     // token_final_scales (Optional: present)
            TVMFFIAny::dltensor(w1),                     // fc1_expert_weights
            TVMFFIAny::none(),                           // fc1_expert_biases (None)
            TVMFFIAny::dltensor(w2),                     // fc2_expert_weights
            TVMFFIAny::none(),                           // fc2_expert_biases (None)
            TVMFFIAny::none(),                           // quant_scales (None)
            TVMFFIAny::none(),                           // input_sf (None)
            TVMFFIAny::none(),                           // swiglu_alpha (None)
            TVMFFIAny::none(),                           // swiglu_beta (None)
            TVMFFIAny::none(),                           // swiglu_limit (None)
            TVMFFIAny::bool_val(false),                  // swizzled_input_sf
            TVMFFIAny::int64(tp_size),                   // tp_size
            TVMFFIAny::int64(tp_rank),                   // tp_rank
            TVMFFIAny::int64(ep_size),                   // ep_size
            TVMFFIAny::int64(ep_rank),                   // ep_rank
            TVMFFIAny::int64(1),                         // cluster_size
            TVMFFIAny::int64(0),                         // cluster_rank
            TVMFFIAny::bool_val(false),                  // enable_alltoall
            TVMFFIAny::bool_val(false),                  // min_latency_mode
            TVMFFIAny::none(),                           // profile_ids (None = auto)
            TVMFFIAny::bool_val(false),                  // enable_pdl
            TVMFFIAny::int64(ACTIVATION_SWIGLU),         // base_activation_type
        ];

        unsafe { self.module.call(b"run_moe\0", &args) }?;
        Ok(())
    }
}

// SAFETY: FusedMoeRunner is internally mutex-protected (CutlassMoeFCRunner).
unsafe impl Send for FusedMoeRunner {}
unsafe impl Sync for FusedMoeRunner {}

/// Get the FlashInfer kernel registry (cached singleton).
fn get_registry() -> &'static KernelRegistry {
    static REGISTRY: OnceLock<KernelRegistry> = OnceLock::new();
    REGISTRY.get_or_init(KernelRegistry::new)
}

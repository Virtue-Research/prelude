//! Mixture-of-Experts layer.
//!
//! Shared MoE module composing router + expert FFN dispatch.
//! Supports Local, ExpertParallel, and Disaggregated execution modes.
//!
//! Models with MoE (Qwen3-MoE, DeepSeek-V3) use this module instead of
//! a dense MLP. The module calls `ops.gemm.grouped_gemm()` for batched
//! expert execution and `ops.comm` for expert-parallel communication.
//!
//! Not yet extracted from model code — placeholder for the shared module.

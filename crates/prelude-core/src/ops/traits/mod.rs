//! Op trait definitions — the shared contract between models and device implementations.
//!
//! Models call these traits. Device crates (prelude-cuda, prelude-rocm, etc.) implement them.
//! prelude-core defines the traits but implements none of the GPU paths.

mod activation;
mod attention;
mod bundle;
mod comm;
mod conv;
mod fused;
mod gemm;
mod kv_cache;
mod norm;
mod session;
mod tensor_ops;

pub use activation::ActivationOps;
pub use attention::{AttentionOps, MaskType, VarlenParams, PagedParams};
pub use bundle::Ops;
pub use comm::CommOps;
pub use conv::ConvOps;
pub use fused::FusedOps;
pub use gemm::{GemmOps, QuantScheme};
pub use kv_cache::{CacheSlotSpec, KvCacheOps};
pub use norm::NormOps;
pub use session::OpsSession;
pub use tensor_ops::{TensorOps, UnaryOp, BinaryOp, CompareOp, ReduceOp};

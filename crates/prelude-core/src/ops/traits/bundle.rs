use std::sync::Arc;

use super::{
    ActivationOps, AttentionOps, CommOps, ConvOps, FusedOps, GemmOps, KvCacheOps, NormOps,
    OpsSession, TensorOps,
};

/// The Ops bundle — models receive this via dependency injection.
///
/// All fields are always present. Devices that don't support an op category
/// return errors from those methods, not panics.
pub struct Ops {
    pub attn: Arc<dyn AttentionOps>,
    pub kv_cache: Arc<dyn KvCacheOps>,
    pub gemm: Arc<dyn GemmOps>,
    pub norm: Arc<dyn NormOps>,
    pub act: Arc<dyn ActivationOps>,
    pub conv: Arc<dyn ConvOps>,
    pub comm: Arc<dyn CommOps>,
    pub fused: Arc<dyn FusedOps>,
    pub session: Arc<dyn OpsSession>,
    pub tensor: Arc<dyn TensorOps>,
}

//! Integration tests for the priority/probe backend registry.
//!
//! Each integration test file is a separate binary, so OnceLock state is clean.
//! This lets us test `register_backend` → `select_ops` end-to-end.

use prelude_core::ops::{self, OpsBackend};
use prelude_core::tensor::Device;

// -- Distinguishable Ops impls (all methods inherited via defaults) --------

struct HighPriorityOps;
impl ops::traits::Ops for HighPriorityOps {
    fn default_impl(&self) -> &dyn ops::traits::Ops { ops::bare_ops() }
    fn attn_name(&self) -> &str { "high_priority" }
}

struct LowPriorityOps;
impl ops::traits::Ops for LowPriorityOps {
    fn default_impl(&self) -> &dyn ops::traits::Ops { ops::bare_ops() }
    fn attn_name(&self) -> &str { "low_priority" }
}

static HIGH: HighPriorityOps = HighPriorityOps;
static LOW: LowPriorityOps = LowPriorityOps;

// One test function because OnceLock resolves only once per process.
#[test]
fn priority_probe_and_device_matching() {
    // Register CPU backends with different priorities.
    ops::register_backend(OpsBackend {
        name: "low_cpu",
        priority: 10,
        probe: || true,
        supports: |d| d.is_cpu(),
        create_ops: || &LOW,
    });
    ops::register_backend(OpsBackend {
        name: "high_cpu",
        priority: 100,
        probe: || true,
        supports: |d| d.is_cpu(),
        create_ops: || &HIGH,
    });

    // Register a GPU backend whose probe fails.
    ops::register_backend(OpsBackend {
        name: "fake_gpu",
        priority: 200,
        probe: || false,
        supports: |d| d.is_cuda(),
        create_ops: || &HIGH,
    });

    // CPU: should pick high_cpu (priority 100 > 10).
    let cpu_ops = ops::select_ops(&Device::Cpu);
    assert_eq!(
        cpu_ops.attn_name(), "high_priority",
        "should select highest-priority CPU backend"
    );

    // GPU: fake_gpu probe fails, no other GPU backend → bare_ops fallback.
    let gpu_ops = ops::select_ops(&Device::Cuda(0));
    assert_eq!(
        gpu_ops.attn_name(), "default",
        "should fall back to bare_ops when GPU probe fails"
    );

    // Calling again should return the same cached result.
    let cpu_ops2 = ops::select_ops(&Device::Cpu);
    assert_eq!(cpu_ops2.attn_name(), "high_priority", "cached result should be stable");
}

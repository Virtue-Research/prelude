//! AMD GPU device implementation for Prelude inference engine.
//!
//! This crate owns:
//! - KFD bare-metal runtime via t0-gpu (zero HIP/ROCm dependency)
//! - GPU arch detection (RDNA3, CDNA3, etc.) via sysfs topology
//! - AmdOps: implements Ops trait + TensorHook overrides from prelude-core
//!
//! Supported architectures:
//! - GFX1100 (RDNA3, RX 7900 XTX): via t0-gpu's BlockDSL/TileIR kernels
//! - GFX942 (CDNA3, MI300X): planned

mod amd_ops;
mod device;
pub mod storage;

/// Register AMD ops and executor. Call once at startup.
pub fn register() {
    if !device::amd_probe() {
        tracing::debug!("AMD GPU not detected, skipping registration");
        return;
    }

    let arch = device::detect_arch();
    tracing::info!("AMD GPU detected: {:?}", arch);

    prelude_core::ops::register_backend(prelude_core::ops::OpsBackend {
        name: "amd",
        priority: 100,
        probe: device::amd_probe,
        supports: |d| d.is_custom(),
        create_ops: amd_ops::amd_ops,
    });
}

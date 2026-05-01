//! Kernel operations — unified `Ops` trait + device backends.
//!
//! - `traits/ops.rs`: The unified `Ops` trait — one trait, override any method.
//! - Device crates register via `register_backend()` with priority/probe at startup.
//! - Basic tensor ops (matmul, unary, binary, etc.) go through candle-core natively.
//!   The Ops trait only covers fused/inference-specific ops.

pub mod traits;
mod hook_bridge;

pub use traits::*;

/// Get a reference to the TensorHook bridge singleton.
/// Used by tests to verify the hook mechanism.
pub fn hook_bridge_ref() -> &'static dyn candle_core::TensorHook {
    &hook_bridge::BRIDGE
}

// ── Device ops registry ────────────────────────────────────────────

use std::sync::{Mutex, OnceLock};
use crate::tensor::Device;

/// A registered ops backend with priority-based selection.
pub struct OpsBackend {
    /// Human-readable name (e.g. "cuda", "cpu", "rocm").
    pub name: &'static str,
    /// Higher priority wins when multiple backends match a device.
    pub priority: u32,
    /// Runtime probe: returns `true` if this backend is usable on the current machine
    /// (e.g. CUDA driver present, ROCm stack installed).
    pub probe: fn() -> bool,
    /// Returns `true` if this backend handles the given device kind.
    pub supports: fn(&Device) -> bool,
    /// Factory that returns the singleton `&'static dyn Ops` for this backend.
    pub create_ops: fn() -> &'static dyn Ops,
}

static OPS_REGISTRY: Mutex<Vec<OpsBackend>> = Mutex::new(Vec::new());
static RESOLVED_CPU: OnceLock<&'static dyn Ops> = OnceLock::new();
static RESOLVED_GPU: OnceLock<&'static dyn Ops> = OnceLock::new();

/// Register an ops backend. Call during startup, before any tensor operations.
pub fn register_backend(entry: OpsBackend) {
    OPS_REGISTRY.lock().unwrap().push(entry);
}

/// Pick the best backend from a list: highest priority among those that
/// support the device and pass the probe. Returns `None` when nothing matches.
fn pick_best<'a>(backends: &'a [OpsBackend], device: &Device) -> Option<&'a OpsBackend> {
    let mut best: Option<&OpsBackend> = None;
    for b in backends {
        if (b.supports)(device) && (b.probe)() {
            if best.map_or(true, |cur| b.priority > cur.priority) {
                best = Some(b);
            }
        }
    }
    best
}

fn resolve_for(device: &Device) -> &'static dyn Ops {
    let backends = OPS_REGISTRY.lock().unwrap();
    match pick_best(&backends, device) {
        Some(b) => {
            tracing::info!("ops backend for {:?}: {} (priority {})", device, b.name, b.priority);
            (b.create_ops)()
        }
        None => {
            tracing::warn!("no ops backend for {:?}, falling back to bare ops", device);
            bare_ops()
        }
    }
}

pub fn select_ops(device: &Device) -> &'static dyn Ops {
    let lock = if device.is_cuda() { &RESOLVED_GPU } else { &RESOLVED_CPU };
    *lock.get_or_init(|| resolve_for(device))
}

/// Device-aware ops: thread-local first, then device-based fallback.
pub fn ops_for(device: &Device) -> &'static dyn Ops {
    THREAD_OPS.with(|cell| {
        if let Some(ops) = cell.get() {
            return ops;
        }
        select_ops(device)
    })
}

// ── Thread-local ops context ─────────────────────────────────────

use std::cell::Cell;

thread_local! {
    static THREAD_OPS: Cell<Option<&'static dyn Ops>> = const { Cell::new(None) };
}

/// Bare ops — minimal fallback with no device-specific kernels.
/// Used by device crates' `default_impl()` to avoid recursion.
/// Basic tensor ops go through candle-core; this only provides Ops trait defaults.
pub fn bare_ops() -> &'static dyn Ops {
    static BARE: BareOps = BareOps;
    &BARE
}

/// Minimal Ops implementation — all fused ops return None, all device-specific ops bail.
/// Basic tensor ops (matmul, add, etc.) are handled by candle-core natively.
struct BareOps;
impl Ops for BareOps {
    fn default_impl(&self) -> &dyn Ops { self }
}

pub fn with_ops<R>(ops: &'static dyn Ops, f: impl FnOnce() -> R) -> R {
    THREAD_OPS.with(|cell| {
        let prev = cell.replace(Some(ops));
        let result = f();
        cell.set(prev);
        result
    })
}

pub fn forward_scope<R>(ops: &'static dyn Ops, f: impl FnOnce() -> R) -> R {
    // Ensure TensorHook bridge is registered (once per thread).
    install_hook();
    with_ops(ops, || {
        ops.begin_forward();
        let result = f();
        ops.end_forward();
        result
    })
}

/// Install the TensorHook bridge if not already set on this thread.
/// This connects candle's tensor ops to prelude's Ops trait.
fn install_hook() {
    if candle_core::hook::get().is_none() {
        candle_core::hook::set(Some(&hook_bridge::BRIDGE));
    }
}

pub struct OpsGuard {
    prev: Option<&'static dyn Ops>,
}

impl OpsGuard {
    pub fn new(ops: &'static dyn Ops) -> Self {
        let prev = THREAD_OPS.with(|cell| cell.replace(Some(ops)));
        Self { prev }
    }
}

impl Drop for OpsGuard {
    fn drop(&mut self) {
        THREAD_OPS.with(|cell| cell.set(self.prev));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_backend(name: &'static str, priority: u32, probe_ok: bool) -> OpsBackend {
        OpsBackend {
            name,
            priority,
            probe: if probe_ok { || true } else { || false },
            supports: |d| d.is_cpu(),
            create_ops: bare_ops,
        }
    }

    fn gpu_backend(name: &'static str, priority: u32, probe_ok: bool) -> OpsBackend {
        OpsBackend {
            name,
            priority,
            probe: if probe_ok { || true } else { || false },
            supports: |d| d.is_cuda(),
            create_ops: bare_ops,
        }
    }

    #[test]
    fn highest_priority_wins() {
        let backends = vec![
            cpu_backend("low", 10, true),
            cpu_backend("high", 100, true),
            cpu_backend("mid", 50, true),
        ];
        let best = pick_best(&backends, &Device::Cpu).unwrap();
        assert_eq!(best.name, "high");
    }

    #[test]
    fn probe_false_skipped() {
        let backends = vec![
            cpu_backend("unavailable", 100, false),
            cpu_backend("available", 10, true),
        ];
        let best = pick_best(&backends, &Device::Cpu).unwrap();
        assert_eq!(best.name, "available");
    }

    #[test]
    fn device_mismatch_skipped() {
        let backends = vec![
            gpu_backend("cuda", 100, true),
        ];
        assert!(pick_best(&backends, &Device::Cpu).is_none());
    }

    #[test]
    fn empty_registry_returns_none() {
        let backends: Vec<OpsBackend> = vec![];
        assert!(pick_best(&backends, &Device::Cpu).is_none());
    }

    #[test]
    fn all_probes_fail_returns_none() {
        let backends = vec![
            cpu_backend("a", 100, false),
            cpu_backend("b", 50, false),
        ];
        assert!(pick_best(&backends, &Device::Cpu).is_none());
    }

    // Note: mixed_devices test removed — Device::Cuda requires a real CudaDevice
    // which can only be constructed with an actual GPU. The other tests cover the
    // priority/probe logic sufficiently.
}

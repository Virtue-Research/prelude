//! Kernel operations — unified `Ops` trait + device backend.
//!
//! - `traits/ops.rs`: The unified `Ops` trait — one trait, override any method.
//! - `device_backend/`: Pure Rust backend (Storage::Device).
//! - Device crates register via `register_backend()` with priority/probe at startup.

pub mod traits;
pub mod device_backend;

pub use traits::*;

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
            tracing::info!("ops backend for {device}: {} (priority {})", b.name, b.priority);
            (b.create_ops)()
        }
        None => {
            tracing::warn!("no ops backend for {device}, falling back to bare ops");
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

/// Bare primitives (Device backend), WITHOUT device crate overlay.
/// Used by device crates' `default_impl()` to avoid recursion.
pub fn bare_ops() -> &'static dyn Ops {
    device_backend::device_ops()
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
    with_ops(ops, || {
        ops.begin_forward();
        let result = f();
        ops.end_forward();
        result
    })
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

    #[test]
    fn mixed_devices() {
        let backends = vec![
            cpu_backend("cpu", 10, true),
            gpu_backend("cuda", 100, true),
        ];
        let cpu_best = pick_best(&backends, &Device::Cpu).unwrap();
        assert_eq!(cpu_best.name, "cpu");
        let gpu_best = pick_best(&backends, &Device::Cuda(0)).unwrap();
        assert_eq!(gpu_best.name, "cuda");
    }
}

//! AMD GPU detection via KFD sysfs topology.

use std::path::Path;

/// Detected GPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GfxArch {
    /// RDNA3 — RX 7900 XTX, etc. (t0-gpu supported)
    Gfx1100,
    /// CDNA3 — MI300X, MI300A (planned, via HSA runtime)
    Gfx942,
    /// Unknown architecture
    Unknown(u32),
}

impl GfxArch {
    /// Whether t0-gpu's bare-metal kernels are available for this arch.
    pub fn has_t0_support(&self) -> bool {
        matches!(self, GfxArch::Gfx1100)
    }
}

/// Check if an AMD GPU is available via KFD.
pub fn amd_probe() -> bool {
    Path::new("/dev/kfd").exists()
}

/// Detect the GPU architecture from KFD sysfs topology.
pub fn detect_arch() -> GfxArch {
    for node in 1..=8 {
        let props_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/properties", node);
        let gpu_id_path = format!("/sys/class/kfd/kfd/topology/nodes/{}/gpu_id", node);

        let is_gpu = std::fs::read_to_string(&gpu_id_path)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .is_some_and(|id| id > 0);
        if !is_gpu {
            continue;
        }

        if let Ok(props) = std::fs::read_to_string(&props_path) {
            for line in props.lines() {
                if let Some(val) = line.strip_prefix("gfx_target_version") {
                    let ver: u32 = val.trim().parse().unwrap_or(0);
                    return match ver {
                        110000 => GfxArch::Gfx1100,
                        90400 | 90402 => GfxArch::Gfx942,
                        v => GfxArch::Unknown(v),
                    };
                }
            }
        }
    }
    GfxArch::Unknown(0)
}

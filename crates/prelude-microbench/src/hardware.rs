use serde::Serialize;
use std::fs;

#[derive(Debug, Serialize, serde::Deserialize, Clone)]
pub struct HardwareInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub cpu_features: Vec<String>,
    pub gpu_model: Option<String>,
}

impl HardwareInfo {
    pub fn detect() -> Self {
        Self {
            cpu_model: detect_cpu_model(),
            cpu_cores: num_cpus(),
            cpu_features: detect_cpu_features(),
            gpu_model: detect_gpu(),
        }
    }

    pub fn print_header(&self) {
        println!("Hardware: {} ({} cores)", self.cpu_model, self.cpu_cores);
        if !self.cpu_features.is_empty() {
            println!("  CPU features: {}", self.cpu_features.join(", "));
        }
        if let Some(gpu) = &self.gpu_model {
            println!("  GPU: {gpu}");
        }
    }
}

fn detect_cpu_model() -> String {
    if let Ok(info) = fs::read_to_string("/proc/cpuinfo") {
        for line in info.lines() {
            if line.starts_with("model name") {
                if let Some(name) = line.split(':').nth(1) {
                    return name.trim().to_string();
                }
            }
        }
    }
    "unknown".into()
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(target_arch = "x86_64")]
fn detect_cpu_features() -> Vec<String> {
    let mut features = Vec::new();
    if is_x86_feature_detected!("avx2") { features.push("AVX2".into()); }
    if is_x86_feature_detected!("fma") { features.push("FMA".into()); }
    if is_x86_feature_detected!("avx512f") { features.push("AVX-512F".into()); }
    if is_x86_feature_detected!("avx512bw") { features.push("AVX-512BW".into()); }
    features
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_cpu_features() -> Vec<String> {
    Vec::new()
}

fn detect_gpu() -> Option<String> {
    // Try nvidia-smi
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if s.is_empty() { None } else { Some(s) }
            } else {
                None
            }
        })
}

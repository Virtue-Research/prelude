use crate::hardware::HardwareInfo;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize, serde::Deserialize)]
pub struct BenchReport {
    pub hardware: HardwareInfo,
    pub timestamp: String,
    pub results: Vec<BenchEntry>,
}

#[derive(Debug, Serialize, serde::Deserialize, Clone)]
pub struct BenchEntry {
    /// e.g. "cpu/quant/dot", "cpu/quant/matmul", "cpu/gemm"
    pub category: String,
    /// e.g. "Q4_K K=4096"
    pub name: String,
    /// Our result in microseconds
    pub ours_us: f64,
    /// Baseline name and result (e.g. "ggml" -> 0.15)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_us: Option<f64>,
    /// Extra info (dimensions, speedup, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl BenchReport {
    pub fn new(hardware: HardwareInfo) -> Self {
        Self {
            hardware,
            timestamp: chrono::Utc::now().to_rfc3339(),
            results: Vec::new(),
        }
    }

    pub fn add(&mut self, entry: BenchEntry) {
        self.results.push(entry);
    }

    pub fn save_json(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        eprintln!("Saved: {}", path.display());
        Ok(())
    }

    pub fn print_comparison(&self, baseline_path: &Path) -> std::io::Result<()> {
        let baseline_json = std::fs::read_to_string(baseline_path)?;
        let baseline: BenchReport = serde_json::from_str(&baseline_json)?;

        println!(
            "\n--- Comparison: {} vs {} ---",
            self.hardware.cpu_model, baseline.hardware.cpu_model
        );

        for ours in &self.results {
            if let Some(theirs) = baseline.results.iter().find(|b| {
                b.category == ours.category && b.name == ours.name
            }) {
                let speedup = theirs.ours_us / ours.ours_us;
                let marker = if speedup > 1.05 {
                    "FASTER"
                } else if speedup < 0.95 {
                    "SLOWER"
                } else {
                    "~same"
                };
                println!(
                    "  {}/{}: {:.2}us vs {:.2}us ({:.2}x) [{marker}]",
                    ours.category, ours.name, ours.ours_us, theirs.ours_us, speedup,
                );
            }
        }
        Ok(())
    }
}

/// Print a single benchmark result line.
pub fn print_result(category: &str, name: &str, us: f64, baseline: Option<(&str, f64)>) {
    if let Some((bl_name, bl_us)) = baseline {
        let ratio = bl_us / us;
        let marker = if ratio > 1.05 {
            format!("ours {ratio:.2}x faster")
        } else if ratio < 0.95 {
            format!("{bl_name} {:.2}x faster", us / bl_us)
        } else {
            "~same".into()
        };
        println!("  {category}/{name}: {us:.2}us  vs {bl_name}={bl_us:.2}us  ({marker})");
    } else {
        println!("  {category}/{name}: {us:.2}us");
    }
}

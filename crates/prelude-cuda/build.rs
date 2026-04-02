fn main() {
    // ── Compile custom CUDA kernels to PTX (loaded at runtime via cudarc) ──
    use std::path::PathBuf;
    use std::process::Command;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernels_dir = PathBuf::from(&manifest_dir).join("src/kernels");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Find nvcc
    let cuda_root = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = PathBuf::from(&cuda_root).join("bin/nvcc");

    // Detect GPU compute capability, default to sm_80 (Ampere)
    let compute_cap = detect_compute_cap().unwrap_or(80);
    let arch_flag = format!("-arch=sm_{compute_cap}");

    // Include path for common headers
    let include_flag = format!("-I{}", kernels_dir.display());

    // Define kernel modules: (category, filename, output_name)
    let kernel_modules = [
        ("elementwise", "add.cu", "add"),
        ("elementwise", "silu_mul.cu", "silu_mul"),
        ("normalization", "rmsnorm.cu", "rmsnorm"),
        ("normalization", "add_rmsnorm.cu", "add_rmsnorm"),
        ("rope", "qknorm_rope.cu", "qknorm_rope"),
        ("moe", "routing.cu", "moe_routing"),
        ("moe", "gateup.cu", "moe_gateup"),
        ("moe", "down.cu", "moe_down"),
        ("kvcache", "append.cu", "kv_append"),
        ("kvcache", "knorm_rope_kv_write.cu", "knorm_rope_kv_write"),
        ("kvcache", "scatter_kv_cache.cu", "scatter_kv_cache"),
    ];

    // Track common headers for rerun-if-changed
    println!("cargo:rerun-if-changed=src/kernels/common/common.cuh");
    println!("cargo:rerun-if-changed=src/kernels/common/vec_utils.cuh");

    for (category, filename, output_name) in kernel_modules.iter() {
        let kernel_src = kernels_dir.join(category).join(filename);
        let ptx_path = out_dir.join(format!("{}.ptx", output_name));

        println!("cargo:rerun-if-changed={}", kernel_src.display());

        let status = Command::new(&nvcc)
            .args([
                "--ptx",
                kernel_src.to_str().unwrap(),
                "-o",
                ptx_path.to_str().unwrap(),
                &arch_flag,
                &include_flag,
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
            ])
            .status()
            .unwrap_or_else(|e| panic!("Failed to run nvcc at {}: {}", nvcc.display(), e));

        assert!(
            status.success(),
            "nvcc PTX compilation of {} failed",
            filename
        );
    }
}

fn detect_compute_cap() -> Option<u32> {
    // Try CUDA_ARCH_LIST env var first, then nvidia-smi
    if let Ok(arch_list) = std::env::var("CUDA_ARCH_LIST") {
        if let Some(cap) = arch_list
            .split(',')
            .filter_map(|s| {
                let s = s.trim().replace('.', "");
                s.parse::<u32>().ok()
            })
            .max()
        {
            return Some(cap);
        }
    }

    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .next()?
        .trim()
        .replace('.', "")
        .parse::<u32>()
        .ok()
}

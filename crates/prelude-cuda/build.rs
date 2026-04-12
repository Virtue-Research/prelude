fn main() {
    // ── Compile custom CUDA kernels to PTX (loaded at runtime via cudarc) ──
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::Arc;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernels_dir = PathBuf::from(&manifest_dir).join("src/kernels/kernels_src");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Find nvcc
    let cuda_root = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = PathBuf::from(&cuda_root).join("bin/nvcc");

    // Detect GPU compute capability, default to sm_80 (Ampere)
    let compute_cap = detect_compute_cap().unwrap_or(80);
    let arch_flag = format!("-arch=sm_{compute_cap}");

    // Include paths for common headers (our own + candle's)
    let include_flag = format!("-I{}", kernels_dir.display());
    let candle_include_flag = format!("-I{}", kernels_dir.join("candle").display());

    // Define kernel modules: (category, filename, output_name)
    let kernel_modules = [
        // Our own optimized kernels
        ("elementwise", "add.cu", "add"),
        ("elementwise", "silu_mul.cu", "silu_mul"),
        ("normalization", "rmsnorm.cu", "rmsnorm"),
        ("rope", "qknorm_rope.cu", "qknorm_rope"),
        ("moe", "routing.cu", "moe_routing"),
        ("moe", "gateup.cu", "moe_gateup"),
        ("moe", "down.cu", "moe_down"),
        ("kvcache", "append.cu", "kv_append"),
        ("kvcache", "knorm_rope_kv_write.cu", "knorm_rope_kv_write"),
        ("kvcache", "scatter_kv_cache.cu", "scatter_kv_cache"),
        ("gdn", "post_conv.cu", "gdn_post_conv"),
        // General-purpose kernels (ported from candle-kernels for TensorOps)
        ("candle", "unary.cu", "candle_unary"),
        ("candle", "binary.cu", "candle_binary"),
        ("candle", "cast.cu", "candle_cast"),
        ("candle", "reduce.cu", "candle_reduce"),
        ("candle", "indexing.cu", "candle_indexing"),
        ("candle", "ternary.cu", "candle_ternary"),
        ("candle", "affine.cu", "candle_affine"),
        ("candle", "fill.cu", "candle_fill"),
        ("candle", "sort.cu", "candle_sort"),
    ];

    // Track common headers for rerun-if-changed
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/common/common.cuh");
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/common/vec_utils.cuh");
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/candle/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/candle/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/kernels/kernels_src/candle/binary_op_macros.cuh");
    println!("cargo:rerun-if-changed=build.rs");

    for &(category, filename, _) in kernel_modules.iter() {
        let kernel_src = kernels_dir.join(category).join(filename);
        println!("cargo:rerun-if-changed={}", kernel_src.display());
    }

    // Compile all PTX kernels in parallel
    let nvcc = Arc::new(nvcc);
    let arch_flag = Arc::new(arch_flag);
    let include_flag = Arc::new(include_flag);
    let candle_include_flag = Arc::new(candle_include_flag);

    let handles: Vec<_> = kernel_modules.iter().map(|&(category, filename, output_name)| {
        let kernels_dir = kernels_dir.clone();
        let out_dir = out_dir.clone();
        let nvcc = nvcc.clone();
        let arch_flag = arch_flag.clone();
        let include_flag = include_flag.clone();
        let candle_include_flag = candle_include_flag.clone();
        std::thread::spawn(move || {
            let kernel_src = kernels_dir.join(category).join(filename);
            let ptx_path = out_dir.join(format!("{output_name}.ptx"));
            let mut args = vec![
                "--ptx".to_string(),
                kernel_src.to_str().unwrap().to_string(),
                "-o".to_string(),
                ptx_path.to_str().unwrap().to_string(),
                arch_flag.to_string(),
                include_flag.to_string(),
                "-O3".to_string(),
                "--use_fast_math".to_string(),
                "--expt-relaxed-constexpr".to_string(),
            ];
            // Candle kernels need C++ std library for sort.cu templates
            if category == "candle" {
                args.push(candle_include_flag.to_string());
                args.push("-std=c++17".to_string());
            }
            let status = Command::new(&*nvcc)
                .args(&args)
                .status()
                .unwrap_or_else(|e| panic!("Failed to run nvcc at {}: {}", nvcc.display(), e));
            assert!(status.success(), "nvcc PTX compilation of {filename} failed");
        })
    }).collect();

    for h in handles {
        h.join().expect("nvcc compilation thread panicked");
    }

    // ── Compile MOE WMMA kernel to static library (FFI, not PTX) ──
    {
        let moe_src = kernels_dir.join("candle").join("moe_wmma.cu");
        let moe_obj = out_dir.join("moe_wmma.o");
        if moe_src.exists() {
            let candle_inc = format!("-I{}", kernels_dir.join("candle").display());
            let mut moe_args = vec![
                "-c",
                moe_src.to_str().unwrap(),
                "-o",
                moe_obj.to_str().unwrap(),
                &arch_flag,
                &candle_inc,
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-std=c++17",
                "-Xcompiler", "-fPIC",
            ];
            if compute_cap < 80 {
                moe_args.push("-DNO_BF16_KERNEL");
            }
            let status = Command::new(&*nvcc)
                .args(&moe_args)
                .status()
                .unwrap_or_else(|e| panic!("Failed to compile moe_wmma.cu: {e}"));
            assert!(status.success(), "nvcc compilation of moe_wmma.cu failed");

            // Create static library
            let moe_lib = out_dir.join("libmoe_wmma.a");
            let status = Command::new("ar")
                .args(["rcs", moe_lib.to_str().unwrap(), moe_obj.to_str().unwrap()])
                .status()
                .unwrap_or_else(|e| panic!("Failed to create libmoe_wmma.a: {e}"));
            assert!(status.success(), "ar failed for moe_wmma");

            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=moe_wmma");
            // Need to link CUDA runtime for the WMMA intrinsics
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rerun-if-changed={}", moe_src.display());
        }
    }
}

fn detect_compute_cap() -> Option<u32> {
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

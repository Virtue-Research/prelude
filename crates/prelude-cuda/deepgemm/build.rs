//! Build script: compile DeepGEMM BF16/FP8 GEMM wrapper with nvcc.
//!
//! Requires:
//! - CUDA Toolkit with nvcc (sm_90a support; sm_100a for Blackwell)
//! - CUTLASS headers (third_party/cutlass submodule)
//! - DeepGEMM headers (third_party/DeepGEMM submodule)

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

include!("../../../build_log.rs");

fn main() {
    println!("cargo:rerun-if-changed=src/deepgemm_wrapper.cu");
    println!("cargo:rerun-if-changed=src/common.cuh");
    println!("cargo:rerun-if-changed=src/sm90_bf16.cuh");
    println!("cargo:rerun-if-changed=src/sm90_fp8.cuh");
    println!("cargo:rerun-if-changed=src/sm100_bf16.cuh");
    println!("cargo:rerun-if-changed=src/sm100_fp8.cuh");
    println!("cargo:rerun-if-changed=src/attention.cuh");
    println!("cargo:rerun-if-changed=src/layout.cuh");
    println!("cargo:rerun-if-changed=src/einsum.cuh");
    track_submodule("DeepGEMM");
    track_submodule("cutlass");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // 1. Ensure CUTLASS headers (third_party/cutlass submodule)
    let workspace_root = PathBuf::from(&manifest_dir).join("../../..");
    let cutlass_dir = workspace_root.join("third_party/cutlass");
    if !cutlass_dir.join("include/cutlass/cutlass.h").exists() {
        panic!(
            "third_party/cutlass submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/cutlass"
        );
    }
    let cutlass_include = cutlass_dir.join("include");

    // 2. Ensure DeepGEMM headers (third_party/DeepGEMM submodule, patched for fat binary)
    let deepgemm_dir = ensure_deepgemm(&out_dir, &workspace_root);
    let deepgemm_include = deepgemm_dir.join("deep_gemm/include");

    // 3. Find CUDA toolkit
    let cuda_path = find_cuda();
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }

    // 4. Compile deepgemm_wrapper.cu
    let common_args = [
        "-std=c++20", "-O3", "--expt-relaxed-constexpr", "-Xcompiler", "-fPIC",
    ];
    let src_dir = manifest_dir.join("src");
    let include_args = [
        "-I", cutlass_include.to_str().unwrap(),
        "-I", deepgemm_include.to_str().unwrap(),
        "-I", &format!("{}/include", cuda_path.display()),
        "-I", src_dir.to_str().unwrap(),
    ];

    // Compile for SM90a always. Add SM100a if toolkit supports it (fat binary).
    // SM90 kernel bodies are guarded by __CUDA_ARCH__ >= 900 && < 1000,
    // SM100 kernel bodies by __CUDA_ARCH__ >= 1000, so both coexist safely.
    let cu_src = manifest_dir.join("src/deepgemm_wrapper.cu");
    let obj = out_dir.join("deepgemm_wrapper.o");

    let sm100 = nvcc_supports_sm100(&nvcc);
    let sm103 = nvcc_supports_sm103(&nvcc);
    if sm103 {
        build_log!("compiling for SM90a + SM100a + SM103a (fat binary)");
    } else if sm100 {
        build_log!("compiling for SM90a + SM100a (fat binary)");
    } else {
        build_log!("compiling for SM90a only (CUDA toolkit lacks SM100 support)");
    }

    let mut nvcc_cmd = Command::new(&nvcc);
    nvcc_cmd.args(&common_args)
        .arg("-gencode=arch=compute_90a,code=sm_90a");
    if sm100 {
        nvcc_cmd.arg("-gencode=arch=compute_100a,code=sm_100a");
    }
    if sm103 {
        // Blackwell-Ultra (B300) has compute cap 10.3. sm_100a cubin won't run
        // on it ("no kernel image is available for execution on the device")
        // and "a"-variant PTX isn't forward-compatible across the family,
        // so emit a native sm_103a cubin when the toolkit supports it.
        nvcc_cmd.arg("-gencode=arch=compute_103a,code=sm_103a");
    }
    let status = nvcc_cmd
        .args(&include_args)
        .args(["-c", cu_src.to_str().unwrap(), "-o", obj.to_str().unwrap()])
        .status()
        .expect("Failed to run nvcc");
    if !status.success() {
        panic!("nvcc compilation failed for deepgemm_wrapper.cu");
    }

    // 4. Create static archive via nvcc --lib (preserves CUDA fatbin sections)
    let lib = out_dir.join("libdeepgemm.a");
    let status = Command::new(&nvcc)
        .arg("--lib")
        .args(["-o", lib.to_str().unwrap()])
        .arg(&obj)
        .status()
        .expect("Failed to run nvcc --lib");
    if !status.success() {
        panic!("nvcc --lib failed");
    }

    // 5. Link
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=deepgemm");

    // CUDA runtime + driver
    let cuda_lib = cuda_path.join("lib64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    let cuda_targets_lib = cuda_path.join("targets/x86_64-linux/lib");
    if cuda_targets_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_targets_lib.display());
    }
    // Stubs for CUDA driver API (cuTensorMapEncodeTiled)
    let cuda_stubs = cuda_path.join("lib64/stubs");
    if cuda_stubs.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_stubs.display());
    }
    let cuda_targets_stubs = cuda_path.join("targets/x86_64-linux/lib/stubs");
    if cuda_targets_stubs.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_targets_stubs.display());
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");   // required by cudart_static
    println!("cargo:rustc-link-lib=dylib=dl");   // required by cudart_static
    println!("cargo:rustc-link-lib=dylib=cuda"); // CUDA driver API (cuTensorMapEncodeTiled)
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn ensure_deepgemm(out_dir: &Path, workspace_root: &Path) -> PathBuf {
    let dg_dir = out_dir.join("deepgemm");

    if dg_dir.join("deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh").exists() {
        return dg_dir;
    }

    // Use third_party/DeepGEMM submodule as source
    let submodule_path = workspace_root.join("third_party/DeepGEMM");
    if !submodule_path.join("deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh").exists() {
        panic!(
            "third_party/DeepGEMM submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/DeepGEMM"
        );
    }

    // Copy submodule to OUT_DIR so we can apply patches without modifying the submodule
    build_log!("Copying DeepGEMM from third_party/ submodule...");
    if dg_dir.exists() {
        std::fs::remove_dir_all(&dg_dir).ok();
    }
    let status = Command::new("cp")
        .args(["-r", submodule_path.to_str().unwrap(), dg_dir.to_str().unwrap()])
        .status()
        .expect("Failed to copy DeepGEMM submodule");
    if !status.success() {
        panic!("Failed to copy DeepGEMM submodule to OUT_DIR");
    }

    // Patch SM90 kernel arch guards for fat binary (SM90a + SM100a) compilation.
    // Upstream uses `__CUDA_ARCH__ >= 900` which includes SM100, but SM90 wgmma
    // instructions are invalid on SM100. Narrow to `>= 900 && < 1000`.
    for file in [
        "deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh",
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh",
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh",
        "deep_gemm/include/deep_gemm/impls/sm90_bmk_bnk_mn.cuh",
    ] {
        let path = dg_dir.join(file);
        if let Ok(content) = std::fs::read_to_string(&path) {
            let patched = content.replace(
                "(defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))",
                "(defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000))",
            );
            if patched != content {
                std::fs::write(&path, patched).expect("Failed to patch DeepGEMM header");
                build_log!("Patched {file} for fat binary arch guard");
            }
        }
    }

    // Patch SM90 MQA/paged-MQA kernels: these don't have arch guards at all in upstream
    // (upstream JIT targets one arch at a time). For fat binary, wrap kernel bodies with
    // __CUDA_ARCH__ >= 900 && < 1000 so SM90 wgmma instructions aren't compiled for SM100.
    patch_kernel_arch_guard(&dg_dir.join(
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh"),
        "sm90_fp8_mqa_logits",
        "(__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)");
    patch_kernel_arch_guard(&dg_dir.join(
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh"),
        "sm90_fp8_paged_mqa_logits",
        "(__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)");

    // Patch SM100 MQA/paged-MQA kernels: same issue, these use SM100 intrinsics
    // (UMMA, TMEM, __ffma2_rn) that don't exist on SM90.
    patch_kernel_arch_guard(&dg_dir.join(
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh"),
        "sm100_fp8_mqa_logits",
        "(__CUDA_ARCH__ >= 1000)");
    patch_kernel_arch_guard(&dg_dir.join(
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh"),
        "sm100_fp8_paged_mqa_logits",
        "(__CUDA_ARCH__ >= 1000)");

    // Patch extern __shared__ declarations in attention headers to avoid
    // type/alignment conflicts in AOT compilation (all kernels in one .cu).
    // Upstream JIT compiles each kernel separately, so this isn't an issue there.
    // We rename smem_buffer → unique names per header file.
    patch_smem_name(&dg_dir, "deep_gemm/include/deep_gemm/impls/smxx_clean_logits.cuh",
                    "smem_buffer", "smem_clean_");
    patch_smem_name(&dg_dir, "deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh",
                    "smem_buffer", "smem_mqa90_");
    patch_smem_name(&dg_dir, "deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh",
                    "smem_buffer", "smem_mqa100_");
    patch_smem_name(&dg_dir, "deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh",
                    "smem_buffer", "smem_pmqa90_");
    patch_smem_name(&dg_dir, "deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh",
                    "smem_buffer", "smem_pmqa100_");

    build_log!("DeepGEMM headers ready");
    dg_dir
}

/// Rename `extern __shared__` variable in a header to avoid type/alignment conflicts.
fn patch_smem_name(dg_dir: &Path, file: &str, old_name: &str, new_name: &str) {
    let path = dg_dir.join(file);
    if let Ok(content) = std::fs::read_to_string(&path) {
        let patched = content.replace(old_name, new_name);
        if patched != content {
            std::fs::write(&path, patched).expect("Failed to patch smem name");
            let fname = path.file_name().unwrap().to_str().unwrap();
            build_log!("Patched {fname}: renamed {old_name} → {new_name}");
        }
    }
}

/// Patch attention kernel headers that lack arch guards for fat binary compilation.
/// `kernel_name_prefix`: e.g. "sm90_fp8_mqa_logits" — matches `void <prefix>(`
/// `arch_guard`: e.g. "(__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)"
fn patch_kernel_arch_guard(path: &Path, kernel_name_prefix: &str, arch_guard: &str) {
    let Ok(content) = std::fs::read_to_string(path) else { return };
    let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
    let file_name = path.file_name().unwrap().to_str().unwrap();
    let pattern = format!("void {}(", kernel_name_prefix);
    let guard_line = format!("#if (defined(__CUDA_ARCH__) and {}) or defined(__CLION_IDE__)", arch_guard);

    let mut insertions: Vec<(usize, String)> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].contains(&pattern) {
            // Find the opening brace of this function
            let mut j = i;
            while j < lines.len() && !lines[j].trim_end().ends_with('{') {
                j += 1;
            }
            if j < lines.len() {
                insertions.push((j + 1, guard_line.clone()));

                // Find matching closing brace
                let mut depth = 1;
                let mut k = j + 1;
                while k < lines.len() && depth > 0 {
                    for ch in lines[k].chars() {
                        if ch == '{' { depth += 1; }
                        if ch == '}' { depth -= 1; }
                        if depth == 0 { break; }
                    }
                    if depth > 0 { k += 1; }
                }
                if depth == 0 {
                    insertions.push((k, "#endif".to_string()));
                }
            }
            i = if !insertions.is_empty() { insertions.last().unwrap().0 + 1 } else { i + 1 };
        } else {
            i += 1;
        }
    }

    if insertions.is_empty() { return; }
    insertions.sort_by(|a, b| b.0.cmp(&a.0));
    for (idx, text) in &insertions {
        lines.insert(*idx, text.clone());
    }
    std::fs::write(path, lines.join("\n")).expect("Failed to patch kernel header");
    build_log!("Patched {file_name} for fat binary arch guard ({} insertions)", insertions.len());
}

fn find_cuda() -> PathBuf {
    if let Ok(p) = env::var("CUDA_PATH") {
        return PathBuf::from(p);
    }
    for p in ["/usr/local/cuda", "/opt/cuda"] {
        if Path::new(p).join("bin/nvcc").exists() {
            return PathBuf::from(p);
        }
    }
    panic!("CUDA toolkit not found. Set CUDA_PATH env var.");
}

fn nvcc_supports_sm100(nvcc: &Path) -> bool {
    let output = Command::new(nvcc)
        .args(["--list-gpu-arch"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).contains("compute_100")
        }
        _ => false,
    }
}

fn nvcc_supports_sm103(nvcc: &Path) -> bool {
    let output = Command::new(nvcc)
        .args(["--list-gpu-arch"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).contains("compute_103")
        }
        _ => false,
    }
}


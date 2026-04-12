//! Build script: compile DeepGEMM BF16/FP8 GEMM wrapper with nvcc.
//!
//! Single translation unit (`deepgemm_wrapper.cu`) pulls in CUTLASS and
//! DeepGEMM headers, emits a fat binary across SM90a (always) and SM100a
//! (when nvcc is new enough). The one crate-specific complication is
//! that upstream DeepGEMM ships SM90 kernels with an `__CUDA_ARCH__ >= 900`
//! guard that's too loose — it accepts SM100 and then emits invalid
//! wgmma instructions. We copy the submodule into `OUT_DIR` and patch the
//! guards before nvcc sees them. That logic is deepgemm-specific so it
//! stays inline here.
//!
//! Everything else — toolkit discovery, arch probing, nvcc invocation,
//! submodule tracking, CUDA runtime linking — flows through
//! `prelude_kernelbuild::nvcc`. DeepGEMM also needs `libcuda` (the CUDA
//! driver API, for `cuTensorMapEncodeTiled`) and the `stubs` search
//! directory, which aren't part of the standard `link_cuda_runtime_static`
//! helper; those two extra lines live here.

use std::path::{Path, PathBuf};
use std::process::Command;

use prelude_kernelbuild::build_log;
use prelude_kernelbuild::nvcc::{
    compile_cu_to_obj, find_cuda, link_cuda_runtime_static, locate_source, nvcc_path,
    nvcc_supports_sm100, track_submodule, ObjCompile,
};

fn main() {
    for header in [
        "src/deepgemm_wrapper.cu",
        "src/common.cuh",
        "src/sm90_bf16.cuh",
        "src/sm90_fp8.cuh",
        "src/sm100_bf16.cuh",
        "src/sm100_fp8.cuh",
        "src/attention.cuh",
        "src/layout.cuh",
        "src/einsum.cuh",
    ] {
        println!("cargo:rerun-if-changed={header}");
    }
    println!("cargo:rerun-if-changed=build.rs");
    track_submodule("DeepGEMM");
    track_submodule("cutlass");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.join("../../..");

    let cutlass_dir = locate_source(
        "CUTLASS_ROOT",
        "cutlass",
        "include/cutlass/cutlass.h",
        &workspace_root.join("third_party/cutlass"),
    );
    let deepgemm_dir = ensure_deepgemm(&out_dir, &workspace_root);
    let deepgemm_include = deepgemm_dir.join("deep_gemm/include");

    let cuda_path = find_cuda();
    let nvcc = nvcc_path(&cuda_path);
    let sm100 = nvcc_supports_sm100(&nvcc);

    if sm100 {
        build_log!("compiling for SM90a + SM100a (fat binary)");
    } else {
        build_log!("compiling for SM90a only (CUDA toolkit lacks SM100 support)");
    }

    // ── Compile the single TU ───────────────────────────────────────
    let src = manifest_dir.join("src/deepgemm_wrapper.cu");
    let obj = out_dir.join("deepgemm_wrapper.o");
    let mut opts = ObjCompile::new(&src, &obj)
        .include(cutlass_dir.join("include"))
        .include(deepgemm_include)
        .include(cuda_path.join("include"))
        .include(manifest_dir.join("src"))
        .gencode("-gencode=arch=compute_90a,code=sm_90a");
    if sm100 {
        opts = opts.gencode("-gencode=arch=compute_100a,code=sm_100a");
    }
    compile_cu_to_obj(&nvcc, &opts);

    // ── Archive via nvcc --lib so the fatbin sections survive ───────
    let lib = out_dir.join("libdeepgemm.a");
    let status = Command::new(&nvcc)
        .arg("--lib")
        .args(["-o", lib.to_str().unwrap()])
        .arg(&obj)
        .status()
        .expect("Failed to run nvcc --lib");
    if !status.success() {
        panic!("nvcc --lib failed for libdeepgemm.a");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=deepgemm");

    link_cuda_runtime_static(&cuda_path);

    // DeepGEMM also needs libcuda.so (the CUDA driver API) for
    // cuTensorMapEncodeTiled — emit the extra link directives that
    // aren't part of the standard `link_cuda_runtime_static` helper.
    let cuda_stubs = cuda_path.join("lib64/stubs");
    if cuda_stubs.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_stubs.display());
    }
    let cuda_targets_stubs = cuda_path.join("targets/x86_64-linux/lib/stubs");
    if cuda_targets_stubs.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_targets_stubs.display());
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
}

// ─────────────────────────────────────────────────────────────────────
// DeepGEMM source prep (submodule copy + header patches)
//
// Upstream DeepGEMM's headers assume per-arch JIT compilation and use
// `__CUDA_ARCH__ >= 900` guards that accept SM100. When we AOT-compile
// everything together into a fat binary, those guards let SM100 bodies
// include SM90 wgmma instructions (→ PTX assembly errors). The fix is
// to copy the submodule to OUT_DIR and patch the guards before nvcc
// consumes them. Leaving this inline in the build script rather than
// pushing it into prelude-kernelbuild because it's genuinely
// DeepGEMM-specific — no other crate has this problem.
// ─────────────────────────────────────────────────────────────────────

fn ensure_deepgemm(out_dir: &Path, workspace_root: &Path) -> PathBuf {
    let dg_dir = out_dir.join("deepgemm");

    if dg_dir
        .join("deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh")
        .exists()
    {
        return dg_dir;
    }

    let submodule_path = workspace_root.join("third_party/DeepGEMM");
    if !submodule_path
        .join("deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh")
        .exists()
    {
        panic!(
            "third_party/DeepGEMM submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/DeepGEMM"
        );
    }

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
    // Upstream uses `__CUDA_ARCH__ >= 900` which also matches SM100, but SM90
    // wgmma instructions are invalid on SM100. Narrow to `>= 900 && < 1000`.
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

    // SM90 MQA / paged-MQA kernels have no arch guards at all in upstream
    // (upstream JIT targets one arch at a time). Wrap kernel bodies so
    // SM90 wgmma isn't compiled for SM100.
    patch_kernel_arch_guard(
        &dg_dir.join("deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh"),
        "sm90_fp8_mqa_logits",
        "(__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)",
    );
    patch_kernel_arch_guard(
        &dg_dir.join("deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh"),
        "sm90_fp8_paged_mqa_logits",
        "(__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)",
    );

    // SM100 MQA / paged-MQA kernels use SM100 intrinsics (UMMA, TMEM,
    // __ffma2_rn) that don't exist on SM90.
    patch_kernel_arch_guard(
        &dg_dir.join("deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh"),
        "sm100_fp8_mqa_logits",
        "(__CUDA_ARCH__ >= 1000)",
    );
    patch_kernel_arch_guard(
        &dg_dir.join("deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh"),
        "sm100_fp8_paged_mqa_logits",
        "(__CUDA_ARCH__ >= 1000)",
    );

    // `extern __shared__` collisions across the combined TU — upstream JIT
    // doesn't hit them because each kernel compiles in its own TU. Rename
    // per header to give each smem_buffer a unique type.
    patch_smem_name(
        &dg_dir,
        "deep_gemm/include/deep_gemm/impls/smxx_clean_logits.cuh",
        "smem_buffer",
        "smem_clean_",
    );
    patch_smem_name(
        &dg_dir,
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh",
        "smem_buffer",
        "smem_mqa90_",
    );
    patch_smem_name(
        &dg_dir,
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh",
        "smem_buffer",
        "smem_mqa100_",
    );
    patch_smem_name(
        &dg_dir,
        "deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh",
        "smem_buffer",
        "smem_pmqa90_",
    );
    patch_smem_name(
        &dg_dir,
        "deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh",
        "smem_buffer",
        "smem_pmqa100_",
    );

    build_log!("DeepGEMM headers ready");
    dg_dir
}

/// Rename `extern __shared__` variable in a header to avoid type/alignment
/// conflicts when multiple kernels share the same TU.
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

/// Patch attention kernel headers that lack arch guards for fat binary
/// compilation. `kernel_name_prefix`: e.g. "sm90_fp8_mqa_logits". Matches
/// `void <prefix>(`. `arch_guard`: e.g. "(__CUDA_ARCH__ >= 900) and
/// (__CUDA_ARCH__ < 1000)".
fn patch_kernel_arch_guard(path: &Path, kernel_name_prefix: &str, arch_guard: &str) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
    let file_name = path.file_name().unwrap().to_str().unwrap();
    let pattern = format!("void {}(", kernel_name_prefix);
    let guard_line = format!(
        "#if (defined(__CUDA_ARCH__) and {}) or defined(__CLION_IDE__)",
        arch_guard
    );

    let mut insertions: Vec<(usize, String)> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].contains(&pattern) {
            let mut j = i;
            while j < lines.len() && !lines[j].trim_end().ends_with('{') {
                j += 1;
            }
            if j < lines.len() {
                insertions.push((j + 1, guard_line.clone()));

                let mut depth = 1;
                let mut k = j + 1;
                while k < lines.len() && depth > 0 {
                    for ch in lines[k].chars() {
                        if ch == '{' {
                            depth += 1;
                        }
                        if ch == '}' {
                            depth -= 1;
                        }
                        if depth == 0 {
                            break;
                        }
                    }
                    if depth > 0 {
                        k += 1;
                    }
                }
                if depth == 0 {
                    insertions.push((k, "#endif".to_string()));
                }
            }
            i = if !insertions.is_empty() {
                insertions.last().unwrap().0 + 1
            } else {
                i + 1
            };
        } else {
            i += 1;
        }
    }

    if insertions.is_empty() {
        return;
    }
    insertions.sort_by(|a, b| b.0.cmp(&a.0));
    for (idx, text) in &insertions {
        lines.insert(*idx, text.clone());
    }
    std::fs::write(path, lines.join("\n")).expect("Failed to patch kernel header");
    build_log!(
        "Patched {file_name} for fat binary arch guard ({} insertions)",
        insertions.len()
    );
}

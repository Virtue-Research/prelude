//! Build script for prelude-flashinfer.
//!
//! 1. Find FlashInfer source (FLASHINFER_SRC env var or auto-clone).
//! 2. Run scripts/compile_kernels.py to AOT-compile kernel variants.
//! 3. Archive all .o files into libflashinfer_kernels.a.
//! 4. Compile vendored tvm_ffi C++ (shared with FA4).
//! 5. Generate fi_dispatch.rs from manifest.json.

use anyhow::{Context, Result};
use std::env;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=scripts/compile_kernels.py");
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-env-changed=FLASHINFER_SRC");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_ARCHS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_HEAD_DIMS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_DTYPES");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_WORKERS");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let kernels_dir = manifest_dir.join("kernels");

    // Phase 1+2: Compile kernels if needed (skip if pre-compiled .o exist)
    let manifest = kernels_dir.join("manifest.json");
    let precompiled_objs = walkdir_ext(&kernels_dir, "o");
    if manifest.exists() && !precompiled_objs.is_empty() {
        let n = precompiled_objs.len();
        println!("cargo:warning=FlashInfer: using {n} pre-compiled kernel objects");
    } else {
        let fi_src = find_flashinfer_source(&manifest_dir)?;
        ensure_kernels(&kernels_dir, &manifest_dir, &fi_src)?;
    }

    // Phase 3: Archive .o files
    let has_kernels = archive_objects(&kernels_dir, &out_dir)?;

    // Phase 4: Compile TVM FFI (reuse FA4's vendored copy).
    // Skip when `skip-tvm-ffi` feature is enabled — another crate (FA4) already
    // compiles the same tvm_ffi library, and linking both causes duplicate symbols.
    if env::var("CARGO_FEATURE_SKIP_TVM_FFI").is_ok() {
        println!("cargo:warning=FlashInfer: skipping tvm_ffi (provided by flash-attn-v4)");
    } else {
        compile_tvm_ffi(&manifest_dir)?;
    }

    // Phase 5: Generate dispatch code
    generate_dispatch(&kernels_dir, &out_dir, has_kernels)?;

    Ok(())
}

fn find_flashinfer_source(manifest_dir: &Path) -> Result<PathBuf> {
    // Check env var first
    if let Ok(src) = env::var("FLASHINFER_SRC") {
        let p = PathBuf::from(&src);
        if p.join("csrc").exists() {
            println!("cargo:warning=Using FlashInfer source: {src}");
            return Ok(p);
        }
        anyhow::bail!("FLASHINFER_SRC={src} does not contain csrc/");
    }

    // Check sibling directory (common development layout)
    let workspace_root = manifest_dir.parent().and_then(|p| p.parent());
    if let Some(root) = workspace_root {
        for candidate in ["flashinfer", "../kernel/flashinfer"] {
            let p = root.join(candidate);
            if p.join("csrc").exists() {
                println!("cargo:warning=Found FlashInfer at {}", p.display());
                return Ok(p);
            }
        }
    }

    anyhow::bail!(
        "FlashInfer source not found. Set FLASHINFER_SRC env var \
         or place flashinfer/ next to this workspace."
    )
}

fn ensure_kernels(
    kernels_dir: &Path, manifest_dir: &Path, fi_src: &Path,
) -> Result<()> {
    // Caller already verified kernels are missing; proceed directly to compilation.
    println!("cargo:warning=FlashInfer: compiling kernels via compile_kernels.py...");

    let script = manifest_dir.join("scripts/compile_kernels.py");
    let python = find_python()?;

    // Determine target archs
    let archs = env::var("PRELUDE_FLASHINFER_ARCHS")
        .unwrap_or_else(|_| "sm_80,sm_90".to_string());
    let head_dims = env::var("PRELUDE_FLASHINFER_HEAD_DIMS")
        .unwrap_or_else(|_| "64,96,128,192,256".to_string());
    let dtypes = env::var("PRELUDE_FLASHINFER_DTYPES")
        .unwrap_or_else(|_| "bf16,fp16".to_string());
    let workers = env::var("PRELUDE_FLASHINFER_WORKERS").unwrap_or_else(|_| {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        cpus.min(8).to_string()
    });

    let status = Command::new(&python)
        .arg(&script)
        .arg("--flashinfer-src").arg(fi_src)
        .arg("--out-dir").arg(kernels_dir)
        .arg("--archs").arg(&archs)
        .arg("--head-dims").arg(&head_dims)
        .arg("--dtypes").arg(&dtypes)
        .arg("-j").arg(&workers)
        .status()
        .context("Failed to run compile_kernels.py")?;

    if !status.success() {
        anyhow::bail!("compile_kernels.py failed");
    }

    Ok(())
}

fn archive_objects(kernels_dir: &Path, out_dir: &Path) -> Result<bool> {
    let obj_files = walkdir_ext(kernels_dir, "o");

    if obj_files.is_empty() {
        println!("cargo:warning=No FlashInfer kernel .o files found");
        return Ok(false);
    }

    println!("cargo:warning=Archiving {} FlashInfer kernel .o files", obj_files.len());

    let archive = out_dir.join("libflashinfer_kernels.a");
    let mut cmd = Command::new("ar");
    cmd.arg("rcs").arg(&archive);
    for obj in &obj_files {
        cmd.arg(obj);
    }
    let status = cmd.status().context("Failed to run ar")?;
    if !status.success() {
        anyhow::bail!("ar failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=flashinfer_kernels");

    Ok(true)
}

fn compile_tvm_ffi(manifest_dir: &Path) -> Result<()> {
    // Try to reuse FA4's vendored tvm_ffi
    let fa4_vendor = manifest_dir
        .parent().unwrap()
        .join("prelude-flash-attn-v4/vendor/tvm_ffi");

    let (tvm_src, tvm_include, dlpack_include) = if fa4_vendor.join("src").exists() {
        (
            fa4_vendor.join("src"),
            fa4_vendor.join("include"),
            fa4_vendor.join("3rdparty/dlpack/include"),
        )
    } else {
        println!("cargo:warning=tvm_ffi vendor not found (FA4 crate needed for TVM FFI)");
        return Ok(());
    };

    let cc_files: Vec<PathBuf> = walkdir_ext(&tvm_src, "cc")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap_or("");
            !name.contains("win") && !name.contains("testing") && name != "backtrace.cc"
        })
        .collect();

    println!("cargo:warning=Compiling tvm_ffi: {} files", cc_files.len());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .pic(true)
        .include(&tvm_include)
        .include(&dlpack_include)
        .define("TVM_FFI_EXPORTS", None)
        .define("NDEBUG", None)
        .warnings(false);

    for f in &cc_files {
        build.file(f);
    }

    build.link_lib_modifier("+whole-archive");
    build.try_compile("tvm_ffi_fi_static")
        .context("Failed to compile tvm_ffi")?;

    // Also link cuda_dialect_runtime if available
    let cuda_dialect = manifest_dir
        .parent().unwrap()
        .join("prelude-flash-attn-v4/vendor/cuda_dialect");
    if cuda_dialect.join("libcuda_dialect_runtime_static.a").exists() {
        println!("cargo:rustc-link-search=native={}", cuda_dialect.display());
        println!("cargo:rustc-link-lib=static:+whole-archive=cuda_dialect_runtime_static");
    }

    // CUDA runtime
    for candidate in [
        "/opt/cuda/targets/x86_64-linux/lib",
        "/opt/cuda/lib64",
        "/usr/local/cuda/lib64",
    ] {
        if Path::new(candidate).join("libcudart.so").exists() {
            println!("cargo:rustc-link-search=native={candidate}");
            break;
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

fn generate_dispatch(
    kernels_dir: &Path, out_dir: &Path, has_kernels: bool,
) -> Result<()> {
    let path = out_dir.join("fi_dispatch.rs");

    if !has_kernels {
        std::fs::write(&path, concat!(
            "pub(crate) fn lookup_prefill(_key: &crate::loader::PrefillKey) -> Option<crate::loader::PrefillVariant> { None }\n",
            "pub(crate) fn lookup_decode(_key: &crate::loader::DecodeKey) -> Option<crate::loader::DecodeVariant> { None }\n",
        ))?;
        return Ok(());
    }

    let manifest_str = std::fs::read_to_string(kernels_dir.join("manifest.json"))
        .context("manifest.json not found")?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_str)?;
    let variants = manifest["variants"].as_array().context("missing variants")?;

    let mut code = String::new();
    writeln!(code, "// AUTO-GENERATED by build.rs — do not edit")?;
    writeln!(code)?;

    // Extern declarations
    writeln!(code, "unsafe extern \"C\" {{")?;
    for v in variants {
        let symbols = v["symbols"].as_object().context("missing symbols")?;
        for (_name, sym) in symbols {
            let s = sym.as_str().unwrap();
            writeln!(code,
                "    fn {s}(handle: *mut c_void, args: *const TVMFFIAny, num_args: i32, ret: *mut TVMFFIAny) -> i32;"
            )?;
        }
    }
    writeln!(code, "}}")?;
    writeln!(code)?;

    // Prefill lookup
    writeln!(code, "pub(crate) fn lookup_prefill(key: &crate::loader::PrefillKey) -> Option<crate::loader::PrefillVariant> {{")?;
    writeln!(code, "    use crate::loader::{{KernelDtype, Backend, PrefillVariant}};")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_qk, key.head_dim_vo, key.sliding_window, key.logits_soft_cap, key.backend as u8) {{")?;

    for v in variants {
        let kind = v["kind"].as_str().unwrap();
        if !kind.starts_with("prefill") { continue; }
        let symbols = v["symbols"].as_object().unwrap();
        let plan = symbols["plan"].as_str().unwrap();
        let ragged_run = symbols["ragged_run"].as_str().unwrap();
        let paged_run = symbols["paged_run"].as_str().unwrap();

        let dtype_val: u8 = match v["dtype"].as_str().unwrap() {
            "fp16" => 1, _ => 0,
        };
        let backend_val: u8 = match v["backend"].as_str().unwrap() {
            "fa3" => 1, _ => 0,
        };
        let hdim_qk = v["hdim_qk"].as_u64().unwrap();
        let hdim_vo = v["hdim_vo"].as_u64().unwrap();
        let swa = v["swa"].as_bool().unwrap();
        let softcap = v["softcap"].as_bool().unwrap();

        writeln!(code,
            "        ({dtype_val}, {hdim_qk}, {hdim_vo}, {swa}, {softcap}, {backend_val}) => Some(PrefillVariant {{ plan: {plan}, ragged_run: {ragged_run}, paged_run: {paged_run} }}),"
        )?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}")?;
    writeln!(code, "}}")?;
    writeln!(code)?;

    // Decode lookup
    writeln!(code, "pub(crate) fn lookup_decode(key: &crate::loader::DecodeKey) -> Option<crate::loader::DecodeVariant> {{")?;
    writeln!(code, "    use crate::loader::{{KernelDtype, DecodeVariant}};")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_qk, key.head_dim_vo, key.sliding_window, key.logits_soft_cap) {{")?;

    for v in variants {
        if v["kind"].as_str().unwrap() != "decode" { continue; }
        let symbols = v["symbols"].as_object().unwrap();
        let plan = symbols["plan"].as_str().unwrap();
        let run = symbols["run"].as_str().unwrap();

        let dtype_val: u8 = match v["dtype"].as_str().unwrap() {
            "fp16" => 1, _ => 0,
        };
        let hdim_qk = v["hdim_qk"].as_u64().unwrap();
        let hdim_vo = v["hdim_vo"].as_u64().unwrap();
        let swa = v["swa"].as_bool().unwrap();
        let softcap = v["softcap"].as_bool().unwrap();

        writeln!(code,
            "        ({dtype_val}, {hdim_qk}, {hdim_vo}, {swa}, {softcap}) => Some(DecodeVariant {{ plan: {plan}, run: {run} }}),"
        )?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}")?;
    writeln!(code, "}}")?;

    std::fs::write(&path, &code)?;
    println!("cargo:warning=Generated fi_dispatch.rs with {} variants", variants.len());

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────

fn walkdir_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut result = Vec::new();
    fn walk(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, ext, out);
                } else if path.extension().is_some_and(|e| e == ext) {
                    out.push(path);
                }
            }
        }
    }
    walk(dir, ext, &mut result);
    result
}

fn find_python() -> Result<PathBuf> {
    for candidate in ["python3", "python"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Ok(PathBuf::from(candidate));
        }
    }
    anyhow::bail!("Python 3 not found")
}

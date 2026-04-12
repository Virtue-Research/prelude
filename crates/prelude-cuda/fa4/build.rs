//! Build script for the `flash-attn-v4` crate.
//!
//! Runs the FA4 Python CuTeDSL compile script to AOT-generate kernel
//! `.o` files, archives them into `libfa4_kernels.a`, and generates a
//! Rust dispatch table mapping `KernelKey` + arch → TVM-FFI function
//! pointer.
//!
//! Shared build-support lives in `prelude_kernelbuild`:
//!
//!   * `venv` — Python venv provisioning, `uv`/pip wrapping, torch
//!     wheel index detection, importability checks
//!   * `dispatch` — manifest parsing, TVM-FFI extern-block emission,
//!     stub fallback
//!   * `archive` — `.o` collection, `ar rcs` + whole-archive linking
//!
//! The one FA4-specific bit that stays inline is the dispatch table
//! layout: FA4 keys by a 12-element tuple of shape params (head_dim,
//! head_dim_v, gqa_ratio, causal, window, ...), not by a simple string
//! name, so the `match` arms get assembled from the manifest fields
//! here rather than in the shared helper.

use anyhow::{Context, Result};
use std::env;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

use prelude_kernelbuild::archive::{self, ArMode};
use prelude_kernelbuild::build_log;
use prelude_kernelbuild::dispatch;
use prelude_kernelbuild::nvcc::{file_hash, track_submodule};
use prelude_kernelbuild::venv::{detect_torch_cuda_index, InstallOpts, PythonVenv};

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=scripts/compile_kernels.py");
    println!("cargo:rerun-if-changed=vendor/");
    track_submodule("flash-attention");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let workspace_root = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let kernels_dir = out_dir.join("fa4_kernels");
    // Stable venv location — survives `cargo clean -p`.
    let venv_dir = workspace_root.join("target/fa4-venv");

    // Phase 1: Ensure FA4 Python source is available
    let fa4_src = ensure_fa4_source(workspace_root)?;

    // Phase 2: Ensure kernel .o files exist (auto-creates Python venv if needed)
    ensure_kernels(&kernels_dir, &manifest_dir, &fa4_src, &venv_dir)?;

    // Phase 3: Archive the .o files and generate the dispatch table.
    let objects = archive::collect_obj_files(&kernels_dir);
    // FA4 kernels have unique basenames per variant (fa4_fwd_hdim64_...)
    // so replace-in-place is correct and the cheapest option.
    let has_kernels = archive::archive_and_whole_link(&objects, &out_dir, "fa4_kernels", ArMode::Replace)
        .map_err(anyhow::Error::msg)?;

    generate_dispatch_code(&kernels_dir, &out_dir, has_kernels)?;

    Ok(())
}

/// FA4's dispatch key is a 12-tuple of shape params, so we can't drop
/// in the generic name-based lookup from `prelude_kernelbuild::dispatch`
/// — we read the manifests via the shared helper, then assemble the
/// match arms by hand.
fn generate_dispatch_code(kernels_dir: &Path, out_dir: &Path, has_kernels: bool) -> Result<()> {
    let dispatch_path = out_dir.join("fa4_dispatch.rs");

    let stub_signature =
        "pub(crate) fn lookup(_key: &crate::loader::KernelKey, _arch: u32) \
         -> Option<crate::loader::TVMSafeCallFn>";

    if !has_kernels {
        std::fs::write(&dispatch_path, dispatch::stub_lookup(stub_signature))?;
        return Ok(());
    }

    let variants = dispatch::collect_manifests(kernels_dir);
    if variants.is_empty() {
        std::fs::write(&dispatch_path, dispatch::stub_lookup(stub_signature))?;
        return Ok(());
    }

    // Extern declarations for every compiled symbol. Loader already has
    // `c_void` and `TVMFFIAny` in scope (the generated file is
    // `include!`-ed into `loader.rs`), so the path we pass is just
    // `TVMFFIAny`.
    let symbols: Vec<&str> = variants
        .iter()
        .filter_map(|v| v["symbol"].as_str())
        .collect();

    let mut code = dispatch::header_comment();
    writeln!(
        code,
        "// Included into loader.rs which already imports c_void and TVMFFIAny."
    )?;
    writeln!(code)?;
    code.push_str(&dispatch::tvm_ffi_extern_block(
        symbols.iter().copied(),
        "TVMFFIAny",
    ));

    // FA4-specific lookup signature: matches on the struct fields.
    writeln!(
        code,
        "pub(crate) fn lookup(key: &crate::loader::KernelKey, arch: u32) \
         -> Option<crate::loader::TVMSafeCallFn> {{"
    )?;
    writeln!(
        code,
        "    match (key.head_dim, key.head_dim_v, key.gqa_ratio, key.causal, key.window, \
         key.pack_gqa, key.softcap_bits, key.paged, key.paged_non_tma, key.dtype as u8, \
         key.has_seqused_q, arch) {{"
    )?;
    for variant in &variants {
        let symbol = variant["symbol"].as_str().unwrap();
        let head_dim = variant["head_dim"].as_u64().unwrap();
        let head_dim_v = variant
            .get("head_dim_v")
            .and_then(|v| v.as_u64())
            .unwrap_or(head_dim);
        let gqa_ratio = variant["gqa_ratio"].as_u64().unwrap();
        let causal = variant["causal"].as_bool().unwrap();
        let window = variant["window"].as_bool().unwrap();
        let arch = variant["arch"].as_u64().unwrap_or(90);
        let pack_gqa = variant
            .get("pack_gqa")
            .and_then(|v| v.as_bool())
            .unwrap_or(gqa_ratio > 1);
        let softcap_bits = match variant.get("softcap") {
            Some(v) if !v.is_null() => {
                let f = v.as_f64().unwrap_or(0.0) as f32;
                f.to_bits()
            }
            _ => 0u32,
        };
        let paged = variant
            .get("paged")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let dtype_val: u8 = match variant.get("dtype").and_then(|v| v.as_str()) {
            Some("fp16") => 1,
            _ => 0, // bf16
        };
        let paged_non_tma = variant
            .get("paged_non_tma")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let has_seqused_q = variant
            .get("has_seqused_q")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        writeln!(
            code,
            "        ({head_dim}, {head_dim_v}, {gqa_ratio}, {causal}, {window}, \
             {pack_gqa}, {softcap_bits}_u32, {paged}, {paged_non_tma}, {dtype_val}_u8, \
             {has_seqused_q}, {arch}) => Some({symbol}),"
        )?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}")?;
    writeln!(code, "}}")?;

    std::fs::write(&dispatch_path, &code)?;
    dispatch::log_generated("FA4", variants.len());

    Ok(())
}

fn ensure_fa4_source(workspace_root: &Path) -> Result<PathBuf> {
    let fa4_src = workspace_root.join("third_party/flash-attention");
    if !fa4_src.join("flash_attn/cute/__init__.py").exists() {
        anyhow::bail!(
            "third_party/flash-attention submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/flash-attention"
        );
    }
    Ok(fa4_src)
}

/// Provision the FA4 venv and run the compile script per target
/// arch. FA4's layout differs from cuLA's in two ways that keep it
/// from using the shared `prelude_kernelbuild::dsl` driver:
///
///   * All archs write `.o` files into a flat `kernels_dir` (not
///     per-arch subdirectories). The same directory is populated
///     incrementally across invocations.
///   * Each arch needs a *set* of env vars (`FLASH_ATTENTION_ARCH`,
///     `FLASH_ATTENTION_FAKE_TENSOR`, `FA_CLC`, `FA_DISABLE_2CTA`,
///     `CUTE_DSL_ARCH`, `PYTHONPATH`) rather than a single
///     `CUTE_DSL_ARCH=smXXa`, so the driver's per-arch-env-var hook
///     isn't a clean fit.
///
/// Forcing those into the driver would add a `flat_output` +
/// callback-shaped env strategy for marginal reuse. cuLA is the only
/// current consumer of the shared driver; if a future crate lands
/// with FA4's flat-output pattern we revisit then. The cache check
/// and sticky-failure logic still stay inline here but are small
/// enough not to matter.
fn ensure_kernels(
    kernels_dir: &Path,
    manifest_dir: &Path,
    fa4_src: &Path,
    venv_dir: &Path,
) -> Result<()> {
    let script = manifest_dir.join("scripts/compile_kernels.py");
    let current_hash = file_hash(&script).unwrap_or_default();

    // Cache hit: script hash hasn't changed since the last successful
    // build.
    if kernels_dir.join("manifest.json").exists() {
        let has_objs = std::fs::read_dir(kernels_dir)
            .ok()
            .into_iter()
            .flatten()
            .any(|e| {
                e.ok()
                    .map(|e| e.path().extension().is_some_and(|x| x == "o"))
                    .unwrap_or(false)
            });
        if has_objs {
            let manifest_str =
                std::fs::read_to_string(kernels_dir.join("manifest.json")).unwrap_or_default();
            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                let stored_hash = manifest
                    .get("script_hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if stored_hash == current_hash {
                    return Ok(());
                }
                build_log!(
                    "compile_kernels.py changed (hash {stored_hash:.8} → {current_hash:.8}), \
                     clearing old kernels and recompiling..."
                );
                for entry in std::fs::read_dir(kernels_dir).into_iter().flatten().flatten() {
                    let p = entry.path();
                    if p.extension().is_some_and(|x| x == "o") {
                        let _ = std::fs::remove_file(&p);
                    }
                }
                let _ = std::fs::remove_file(kernels_dir.join("manifest.json"));
            } else {
                return Ok(());
            }
        } else {
            let _ = std::fs::remove_file(kernels_dir.join("manifest.json"));
        }
    }

    build_log!("No pre-compiled FA4 kernels, attempting AOT compilation...");
    if !script.exists() {
        anyhow::bail!(
            "No pre-compiled FA4 kernels and no compile script.\n\
             Run compile_kernels.py manually or copy kernels/ from a build machine."
        );
    }

    let python = ensure_fa4_python_env(venv_dir)?;

    // Collect target archs: local GPU + PRELUDE_FA4_ARCHS env var
    // override. Falls back to sm_90 for headless build hosts.
    let mut archs = Vec::new();
    if let Ok(local) = detect_gpu_arch() {
        archs.push(local.clone());
    }
    if let Ok(extra) = env::var("PRELUDE_FA4_ARCHS") {
        for a in extra.split(',') {
            let a = a.trim().to_string();
            if !a.is_empty() && !archs.contains(&a) {
                archs.push(a);
            }
        }
    }
    if archs.is_empty() {
        archs.push("sm_90".to_string());
    }

    let workers = env::var("PRELUDE_FA4_WORKERS").unwrap_or_else(|_| {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        num_cpus.min(8).to_string()
    });

    for arch in &archs {
        build_log!("FA4 AOT compiling for {arch}...");

        let status = Command::new(&python)
            .arg(&script)
            .arg("--output-dir")
            .arg(kernels_dir)
            .args(["-j", &workers])
            .env("PYTHONPATH", fa4_src)
            .env("FLASH_ATTENTION_FAKE_TENSOR", "1")
            .env("FLASH_ATTENTION_ARCH", arch)
            .env("CUTE_DSL_ARCH", format!("{arch}a"))
            .env("FA_CLC", "0")
            .env("FA_DISABLE_2CTA", "0")
            // Expose the shared dsl_driver helpers to the compile
            // script. See prelude_kernelbuild::scripts_dir() for the
            // resolution mechanism.
            .env("PRELUDE_KB_SCRIPTS_DIR", prelude_kernelbuild::scripts_dir())
            .status()
            .context("Failed to run FA4 compile script")?;

        if !status.success() {
            build_log!("FA4 kernel compilation failed for {arch}");
        }
    }

    let has_objs = std::fs::read_dir(kernels_dir)
        .ok()
        .into_iter()
        .flatten()
        .any(|e| {
            e.ok()
                .map(|e| e.path().extension().is_some_and(|x| x == "o"))
                .unwrap_or(false)
        });
    if !has_objs {
        anyhow::bail!(
            "FA4 kernel compilation produced no .o files for archs: {archs:?}.\n\
             Check Python environment and CUDA setup."
        );
    }

    Ok(())
}

/// Provision (or reuse) the FA4 venv with cutlass-dsl + quack + torch
/// installed. Short-circuits when the venv's `cutlass` import already
/// works (covers both "system Python already has it" and "cached venv
/// survives `cargo clean`").
fn ensure_fa4_python_env(venv_dir: &Path) -> Result<PathBuf> {
    // 1. Short-circuit: if system python3 already has cutlass, use it
    //    directly — no venv construction needed.
    for candidate in ["python3", "python"] {
        let ok = Command::new(candidate)
            .args(["-c", "import cutlass"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            build_log!("Using {candidate} (cutlass already available)");
            return Ok(PathBuf::from(candidate));
        }
    }

    // 2. Otherwise, construct (or reuse) a persistent venv at venv_dir.
    let venv = PythonVenv::ensure(venv_dir).map_err(anyhow::Error::msg)?;
    if venv.check_import("import cutlass") {
        build_log!("Using cached venv at {}", venv_dir.display());
        return Ok(venv.python_path().to_path_buf());
    }

    let torch_index = detect_torch_cuda_index();
    if let Some(ref idx) = torch_index {
        build_log!("Detected CUDA index: {idx}");
    }
    build_log!("Installing FA4 Python deps (nvidia-cutlass-dsl, torch)...");

    venv.pip_install(
        &[
            "nvidia-cutlass-dsl>=4.4.2",
            "quack-kernels@git+https://github.com/Dao-AILab/quack.git",
            "torch",
            "einops",
        ],
        InstallOpts::new().extra_index_url(torch_index.as_deref()),
    )
    .map_err(anyhow::Error::msg)?;

    if !venv.check_import("import cutlass") {
        anyhow::bail!("cutlass not importable after install — check Python/CUDA setup");
    }

    build_log!("FA4 Python env ready at {}", venv_dir.display());
    Ok(venv.python_path().to_path_buf())
}

fn detect_gpu_arch() -> Result<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .context("nvidia-smi not found")?;

    if !output.status.success() {
        anyhow::bail!("nvidia-smi failed");
    }

    let cap = String::from_utf8_lossy(&output.stdout);
    let cap = cap.trim().lines().next().unwrap_or("9.0");
    let parts: Vec<&str> = cap.split('.').collect();
    if parts.len() == 2 {
        let major: u32 = parts[0].parse().unwrap_or(9);
        let minor: u32 = parts[1].parse().unwrap_or(0);
        Ok(format!("sm_{}{}", major, minor))
    } else {
        Ok("sm_90".to_string())
    }
}

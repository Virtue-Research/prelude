//! Build script for prelude-flashinfer.
//!
//! 1. Find FlashInfer source (FLASHINFER_SRC env var or auto-clone).
//! 2. Run scripts/compile_kernels.py to AOT-compile kernel variants.
//! 3. Archive all .o files into libflashinfer_kernels.a.
//! 4. Compile vendored tvm_ffi C++ (shared with FA4).
//! 5. Generate fi_dispatch.rs from manifest.json.
//!
//! All build artifacts go into OUT_DIR. `cargo clean` = full rebuild.

use anyhow::{Context, Result};
use std::env;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

const FLASHINFER_REPO: &str = "https://github.com/flashinfer-ai/flashinfer.git";
const FLASHINFER_BRANCH: &str = "main";

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=scripts/compile_kernels.py");
    println!("cargo:rerun-if-env-changed=FLASHINFER_SRC");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_ARCHS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_HEAD_DIMS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_DTYPES");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_WORKERS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_MLA_DIMS");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let kernels_dir = out_dir.join("kernels");

    // Phase 1: Find FlashInfer source
    let fi_src = find_flashinfer_source(&out_dir)?;

    // Phase 2: Compile kernels (incremental — .o timestamp check inside python)
    ensure_kernels(&kernels_dir, &manifest_dir, &fi_src)?;

    // Phase 3: Archive .o → .a
    let has_kernels = archive_objects(&kernels_dir, &out_dir)?;

    // Phase 4: Compile TVM FFI
    if env::var("CARGO_FEATURE_SKIP_TVM_FFI").is_ok() {
        println!("cargo:warning=FlashInfer: skipping tvm_ffi (provided by flash-attn-v4)");
    } else {
        compile_tvm_ffi(&manifest_dir)?;
    }

    // Phase 5: Generate dispatch code
    generate_dispatch(&kernels_dir, &out_dir, has_kernels)?;

    Ok(())
}

// ── FlashInfer source ────────────────────────────────────────────────

fn find_flashinfer_source(out_dir: &Path) -> Result<PathBuf> {
    if let Ok(src) = env::var("FLASHINFER_SRC") {
        let p = PathBuf::from(&src);
        if p.join("csrc").exists() {
            println!("cargo:warning=FlashInfer: using source at {src}");
            return Ok(p);
        }
        anyhow::bail!("FLASHINFER_SRC={src} does not contain csrc/");
    }

    let fi_src = out_dir.join("flashinfer-src");
    if fi_src.join("csrc").exists() {
        return Ok(fi_src);
    }

    println!("cargo:warning=FlashInfer: cloning {FLASHINFER_BRANCH}...");
    if fi_src.exists() {
        std::fs::remove_dir_all(&fi_src)?;
    }

    let status = Command::new("git")
        .args([
            "clone", "--depth", "1",
            "--branch", FLASHINFER_BRANCH,
            "--single-branch",
            "--filter=blob:limit=1m",
            FLASHINFER_REPO,
        ])
        .arg(&fi_src)
        .status()
        .context("git clone failed")?;
    if !status.success() {
        anyhow::bail!("git clone failed for {FLASHINFER_REPO}");
    }

    // Init submodules (tvm_ffi headers etc.)
    let _ = Command::new("git")
        .args(["submodule", "update", "--init", "--recursive", "--depth", "1"])
        .current_dir(&fi_src)
        .status();

    Ok(fi_src)
}

// ── Kernel compilation ───────────────────────────────────────────────

fn ensure_kernels(kernels_dir: &Path, manifest_dir: &Path, fi_src: &Path) -> Result<()> {
    // If manifest exists, kernels were already compiled (incremental is inside python)
    if kernels_dir.join("manifest.json").exists() {
        let n = walkdir_ext(kernels_dir, "o").len();
        println!("cargo:warning=FlashInfer: {n} kernel objects up-to-date");
        return Ok(());
    }

    println!("cargo:warning=FlashInfer: compiling kernels...");

    let script = manifest_dir.join("scripts/compile_kernels.py");
    let python = find_python()?;

    let archs = env::var("PRELUDE_FLASHINFER_ARCHS")
        .unwrap_or_else(|_| "sm_80,sm_90".to_string());
    let head_dims = env::var("PRELUDE_FLASHINFER_HEAD_DIMS")
        .unwrap_or_else(|_| "64,96,128,192,256".to_string());
    let dtypes = env::var("PRELUDE_FLASHINFER_DTYPES")
        .unwrap_or_else(|_| "bf16,fp16".to_string());
    let workers = env::var("PRELUDE_FLASHINFER_WORKERS").unwrap_or_else(|_| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .to_string()
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
        println!("cargo:warning=FlashInfer: no kernel .o files found");
        return Ok(false);
    }

    let archive = out_dir.join("libflashinfer_kernels.a");
    if !archive.exists() {
        println!("cargo:warning=FlashInfer: creating archive ({} objects)", obj_files.len());
        let mut cmd = Command::new("ar");
        cmd.arg("rcs").arg(&archive);
        for obj in &obj_files {
            cmd.arg(obj);
        }
        let status = cmd.status().context("ar failed")?;
        if !status.success() {
            anyhow::bail!("ar rcs failed");
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=flashinfer_kernels");
    Ok(true)
}

// ── TVM FFI ──────────────────────────────────────────────────────────

fn compile_tvm_ffi(manifest_dir: &Path) -> Result<()> {
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
        println!("cargo:warning=tvm_ffi vendor not found (FA4 crate needed)");
        return Ok(());
    };

    let cc_files: Vec<PathBuf> = walkdir_ext(&tvm_src, "cc")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap_or("");
            !name.contains("win") && !name.contains("testing") && name != "backtrace.cc"
        })
        .collect();

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

    let cuda_dialect = manifest_dir
        .parent().unwrap()
        .join("prelude-flash-attn-v4/vendor/cuda_dialect");
    if cuda_dialect.join("libcuda_dialect_runtime_static.a").exists() {
        println!("cargo:rustc-link-search=native={}", cuda_dialect.display());
        println!("cargo:rustc-link-lib=static:+whole-archive=cuda_dialect_runtime_static");
    }

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
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

// ── Dispatch codegen ─────────────────────────────────────────────────

fn generate_dispatch(kernels_dir: &Path, out_dir: &Path, has_kernels: bool) -> Result<()> {
    let path = out_dir.join("fi_dispatch.rs");

    if !has_kernels {
        std::fs::write(&path, concat!(
            "pub(crate) fn lookup_prefill(_key: &crate::loader::PrefillKey) -> Option<crate::loader::PrefillVariant> { None }\n",
            "pub(crate) fn lookup_prefill_fp8(_key: &crate::loader::FP8PrefillKey) -> Option<crate::loader::PrefillVariant> { None }\n",
            "pub(crate) fn lookup_decode(_key: &crate::loader::DecodeKey) -> Option<crate::loader::DecodeVariant> { None }\n",
            "pub(crate) fn lookup_mla_decode(_key: &crate::loader::MLADecodeKey) -> Option<crate::loader::MLADecodeVariant> { None }\n",
            "pub(crate) fn lookup_mla_paged(_key: &crate::loader::MLAPagedKey) -> Option<crate::loader::MLAPagedVariant> { None }\n",
            "pub(crate) fn lookup_utility(_name: &str) -> Option<crate::loader::TVMSafeCallFn> { None }\n",
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

    // Prefill lookup (FA2 + FA3, not FP8)
    writeln!(code, "pub(crate) fn lookup_prefill(key: &crate::loader::PrefillKey) -> Option<crate::loader::PrefillVariant> {{")?;
    writeln!(code, "    use crate::loader::PrefillVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_qk, key.head_dim_vo, key.sliding_window, key.logits_soft_cap, key.backend as u8) {{")?;
    for v in variants {
        let kind = v["kind"].as_str().unwrap();
        if kind != "prefill_fa2" && kind != "prefill_fa3" { continue; }
        let symbols = v["symbols"].as_object().unwrap();
        let (plan, ragged_run, paged_run) = (
            symbols["plan"].as_str().unwrap(),
            symbols["ragged_run"].as_str().unwrap(),
            symbols["paged_run"].as_str().unwrap(),
        );
        let dtype_val: u8 = if v["dtype"].as_str().unwrap() == "fp16" { 1 } else { 0 };
        let backend_val: u8 = if v["backend"].as_str().unwrap() == "fa3" { 1 } else { 0 };
        let (hqk, hvo) = (v["hdim_qk"].as_u64().unwrap(), v["hdim_vo"].as_u64().unwrap());
        let (swa, cap) = (v["swa"].as_bool().unwrap(), v["softcap"].as_bool().unwrap());
        writeln!(code, "        ({dtype_val}, {hqk}, {hvo}, {swa}, {cap}, {backend_val}) => Some(PrefillVariant {{ plan: {plan}, ragged_run: {ragged_run}, paged_run: {paged_run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // FP8 Prefill lookup
    writeln!(code, "pub(crate) fn lookup_prefill_fp8(key: &crate::loader::FP8PrefillKey) -> Option<crate::loader::PrefillVariant> {{")?;
    writeln!(code, "    use crate::loader::PrefillVariant;")?;
    writeln!(code, "    match (key.head_dim, key.sliding_window) {{")?;
    for v in variants {
        if v["kind"].as_str().unwrap() != "prefill_fp8" { continue; }
        let symbols = v["symbols"].as_object().unwrap();
        let (plan, ragged_run, paged_run) = (
            symbols["plan"].as_str().unwrap(),
            symbols["ragged_run"].as_str().unwrap(),
            symbols["paged_run"].as_str().unwrap(),
        );
        let hdim = v["hdim_qk"].as_u64().unwrap();
        let swa = v["swa"].as_bool().unwrap();
        writeln!(code, "        ({hdim}, {swa}) => Some(PrefillVariant {{ plan: {plan}, ragged_run: {ragged_run}, paged_run: {paged_run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // Decode lookup
    writeln!(code, "pub(crate) fn lookup_decode(key: &crate::loader::DecodeKey) -> Option<crate::loader::DecodeVariant> {{")?;
    writeln!(code, "    use crate::loader::DecodeVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_qk, key.head_dim_vo, key.sliding_window, key.logits_soft_cap) {{")?;
    for v in variants {
        if v["kind"].as_str().unwrap() != "decode" { continue; }
        let symbols = v["symbols"].as_object().unwrap();
        let (plan, run) = (symbols["plan"].as_str().unwrap(), symbols["run"].as_str().unwrap());
        let dtype_val: u8 = if v["dtype"].as_str().unwrap() == "fp16" { 1 } else { 0 };
        let (hqk, hvo) = (v["hdim_qk"].as_u64().unwrap(), v["hdim_vo"].as_u64().unwrap());
        let (swa, cap) = (v["swa"].as_bool().unwrap(), v["softcap"].as_bool().unwrap());
        writeln!(code, "        ({dtype_val}, {hqk}, {hvo}, {swa}, {cap}) => Some(DecodeVariant {{ plan: {plan}, run: {run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // MLA decode lookup
    writeln!(code, "pub(crate) fn lookup_mla_decode(key: &crate::loader::MLADecodeKey) -> Option<crate::loader::MLADecodeVariant> {{")?;
    writeln!(code, "    use crate::loader::MLADecodeVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_ckv, key.head_dim_kpe) {{")?;
    for v in variants {
        if v["kind"].as_str().unwrap() != "mla_decode" { continue; }
        let symbols = v["symbols"].as_object().unwrap();
        let (plan, run) = (symbols["plan"].as_str().unwrap(), symbols["run"].as_str().unwrap());
        let dtype_val: u8 = if v["dtype"].as_str().unwrap() == "fp16" { 1 } else { 0 };
        let (ckv, kpe) = (v["head_dim_ckv"].as_u64().unwrap(), v["head_dim_kpe"].as_u64().unwrap());
        writeln!(code, "        ({dtype_val}, {ckv}, {kpe}) => Some(MLADecodeVariant {{ plan: {plan}, run: {run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // MLA paged lookup
    writeln!(code, "pub(crate) fn lookup_mla_paged(key: &crate::loader::MLAPagedKey) -> Option<crate::loader::MLAPagedVariant> {{")?;
    writeln!(code, "    use crate::loader::MLAPagedVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_ckv, key.head_dim_kpe) {{")?;
    for v in variants {
        if v["kind"].as_str().unwrap() != "mla_paged" { continue; }
        let symbols = v["symbols"].as_object().unwrap();
        let (plan, run) = (symbols["plan"].as_str().unwrap(), symbols["run"].as_str().unwrap());
        let dtype_val: u8 = if v["dtype"].as_str().unwrap() == "fp16" { 1 } else { 0 };
        let (ckv, kpe) = (v["head_dim_ckv"].as_u64().unwrap(), v["head_dim_kpe"].as_u64().unwrap());
        writeln!(code, "        ({dtype_val}, {ckv}, {kpe}) => Some(MLAPagedVariant {{ plan: {plan}, run: {run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // Utility kernel lookup
    writeln!(code, "pub(crate) fn lookup_utility(name: &str) -> Option<crate::loader::TVMSafeCallFn> {{")?;
    writeln!(code, "    match name {{")?;
    for v in variants {
        let kind = v["kind"].as_str().unwrap();
        if !["page", "sampling", "norm", "rope", "cascade", "activation", "moe_routing"].contains(&kind) { continue; }
        let symbols = v["symbols"].as_object().unwrap();
        for (name, sym) in symbols {
            let s = sym.as_str().unwrap();
            writeln!(code, "        \"{name}\" => Some({s}),")?;
        }
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}")?;

    std::fs::write(&path, &code)?;
    println!("cargo:warning=FlashInfer: generated dispatch ({} variants)", variants.len());

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────

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

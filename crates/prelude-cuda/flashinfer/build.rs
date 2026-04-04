//! Build script for prelude-flashinfer.
//!
//! 1. Find FlashInfer source (third_party/flashinfer/ or FLASHINFER_SRC env var).
//! 2. Run scripts/compile_kernels.py to AOT-compile kernel variants.
//! 3. Archive all .o files into libflashinfer_kernels.a.
//! 4. Compile vendored tvm_ffi C++ (shared with FA4).
//! 5. Generate fi_dispatch.rs from manifest.json.
//!
//! All build artifacts go into OUT_DIR. `cargo clean` = full rebuild.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

include!("../../../build_log.rs");

// ── Manifest schema ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct Manifest {
    variants: Vec<Variant>,
}

#[derive(Deserialize)]
struct Variant {
    kind: String,
    symbols: HashMap<String, String>,
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    hdim_qk: Option<u64>,
    #[serde(default)]
    hdim_vo: Option<u64>,
    #[serde(default)]
    head_dim_ckv: Option<u64>,
    #[serde(default)]
    head_dim_kpe: Option<u64>,
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=scripts/compile_kernels.py");
    println!("cargo:rerun-if-env-changed=FLASHINFER_SRC");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_ARCHS");
    track_submodule("flashinfer");
    track_submodule("tvm-ffi");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_HEAD_DIMS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_DTYPES");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_WORKERS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_MLA_DIMS");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let kernels_dir = out_dir.join("kernels");

    // Phase 1: Find FlashInfer source
    let fi_src = find_flashinfer_source(&manifest_dir)?;


    // Phase 2: Compile kernels (incremental — .o timestamp check inside python)
    ensure_kernels(&kernels_dir, &manifest_dir, &fi_src)?;

    // Phase 3: Archive .o → .a
    let has_kernels = archive_objects(&kernels_dir, &out_dir)?;

    // Phase 4: Compile TVM FFI
    if env::var("CARGO_FEATURE_SKIP_TVM_FFI").is_ok() {
        build_log!("skipping tvm_ffi (provided by flash-attn-v4)");
    } else {
        compile_tvm_ffi(&manifest_dir)?;
    }

    // Phase 5: Generate dispatch code
    generate_dispatch(&kernels_dir, &out_dir, has_kernels)?;

    Ok(())
}

// ── FlashInfer source ────────────────────────────────────────────────

fn find_flashinfer_source(manifest_dir: &Path) -> Result<PathBuf> {
    // Priority 1: FLASHINFER_SRC env var (for development overrides)
    if let Ok(src) = env::var("FLASHINFER_SRC") {
        let p = PathBuf::from(&src);
        if p.join("csrc").exists() {
            build_log!("using source at {src}");
            return Ok(p);
        }
        anyhow::bail!("FLASHINFER_SRC={src} does not contain csrc/");
    }

    // Priority 2: third_party/flashinfer/ submodule (standard path)
    // flashinfer crate is at crates/prelude-cuda/flashinfer/ → 3 levels up to workspace root
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap().parent().unwrap();
    let fi_src = workspace_root.join("third_party/flashinfer");
    if fi_src.join("csrc").exists() {
        return Ok(fi_src);
    }

    anyhow::bail!(
        "FlashInfer source not found. Either:\n\
         1. Run: git submodule update --init --recursive third_party/flashinfer\n\
         2. Set FLASHINFER_SRC=/path/to/flashinfer"
    )
}


// ── Kernel compilation ───────────────────────────────────────────────

fn ensure_kernels(kernels_dir: &Path, manifest_dir: &Path, fi_src: &Path) -> Result<()> {
    let script = manifest_dir.join("scripts/compile_kernels.py");
    let manifest = kernels_dir.join("manifest.json");

    // Skip recompilation only if manifest exists AND is newer than the build script.
    if manifest.exists() {
        let script_mtime = script.metadata()?.modified()?;
        let manifest_mtime = manifest.metadata()?.modified()?;
        if manifest_mtime > script_mtime {
            let n = walkdir_ext(kernels_dir, "o").len();
            build_log!("{n} kernel objects up-to-date");
            return Ok(());
        }
        build_log!("compile_kernels.py changed, recompiling...");
        // Remove stale manifest so the script regenerates everything
        std::fs::remove_file(&manifest)?;
    } else {
        build_log!("compiling kernels...");
    }
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
        build_log!("no kernel .o files found");
        return Ok(false);
    }

    let archive = out_dir.join("libflashinfer_kernels.a");
    if !archive.exists() {
        build_log!("creating archive ({} objects)", obj_files.len());
        // Use `q` (quick append) not `r` (replace) — multiple variants have
        // identically named .o files (e.g. batch_decode.o) and `r` would
        // keep only the last one, losing symbols from other variants.
        let mut cmd = Command::new("ar");
        cmd.arg("qcs").arg(&archive);
        for obj in &obj_files {
            cmd.arg(obj);
        }
        let status = cmd.status().context("ar failed")?;
        if !status.success() {
            anyhow::bail!("ar qcs failed");
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=flashinfer_kernels");
    Ok(true)
}

// ── TVM FFI ──────────────────────────────────────────────────────────

fn compile_tvm_ffi(manifest_dir: &Path) -> Result<()> {
    // tvm-ffi lives in workspace third_party/
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap().parent().unwrap();
    let tvm_ffi_dir = workspace_root.join("third_party/tvm-ffi");

    let (tvm_src, tvm_include, dlpack_include) = if tvm_ffi_dir.join("src").exists() {
        (
            tvm_ffi_dir.join("src"),
            tvm_ffi_dir.join("include"),
            tvm_ffi_dir.join("3rdparty/dlpack/include"),
        )
    } else {
        anyhow::bail!(
            "third_party/tvm-ffi not found. Run: git submodule update --init third_party/tvm-ffi"
        );
    };

    let cc_files: Vec<PathBuf> = walkdir_ext(&tvm_src, "cc")
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_str().unwrap_or("");
            !name.contains("win") && !name.contains("testing")
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

    // Our C++ helper that uses upstream TVM FFI APIs for error extraction
    let tvm_error_helper = manifest_dir.join("src/tvm_error_helper.cc");
    build.file(&tvm_error_helper);

    // Compile libbacktrace (backtrace.cc references it)
    compile_libbacktrace(&tvm_ffi_dir)?;

    build.link_lib_modifier("+whole-archive");
    build.try_compile("tvm_ffi_fi_static")
        .context("Failed to compile tvm_ffi")?;

    // cuda_dialect_runtime_static.a — installed by cutlass-dsl in fa4-venv
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap().parent().unwrap();
    let venv_dir = workspace_root.join("target/fa4-venv");
    if let Some(p) = find_file_recursive(&venv_dir, "libcuda_dialect_runtime_static.a") {
        println!("cargo:rustc-link-search=native={}", p.parent().unwrap().display());
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
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=cuda");  // CUDA Driver API (comes with GPU driver)
    println!("cargo:rustc-link-lib=dylib=rt");   // required by cudart_static
    println!("cargo:rustc-link-lib=dylib=dl");   // required by cudart_static
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

// ── libbacktrace (required by tvm_ffi backtrace.cc) ────────────────

fn compile_libbacktrace(tvm_ffi_dir: &Path) -> Result<()> {
    let bt_dir = tvm_ffi_dir.join("3rdparty/libbacktrace");
    if !bt_dir.exists() {
        anyhow::bail!(
            "libbacktrace not found. Run: git submodule update --init --recursive third_party/tvm-ffi"
        );
    }

    // Generate config.h for the target platform
    let out = PathBuf::from(env::var("OUT_DIR")?);
    let config_dir = out.join("libbacktrace");
    std::fs::create_dir_all(&config_dir)?;

    let config_h = if cfg!(target_os = "linux") {
        r#"
#define BACKTRACE_ELF_SIZE 64
#define HAVE_ATOMIC_FUNCTIONS 1
#define HAVE_DL_ITERATE_PHDR 1
#define HAVE_DLFCN_H 1
#define HAVE_FCNTL 1
#define HAVE_LINK_H 1
#define HAVE_LSTAT 1
#define HAVE_READLINK 1
#define HAVE_SYS_MMAN_H 1
#define HAVE_DECL_STRNLEN 1
#define HAVE_DECL_GETPAGESIZE 0
"#
    } else if cfg!(target_os = "macos") {
        r#"
#define BACKTRACE_ELF_SIZE 64
#define HAVE_ATOMIC_FUNCTIONS 1
#define HAVE_DLFCN_H 1
#define HAVE_FCNTL 1
#define HAVE_MACH_O_DYLD_H 1
#define HAVE_SYS_MMAN_H 1
#define HAVE_DECL_STRNLEN 1
#define HAVE_DECL_GETPAGESIZE 0
"#
    } else {
        // Windows or unknown — minimal config
        r#"
#define HAVE_ATOMIC_FUNCTIONS 1
#define HAVE_DECL_STRNLEN 1
#define HAVE_DECL_GETPAGESIZE 0
"#
    };
    std::fs::write(config_dir.join("config.h"), config_h)?;

    // Select platform-specific source files
    let core_files = ["backtrace.c", "dwarf.c", "fileline.c", "posix.c",
                      "sort.c", "state.c", "alloc.c", "read.c", "mmapio.c", "mmap.c"];
    let format_file = if cfg!(target_os = "macos") { "macho.c" } else { "elf.c" };

    let mut build = cc::Build::new();
    build
        .opt_level(2)
        .pic(true)
        .include(&bt_dir)
        .include(&config_dir)  // for generated config.h
        .define("_GNU_SOURCE", None)
        .warnings(false);

    for name in &core_files {
        let f = bt_dir.join(name);
        if f.exists() {
            build.file(&f);
        }
    }
    let fmt = bt_dir.join(format_file);
    if fmt.exists() {
        build.file(&fmt);
    }

    build.try_compile("backtrace")
        .context("Failed to compile libbacktrace")?;

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
            "pub(crate) fn lookup_pod(_key: &crate::loader::PodKey) -> Option<crate::loader::PodVariant> { None }\n",
            "pub(crate) fn lookup_utility(_name: &str) -> Option<crate::loader::TVMSafeCallFn> { None }\n",
        ))?;
        return Ok(());
    }

    let manifest_str = std::fs::read_to_string(kernels_dir.join("manifest.json"))
        .context("manifest.json not found")?;
    let manifest: Manifest = serde_json::from_str(&manifest_str)
        .context("failed to parse manifest.json")?;

    let mut code = String::new();
    writeln!(code, "// AUTO-GENERATED by build.rs — do not edit")?;
    writeln!(code)?;

    // Helper: map dtype string to u8 (bf16=0, fp16=1)
    let dtype_val = |dt: &Option<String>| -> u8 {
        match dt.as_deref() {
            Some("fp16") => 1,
            _ => 0,
        }
    };

    // Extern declarations
    writeln!(code, "unsafe extern \"C\" {{")?;
    for v in &manifest.variants {
        for sym in v.symbols.values() {
            writeln!(code,
                "    fn {sym}(handle: *mut c_void, args: *const TVMFFIAny, num_args: i32, ret: *mut TVMFFIAny) -> i32;"
            )?;
        }
    }
    writeln!(code, "}}")?;
    writeln!(code)?;

    // Prefill lookup (FA2 merged + FA3 per-swa/softcap)
    // FA2: swa/softcap dispatched at runtime in CUDA — lookup by (dtype, hdim, backend) only
    // FA3: still per-swa/softcap (different variant type)
    writeln!(code, "pub(crate) fn lookup_prefill(key: &crate::loader::PrefillKey) -> Option<crate::loader::PrefillVariant> {{")?;
    writeln!(code, "    use crate::loader::PrefillVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_qk, key.head_dim_vo, key.sliding_window, key.logits_soft_cap, key.backend as u8) {{")?;
    for v in &manifest.variants {
        if v.kind != "prefill_fa2" && v.kind != "prefill_fa3" { continue; }
        let plan = &v.symbols["plan"];
        let ragged_run = &v.symbols["ragged_run"];
        let paged_run = &v.symbols["paged_run"];
        let dv = dtype_val(&v.dtype);
        let backend_val: u8 = if v.backend.as_deref() == Some("fa3") { 1 } else { 0 };
        let hqk = v.hdim_qk.context("prefill variant missing hdim_qk")?;
        let hvo = v.hdim_vo.context("prefill variant missing hdim_vo")?;
        // Both FA2 and FA3 are merged: match any swa/softcap (runtime dispatched in CUDA)
        for swa in [false, true] {
            for cap in [false, true] {
                writeln!(code, "        ({dv}, {hqk}, {hvo}, {swa}, {cap}, {backend_val}) => Some(PrefillVariant {{ plan: {plan}, ragged_run: {ragged_run}, paged_run: {paged_run} }}),")?;
            }
        }
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // FP8 Prefill lookup (merged: swa dispatched at runtime)
    writeln!(code, "pub(crate) fn lookup_prefill_fp8(key: &crate::loader::FP8PrefillKey) -> Option<crate::loader::PrefillVariant> {{")?;
    writeln!(code, "    use crate::loader::PrefillVariant;")?;
    writeln!(code, "    match (key.head_dim, key.sliding_window) {{")?;
    for v in &manifest.variants {
        if v.kind != "prefill_fp8" { continue; }
        let plan = &v.symbols["plan"];
        let ragged_run = &v.symbols["ragged_run"];
        let paged_run = &v.symbols["paged_run"];
        let hdim = v.hdim_qk.context("prefill_fp8 variant missing hdim_qk")?;
        // Merged: match any swa (runtime dispatched)
        for swa in [false, true] {
            writeln!(code, "        ({hdim}, {swa}) => Some(PrefillVariant {{ plan: {plan}, ragged_run: {ragged_run}, paged_run: {paged_run} }}),")?;
        }
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // Decode lookup (merged: swa/softcap dispatched at runtime in CUDA)
    writeln!(code, "pub(crate) fn lookup_decode(key: &crate::loader::DecodeKey) -> Option<crate::loader::DecodeVariant> {{")?;
    writeln!(code, "    use crate::loader::DecodeVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_qk, key.head_dim_vo, key.sliding_window, key.logits_soft_cap) {{")?;
    for v in &manifest.variants {
        if v.kind != "decode" { continue; }
        let plan = &v.symbols["plan"];
        let run = &v.symbols["run"];
        let dv = dtype_val(&v.dtype);
        let hqk = v.hdim_qk.context("decode variant missing hdim_qk")?;
        let hvo = v.hdim_vo.context("decode variant missing hdim_vo")?;
        // Merged: match any swa/softcap
        for swa in [false, true] {
            for cap in [false, true] {
                writeln!(code, "        ({dv}, {hqk}, {hvo}, {swa}, {cap}) => Some(DecodeVariant {{ plan: {plan}, run: {run} }}),")?;
            }
        }
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // MLA decode lookup
    writeln!(code, "pub(crate) fn lookup_mla_decode(key: &crate::loader::MLADecodeKey) -> Option<crate::loader::MLADecodeVariant> {{")?;
    writeln!(code, "    use crate::loader::MLADecodeVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_ckv, key.head_dim_kpe) {{")?;
    for v in &manifest.variants {
        if v.kind != "mla_decode" { continue; }
        let plan = &v.symbols["plan"];
        let run = &v.symbols["run"];
        let dv = dtype_val(&v.dtype);
        let ckv = v.head_dim_ckv.context("mla_decode variant missing head_dim_ckv")?;
        let kpe = v.head_dim_kpe.context("mla_decode variant missing head_dim_kpe")?;
        writeln!(code, "        ({dv}, {ckv}, {kpe}) => Some(MLADecodeVariant {{ plan: {plan}, run: {run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // MLA paged lookup
    writeln!(code, "pub(crate) fn lookup_mla_paged(key: &crate::loader::MLAPagedKey) -> Option<crate::loader::MLAPagedVariant> {{")?;
    writeln!(code, "    use crate::loader::MLAPagedVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_ckv, key.head_dim_kpe) {{")?;
    for v in &manifest.variants {
        if v.kind != "mla_paged" { continue; }
        let plan = &v.symbols["plan"];
        let run = &v.symbols["run"];
        let dv = dtype_val(&v.dtype);
        let ckv = v.head_dim_ckv.context("mla_paged variant missing head_dim_ckv")?;
        let kpe = v.head_dim_kpe.context("mla_paged variant missing head_dim_kpe")?;
        writeln!(code, "        ({dv}, {ckv}, {kpe}) => Some(MLAPagedVariant {{ plan: {plan}, run: {run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // POD (Prefill-On-Decode) lookup (merged: swa/softcap dispatched at runtime)
    writeln!(code, "pub(crate) fn lookup_pod(key: &crate::loader::PodKey) -> Option<crate::loader::PodVariant> {{")?;
    writeln!(code, "    use crate::loader::PodVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_qk, key.head_dim_vo) {{")?;
    for v in &manifest.variants {
        if v.kind != "pod_merged" { continue; }
        let run = &v.symbols["run"];
        let dv = dtype_val(&v.dtype);
        let hqk = v.hdim_qk.context("pod variant missing hdim_qk")?;
        let hvo = v.hdim_vo.context("pod variant missing hdim_vo")?;
        writeln!(code, "        ({dv}, {hqk}, {hvo}) => Some(PodVariant {{ run: {run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // Utility kernel lookup
    writeln!(code, "pub(crate) fn lookup_utility(name: &str) -> Option<crate::loader::TVMSafeCallFn> {{")?;
    writeln!(code, "    match name {{")?;
    for v in &manifest.variants {
        if !["page", "sampling", "norm", "rope", "cascade", "activation",
             "moe_routing", "fp4", "quantization", "fmha_sm100",
             "topk", "mla", "moe_utils", "moe", "gemm", "comm",
             "gdn", "mamba"].contains(&v.kind.as_str()) { continue; }
        for (name, sym) in &v.symbols {
            writeln!(code, "        \"{name}\" => Some({sym}),")?;
        }
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}")?;

    std::fs::write(&path, &code)?;
    build_log!("generated dispatch ({} variants)", manifest.variants.len());

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

fn find_file_recursive(dir: &Path, name: &str) -> Option<PathBuf> {
    if !dir.exists() { return None; }
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_file() && path.file_name().is_some_and(|n| n == name) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_file_recursive(&path, name) {
                return Some(found);
            }
        }
    }
    None
}

//! Build script for the `flashinfer` crate.
//!
//! Four phases:
//!
//!   1. **Locate source** — `FLASHINFER_SRC` env override or the
//!      `third_party/flashinfer` submodule. Marker: `csrc/` dir.
//!   2. **Run compile_kernels.py** — the Python script owns every
//!      per-kernel bit (jinja template expansion, nvcc command
//!      construction, per-variant symbol naming). Our build.rs just
//!      invokes it with the right CLI args and env vars.
//!   3. **Archive .o → .a** — flashinfer's compile script emits .o
//!      files with overlapping basenames across variants (e.g.
//!      `batch_decode.o` for each FA2 dtype/head_dim tuple), so we
//!      use `ar qcs` (Append mode) to keep them all. Replace mode
//!      would silently drop all but the last.
//!   4. **Dispatch codegen** — the only non-shared bit. flashinfer
//!      has five distinct lookup functions (prefill, prefill_fp8,
//!      decode, mla_decode, mla_paged, utility) with different key
//!      shapes, so we assemble the `match` arms inline here against
//!      the manifest entries. The extern-block emission still goes
//!      through `prelude_kernelbuild::dispatch`.
//!
//! Shared helpers used: `archive::{collect_obj_files,
//! archive_and_whole_link}`, `dispatch::{header_comment,
//! tvm_ffi_extern_block}`, `nvcc::{file_hash, track_submodule}`.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

use prelude_kernelbuild::archive::{self, ArMode};
use prelude_kernelbuild::build_log;
use prelude_kernelbuild::dispatch;
use prelude_kernelbuild::nvcc::track_submodule;

// ── Manifest schema ─────────────────────────────────────────────────
//
// The compile script writes a manifest.json per build describing
// every variant it produced, keyed by `kind`. Dispatch codegen
// walks the manifest and emits extern decls + per-kind match arms.

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
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=scripts/compile_kernels.py");
    println!("cargo:rerun-if-env-changed=FLASHINFER_SRC");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_ARCHS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_HEAD_DIMS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_DTYPES");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_WORKERS");
    println!("cargo:rerun-if-env-changed=PRELUDE_FLASHINFER_MLA_DIMS");
    track_submodule("flashinfer");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let kernels_dir = out_dir.join("kernels");

    let fi_src = find_flashinfer_source(&manifest_dir)?;
    ensure_kernels(&kernels_dir, &manifest_dir, &fi_src)?;

    // Phase 3: archive + whole-archive link. Use Append mode because
    // compile_kernels.py emits .o files with overlapping basenames
    // across variants. Archive is removed-and-rebuilt every time to
    // prevent stale members from lingering (see ArMode::Append doc).
    let objects = archive::collect_obj_files(&kernels_dir);
    let has_kernels = archive::archive_and_whole_link(
        &objects,
        &out_dir,
        "flashinfer_kernels",
        ArMode::Append,
    )
    .map_err(anyhow::Error::msg)?;

    generate_dispatch(&kernels_dir, &out_dir, has_kernels)?;

    Ok(())
}

// ── FlashInfer source ────────────────────────────────────────────────

fn find_flashinfer_source(manifest_dir: &Path) -> Result<PathBuf> {
    // Priority 1: FLASHINFER_SRC env var (for development overrides).
    if let Ok(src) = env::var("FLASHINFER_SRC") {
        let p = PathBuf::from(&src);
        if p.join("csrc").exists() {
            build_log!("using source at {src}");
            return Ok(p);
        }
        anyhow::bail!("FLASHINFER_SRC={src} does not contain csrc/");
    }

    // Priority 2: third_party/flashinfer/ submodule (standard path).
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

    // Script-mtime vs manifest-mtime cache check. flashinfer's compile
    // script does its own fine-grained .o mtime checks internally, so
    // we only short-circuit at the script level — any change to the
    // script rebuilds everything, any change to a single .cu source
    // is handled by the script itself.
    if manifest.exists() {
        let script_mtime = script.metadata()?.modified()?;
        let manifest_mtime = manifest.metadata()?.modified()?;
        if manifest_mtime > script_mtime {
            let n = archive::collect_obj_files(kernels_dir).len();
            build_log!("{n} kernel objects up-to-date");
            return Ok(());
        }
        build_log!("compile_kernels.py changed, recompiling...");
        std::fs::remove_file(&manifest)?;
    } else {
        build_log!("compiling kernels...");
    }

    let python = find_python()?;

    let archs = env::var("PRELUDE_FLASHINFER_ARCHS").unwrap_or_else(|_| "sm_80,sm_90".to_string());
    let head_dims = env::var("PRELUDE_FLASHINFER_HEAD_DIMS")
        .unwrap_or_else(|_| "64,96,128,192,256,512".to_string());
    let dtypes = env::var("PRELUDE_FLASHINFER_DTYPES").unwrap_or_else(|_| "bf16,fp16".to_string());
    let workers = env::var("PRELUDE_FLASHINFER_WORKERS").unwrap_or_else(|_| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .to_string()
    });

    let status = Command::new(&python)
        .arg(&script)
        .arg("--flashinfer-src")
        .arg(fi_src)
        .arg("--out-dir")
        .arg(kernels_dir)
        .arg("--archs")
        .arg(&archs)
        .arg("--head-dims")
        .arg(&head_dims)
        .arg("--dtypes")
        .arg(&dtypes)
        .arg("-j")
        .arg(&workers)
        .status()
        .context("Failed to run compile_kernels.py")?;

    if !status.success() {
        anyhow::bail!("compile_kernels.py failed");
    }

    Ok(())
}

fn find_python() -> Result<PathBuf> {
    for candidate in ["python3", "python"] {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Ok(PathBuf::from(candidate));
        }
    }
    anyhow::bail!("Python 3 not found")
}

// ── Dispatch codegen ─────────────────────────────────────────────────
//
// FlashInfer has six lookup functions, each keyed on a different set
// of shape/dtype params. Because the match arm shapes are entirely
// flashinfer-specific we can't use a generic codegen helper — only
// the extern block + the boilerplate header come from the shared
// `prelude_kernelbuild::dispatch` module.

fn generate_dispatch(kernels_dir: &Path, out_dir: &Path, has_kernels: bool) -> Result<()> {
    let path = out_dir.join("fi_dispatch.rs");

    if !has_kernels {
        // Every lookup stubbed to return None — the consumer crate
        // already knows how to fall back to a CPU path when a kernel
        // isn't available, so this is safe even in the "no kernels
        // compiled" case (headless CI build host without nvcc).
        std::fs::write(&path, concat!(
            "// AUTO-GENERATED by build.rs via prelude-kernelbuild — no kernels compiled\n\n",
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
    let manifest: Manifest =
        serde_json::from_str(&manifest_str).context("failed to parse manifest.json")?;

    let mut code = dispatch::header_comment();

    // Extern declarations. The generated file is `include!`-ed into
    // `loader.rs` which already has `c_void` and `TVMFFIAny` in
    // scope, so we pass `TVMFFIAny` as the path (unprefixed).
    let symbols_iter = manifest
        .variants
        .iter()
        .flat_map(|v| v.symbols.values().map(String::as_str));
    code.push_str(&dispatch::tvm_ffi_extern_block(symbols_iter, "TVMFFIAny"));

    // Helper: map dtype string to u8 (bf16=0, fp16=1).
    let dtype_val = |dt: &Option<String>| -> u8 {
        match dt.as_deref() {
            Some("fp16") => 1,
            _ => 0,
        }
    };

    // Prefill lookup (FA2 merged + FA3 per-swa/softcap).
    // FA2: swa/softcap dispatched at runtime in CUDA — lookup by
    // (dtype, hdim, backend) only.
    // FA3: still per-swa/softcap (different variant type).
    writeln!(code, "pub(crate) fn lookup_prefill(key: &crate::loader::PrefillKey) -> Option<crate::loader::PrefillVariant> {{")?;
    writeln!(code, "    use crate::loader::PrefillVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_qk, key.head_dim_vo, key.sliding_window, key.logits_soft_cap, key.backend as u8) {{")?;
    for v in &manifest.variants {
        if v.kind != "prefill_fa2" && v.kind != "prefill_fa3" {
            continue;
        }
        let plan = &v.symbols["plan"];
        let ragged_run = &v.symbols["ragged_run"];
        let paged_run = &v.symbols["paged_run"];
        let dv = dtype_val(&v.dtype);
        let backend_val: u8 = if v.backend.as_deref() == Some("fa3") { 1 } else { 0 };
        let hqk = v.hdim_qk.context("prefill variant missing hdim_qk")?;
        let hvo = v.hdim_vo.context("prefill variant missing hdim_vo")?;
        // Both FA2 and FA3 are merged: match any swa/softcap
        // (runtime dispatched in CUDA).
        for swa in [false, true] {
            for cap in [false, true] {
                writeln!(code, "        ({dv}, {hqk}, {hvo}, {swa}, {cap}, {backend_val}) => Some(PrefillVariant {{ plan: {plan}, ragged_run: {ragged_run}, paged_run: {paged_run} }}),")?;
            }
        }
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // FP8 Prefill lookup (merged: swa dispatched at runtime).
    writeln!(code, "pub(crate) fn lookup_prefill_fp8(key: &crate::loader::FP8PrefillKey) -> Option<crate::loader::PrefillVariant> {{")?;
    writeln!(code, "    use crate::loader::PrefillVariant;")?;
    writeln!(code, "    match (key.head_dim, key.sliding_window) {{")?;
    for v in &manifest.variants {
        if v.kind != "prefill_fp8" {
            continue;
        }
        let plan = &v.symbols["plan"];
        let ragged_run = &v.symbols["ragged_run"];
        let paged_run = &v.symbols["paged_run"];
        let hdim = v.hdim_qk.context("prefill_fp8 variant missing hdim_qk")?;
        // Merged: match any swa (runtime dispatched).
        for swa in [false, true] {
            writeln!(code, "        ({hdim}, {swa}) => Some(PrefillVariant {{ plan: {plan}, ragged_run: {ragged_run}, paged_run: {paged_run} }}),")?;
        }
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // Decode lookup (merged: swa/softcap dispatched at runtime in CUDA).
    writeln!(code, "pub(crate) fn lookup_decode(key: &crate::loader::DecodeKey) -> Option<crate::loader::DecodeVariant> {{")?;
    writeln!(code, "    use crate::loader::DecodeVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_qk, key.head_dim_vo, key.sliding_window, key.logits_soft_cap) {{")?;
    for v in &manifest.variants {
        if v.kind != "decode" {
            continue;
        }
        let plan = &v.symbols["plan"];
        let run = &v.symbols["run"];
        let dv = dtype_val(&v.dtype);
        let hqk = v.hdim_qk.context("decode variant missing hdim_qk")?;
        let hvo = v.hdim_vo.context("decode variant missing hdim_vo")?;
        for swa in [false, true] {
            for cap in [false, true] {
                writeln!(code, "        ({dv}, {hqk}, {hvo}, {swa}, {cap}) => Some(DecodeVariant {{ plan: {plan}, run: {run} }}),")?;
            }
        }
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // MLA decode lookup.
    writeln!(code, "pub(crate) fn lookup_mla_decode(key: &crate::loader::MLADecodeKey) -> Option<crate::loader::MLADecodeVariant> {{")?;
    writeln!(code, "    use crate::loader::MLADecodeVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_ckv, key.head_dim_kpe) {{")?;
    for v in &manifest.variants {
        if v.kind != "mla_decode" {
            continue;
        }
        let plan = &v.symbols["plan"];
        let run = &v.symbols["run"];
        let dv = dtype_val(&v.dtype);
        let ckv = v.head_dim_ckv.context("mla_decode variant missing head_dim_ckv")?;
        let kpe = v.head_dim_kpe.context("mla_decode variant missing head_dim_kpe")?;
        writeln!(code, "        ({dv}, {ckv}, {kpe}) => Some(MLADecodeVariant {{ plan: {plan}, run: {run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // MLA paged lookup.
    writeln!(code, "pub(crate) fn lookup_mla_paged(key: &crate::loader::MLAPagedKey) -> Option<crate::loader::MLAPagedVariant> {{")?;
    writeln!(code, "    use crate::loader::MLAPagedVariant;")?;
    writeln!(code, "    match (key.dtype as u8, key.head_dim_ckv, key.head_dim_kpe) {{")?;
    for v in &manifest.variants {
        if v.kind != "mla_paged" {
            continue;
        }
        let plan = &v.symbols["plan"];
        let run = &v.symbols["run"];
        let dv = dtype_val(&v.dtype);
        let ckv = v.head_dim_ckv.context("mla_paged variant missing head_dim_ckv")?;
        let kpe = v.head_dim_kpe.context("mla_paged variant missing head_dim_kpe")?;
        writeln!(code, "        ({dv}, {ckv}, {kpe}) => Some(MLAPagedVariant {{ plan: {plan}, run: {run} }}),")?;
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}\n")?;

    // Utility kernel lookup — everything that doesn't fit the structured
    // lookups above (norm, rope, sampling, moe, ...). Keyed by the
    // string name the runtime loader passes.
    writeln!(code, "pub(crate) fn lookup_utility(name: &str) -> Option<crate::loader::TVMSafeCallFn> {{")?;
    writeln!(code, "    match name {{")?;
    for v in &manifest.variants {
        if ![
            "page", "sampling", "norm", "rope", "cascade", "activation", "moe_routing", "fp4",
            "quantization", "fmha_sm100", "topk", "mla", "moe_utils", "moe", "gemm", "comm",
            "gdn", "mamba",
        ]
        .contains(&v.kind.as_str())
        {
            continue;
        }
        for (name, sym) in &v.symbols {
            writeln!(code, "        \"{name}\" => Some({sym}),")?;
        }
    }
    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}\n}}")?;

    std::fs::write(&path, &code)?;
    dispatch::log_generated("flashinfer", manifest.variants.len());

    Ok(())
}

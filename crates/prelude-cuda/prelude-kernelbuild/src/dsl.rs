//! Driver for running per-crate `compile_kernels.py` CuTeDSL AOT
//! scripts and producing TVM-FFI `.o` files under
//! `OUT_DIR/<kernels_dir>/<arch>/`.
//!
//! Before this module existed, `cula/build.rs` and `fa4/build.rs` each
//! had their own ~150-line copy of "hash the script, check the cache,
//! spawn the Python process per-arch, deal with exit codes, log stderr
//! tails". The only per-crate parts are really:
//!
//!   1. Which compile script to invoke (`scripts/compile_kernels.py`).
//!   2. Which env vars to set when spawning it (cuLA uses
//!      `CUTE_DSL_ENABLE_TVM_FFI`, fa4 uses `FLASH_ATTENTION_*`, etc.).
//!   3. Whether to wrap the script in a Python `-c <bootstrap>` snippet
//!      (cuLA does, to monkey-patch `cula.utils.assert_hopper` for
//!      cross-compile).
//!
//! Everything else (cache invalidation via script hash, per-arch
//! subdirectory layout, sticky failure marker, exit-code logging) is
//! uniform across consumers and now lives in [`run`].
//!
//! ## Cache strategy
//!
//! Each per-arch directory contains a `manifest.json` with a
//! `script_hash` field (written by the Python script after a
//! successful compile). On the next build, [`run`] reads the stored
//! hash, compares against the current file hash of the compile script,
//! and short-circuits if they match. Touching `compile_kernels.py`
//! clears the cache; editing `build.rs` does **not** (the build.rs is
//! driven by Cargo's own rerun-if-changed, which is separate).
//!
//! ## Sticky failure
//!
//! Some compile-script failures are deterministic: missing Python
//! dependency, upstream kernel bug, mismatched nvcc version. Retrying
//! them on every `cargo build` wastes minutes. When `sticky_failure`
//! is set, a failed compile writes `.dsl_compile_failed` with the
//! script's current hash; subsequent builds with the same hash
//! short-circuit via a warning instead of re-running the compile.
//! Touching the compile script clears the marker.

use std::path::Path;
use std::process::Command;

use crate::build_log;
use crate::nvcc::file_hash;
use crate::scripts_dir;

/// Spec for a single DSL compile invocation. Construct, pass to
/// [`run`], and optionally inspect the returned bool to decide whether
/// to archive+link downstream.
pub struct DslCompile<'a> {
    /// Python binary to invoke. Typically
    /// `venv.python_path().to_path_buf()` after provisioning via
    /// [`crate::venv::PythonVenv`].
    pub python: &'a Path,

    /// Path to the consumer's compile_kernels.py. Its SHA-256 prefix
    /// is used as the cache key; touching the script invalidates the
    /// cached `.o` files.
    pub script: &'a Path,

    /// Root directory under which per-arch subdirectories
    /// (`sm_90/`, `sm_100/`) get written. Usually
    /// `out_dir.join("dsl_kernels")`.
    pub kernels_dir: &'a Path,

    /// Target GPU arches as lowercase strings like `"sm_90"`,
    /// `"sm_100"`. Each becomes a subdirectory; the per-arch env var
    /// (see [`Self::arch_env_var`]) is set to `{arch}a`, matching the
    /// `CUTE_DSL_ARCH` convention cuLA and fa4 already use.
    pub target_archs: &'a [String],

    /// Worker count passed to the compile script's `-j` flag. cuLA
    /// forces `"1"` (its per-kernel `cute.compile` calls touch CUDA,
    /// breaking fork-based pools); fa4 goes up to `min(ncpu, 8)`.
    pub workers: &'a str,

    /// Env vars to set for every per-arch invocation. Consumers use
    /// this for script-specific configuration (e.g. fa4's
    /// `FLASH_ATTENTION_FAKE_TENSOR=1`). Not including the arch var —
    /// [`Self::arch_env_var`] handles that.
    pub env: &'a [(&'a str, String)],

    /// Name of the env var that receives the per-arch `sm_XXa`
    /// string. Both current consumers set `"CUTE_DSL_ARCH"`. Can be
    /// left empty to skip this step.
    pub arch_env_var: &'a str,

    /// Optional Python bootstrap snippet. When `Some(code)`, the
    /// command becomes `python -c <code> <script> <args...>` instead
    /// of `python <script> <args...>`. cuLA uses this to monkey-patch
    /// `cula.utils.get_device_sm_version` for cross-compile.
    pub bootstrap: Option<&'a str>,

    /// Label shown in the `[DSL]` log lines. Usually just the consumer
    /// crate name, e.g. `"cuLA"` or `"FA4"`.
    pub label: &'a str,

    /// If true, a failed compile with a specific script hash writes a
    /// `.dsl_compile_failed` marker so subsequent builds short-circuit
    /// instead of retrying. Touching the compile script clears the
    /// marker.
    pub sticky_failure: bool,
}

/// Run the DSL compile phase described by `spec`. Returns:
///
///   * `Ok(true)`  — at least one `.o` file exists under `kernels_dir`
///                   after the compile (or was already cached). The
///                   caller should proceed with archive+link.
///   * `Ok(false)` — the compile produced no `.o` files (Python env
///                   missing, sticky failure, script missing). Caller
///                   emits a stub dispatch table and skips linking.
///   * `Err(e)`    — catastrophic failure (hash computation IO error,
///                   etc.) where even graceful fallback isn't safe.
///                   Rare.
///
/// Cache hits log `[DSL] using cached kernels` and return without
/// spawning Python.
pub fn run(spec: &DslCompile<'_>) -> Result<bool, String> {
    let kernels_dir = spec.kernels_dir;
    let script = spec.script;

    // ── Cache hit check ─────────────────────────────────────────────
    if let Some(hash) = file_hash(script) {
        if kernels_dir.join("manifest.json").exists() && has_obj_files(kernels_dir) {
            let manifest_str = std::fs::read_to_string(kernels_dir.join("manifest.json"))
                .unwrap_or_default();
            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                let stored = manifest
                    .get("script_hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if stored == hash {
                    build_log!("[DSL] {} using cached kernels", spec.label);
                    return Ok(true);
                }
                build_log!(
                    "[DSL] {} compile_kernels.py changed ({}→{}), recompiling...",
                    spec.label,
                    short_hash(stored),
                    short_hash(&hash)
                );
                clear_obj_files(kernels_dir);
            }
        }
    }

    // ── Sticky failure check ────────────────────────────────────────
    let fail_marker = kernels_dir.join(".dsl_compile_failed");
    if spec.sticky_failure {
        if let Some(hash) = file_hash(script) {
            if fail_marker.exists() {
                let stored = std::fs::read_to_string(&fail_marker).unwrap_or_default();
                if stored.trim() == hash {
                    build_log!(
                        "[DSL] {} skipping (previously failed, script unchanged)",
                        spec.label
                    );
                    return Ok(false);
                }
            }
        }
    }

    build_log!("[DSL] {} attempting AOT compilation...", spec.label);
    if !script.exists() {
        build_log!("[DSL] {} no compile script, skipping DSL kernels", spec.label);
        return Ok(false);
    }

    build_log!(
        "[DSL] {} AOT compiling for {:?}...",
        spec.label,
        spec.target_archs
    );

    for arch in spec.target_archs {
        build_log!("[DSL] {} compiling for {arch}...", spec.label);
        let arch_dir = kernels_dir.join(arch);
        let _ = std::fs::create_dir_all(&arch_dir);

        let mut cmd = Command::new(spec.python);
        if let Some(code) = spec.bootstrap {
            cmd.arg("-c").arg(code);
        }
        cmd.arg(script)
            .arg("--output-dir")
            .arg(&arch_dir)
            .args(["-j", spec.workers]);

        // Expose prelude-kernelbuild's `scripts/` directory to the
        // compile script so it can import the shared `dsl_driver`
        // helpers (symbol verification, manifest IO, script-hash
        // invalidation, parallel runner). Scripts that don't need the
        // helpers can simply not import them.
        cmd.env("PRELUDE_KB_SCRIPTS_DIR", scripts_dir());

        for (k, v) in spec.env {
            cmd.env(k, v);
        }
        if !spec.arch_env_var.is_empty() {
            cmd.env(spec.arch_env_var, format!("{arch}a"));
        }

        match cmd.output() {
            Ok(o) if o.status.success() => {
                build_log!("[DSL] {} {arch} succeeded", spec.label);
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                let stdout = String::from_utf8_lossy(&o.stdout);
                build_log!(
                    "[DSL] {} {arch} failed (exit code: {:?})",
                    spec.label,
                    o.status.code()
                );
                for line in stderr
                    .lines()
                    .rev()
                    .take(10)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                {
                    build_log!("[DSL]   {line}");
                }
                if !stdout.is_empty() {
                    for line in stdout
                        .lines()
                        .rev()
                        .take(5)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                    {
                        build_log!("[DSL]   [stdout] {line}");
                    }
                }
            }
            Err(e) => {
                build_log!("[DSL] {} failed to run script: {e}", spec.label);
            }
        }
    }

    let success = has_obj_files(kernels_dir);
    if !success && spec.sticky_failure {
        if let Some(hash) = file_hash(script) {
            let _ = std::fs::create_dir_all(kernels_dir);
            let _ = std::fs::write(&fail_marker, &hash);
        }
    }
    Ok(success)
}

/// Recursively check whether any `.o` file exists under `dir`. Used
/// as the "did the compile actually do anything?" signal.
fn has_obj_files(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|x| x == "o") {
            return true;
        }
        if p.is_dir() && has_obj_files(&p) {
            return true;
        }
    }
    false
}

/// Drop every `.o` file at the top level of `dir` plus its
/// `manifest.json`. Used when the compile script's hash doesn't match
/// the stored one — rather than try to figure out which specific
/// kernels changed we just nuke the cache and let the Python script
/// repopulate.
fn clear_obj_files(dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|x| x == "o") {
                let _ = std::fs::remove_file(&p);
            }
        }
    }
    let _ = std::fs::remove_file(dir.join("manifest.json"));
}

/// First 8 hex chars of a hash string, for readable `changed (XXX→YYY)` logs.
fn short_hash(h: &str) -> String {
    if h.is_empty() {
        "∅".to_string()
    } else {
        let n = h.len().min(8);
        h[..n].to_string()
    }
}

/// Bootstrap Python snippet that runs `compile_kernels.py` via
/// `runpy.run_path`, monkey-patching `cula.utils.assert_hopper` /
/// `assert_blackwell` first so the AOT compile works on any host GPU
/// (arch is read from `CUTE_DSL_ARCH` instead of probing the live
/// device). Only cuLA needs this today — fa4's compile script uses
/// `FLASH_ATTENTION_FAKE_TENSOR=1` to get the same effect through
/// upstream hooks instead.
///
/// Exposed as a constant so the cuLA build script can pass it via
/// [`DslCompile::bootstrap`] without re-embedding the literal.
pub const CULA_BOOTSTRAP: &str = r#"
import sys, os, runpy
arch = int(''.join(c for c in os.environ.get('CUTE_DSL_ARCH','sm_90a').split('_')[1] if c.isdigit()))
major, minor = arch // 10, arch % 10
import cula.utils
cula.utils.get_device_sm_version = lambda device=None: (major, minor)
cula.utils.assert_blackwell = lambda device=None: None
cula.utils.assert_hopper = lambda device=None: None
script_path = sys.argv[1]
sys.argv = [script_path] + sys.argv[2:]
runpy.run_path(script_path, run_name='__main__')
"#;

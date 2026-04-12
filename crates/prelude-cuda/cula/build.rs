//! Build script for the `cula` crate — Rust bindings to cuLA linear
//! attention kernels.
//!
//! Three phases:
//!
//!   * **Phase 1 (nvcc C++)**: Compile C++ CUTLASS 3.x KDA kernels
//!     (SM90 always, SM100 when the toolkit supports it) into
//!     `libcula_cpp.a`. Uses [`prelude_kernelbuild::nvcc`].
//!   * **Phase 2 (CuTe DSL Python)**: AOT compile the CuTe DSL kernels
//!     (`kda_decode`, `lightning_prefill`, `chunk_delta_h`, `fwd_o`)
//!     by invoking `scripts/compile_kernels.py` inside a provisioned
//!     Python venv. Produces `.o` files under
//!     `OUT_DIR/dsl_kernels/<arch>/` with TVM-FFI symbols. Gated behind
//!     the `dsl` cargo feature (on by default). Runs in parallel with
//!     Phase 1.
//!   * **Phase 3 (dispatch codegen)**: Walk the per-arch manifests from
//!     Phase 2, archive all `.o` files into `libcula_dsl_kernels.a`
//!     with whole-archive linking, and emit `cula_dsl_dispatch.rs` with
//!     extern declarations and a `lookup_dsl(key, arch) → fn_ptr`
//!     match. Uses [`prelude_kernelbuild::dispatch`] +
//!     [`prelude_kernelbuild::archive`].
//!
//! Shared build-support lives in `prelude_kernelbuild`; only the
//! cuLA-specific glue (which sources to compile, which packages to
//! install in the venv, how the compile script is invoked) stays here.

use anyhow::Result;
use std::env;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

use prelude_kernelbuild::archive;
use prelude_kernelbuild::build_log;
use prelude_kernelbuild::dispatch;
use prelude_kernelbuild::nvcc::{
    compile_cu_to_obj, file_hash, find_cuda, link_cuda_runtime_static, locate_source, nvcc_path,
    nvcc_supports_sm100, track_submodule, ObjCompile,
};
use prelude_kernelbuild::venv::{detect_torch_cuda_index, InstallOpts, PythonVenv};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cula_wrapper.cu");
    println!("cargo:rerun-if-changed=scripts/compile_kernels.py");
    println!("cargo:rerun-if-env-changed=CULA_ROOT");
    println!("cargo:rerun-if-env-changed=CUTLASS_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    track_submodule("cuLA");
    track_submodule("cutlass");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Source discovery:
    //   - `CULA_ROOT`    overrides cuLA source path
    //   - `CUTLASS_ROOT` overrides CUTLASS include path
    // Both fall back to the prelude workspace layout so zero-config
    // builds inside the monorepo "just work".
    let workspace_root = manifest_dir.join("../../..");
    let cula_dir = locate_source(
        "CULA_ROOT",
        "cuLA",
        "csrc/kda/sm90/prefill_kernel.hpp",
        &workspace_root.join("third_party/cuLA"),
    );
    let cutlass_dir = locate_source(
        "CUTLASS_ROOT",
        "cutlass",
        "include/cutlass/cutlass.h",
        &workspace_root.join("third_party/cutlass"),
    );

    let cuda_path = find_cuda();
    let nvcc = nvcc_path(&cuda_path);
    let sm100 = nvcc_supports_sm100(&nvcc);

    // ── Phase 1 + Phase 2 run in parallel ──────────────────────────
    //   Phase 1: nvcc compiles C++ CUTLASS kernels (uses GPU compiler)
    //   Phase 2: Python compiles CuTe DSL kernels (uses cutlass-dsl)
    // They share no state and use different tools → safe to parallelize.

    let dsl_enabled = env::var("CARGO_FEATURE_DSL").is_ok();
    let kernels_dir = out_dir.join("dsl_kernels");

    // Target arch list: SM90 always, SM100 if nvcc supports it.
    let mut dsl_archs = vec!["sm_90".to_string()];
    if sm100 {
        dsl_archs.push("sm_100".to_string());
    }

    // Spawn Phase 2 in a background thread; venv setup can take
    // minutes on a clean build.
    let dsl_kernels_dir = kernels_dir.clone();
    let dsl_manifest_dir = manifest_dir.clone();
    let dsl_cula_dir = cula_dir.clone();
    let dsl_out_dir = out_dir.clone();
    let dsl_handle = std::thread::spawn(move || {
        if !dsl_enabled {
            build_log!("[DSL] feature disabled, skipping Python AOT compile");
            return false;
        }
        compile_dsl_kernels(
            &dsl_kernels_dir,
            &dsl_manifest_dir,
            &dsl_cula_dir,
            &dsl_out_dir,
            &dsl_archs,
        )
    });

    // Phase 1 runs on the main thread.
    let cpp_objects = compile_cpp_kernels(
        &nvcc,
        &cula_dir,
        &cutlass_dir,
        &cuda_path,
        &out_dir,
        &manifest_dir,
        sm100,
    );

    // Archive the C++ objects via nvcc --lib so any fatbin sections
    // survive. (A plain `ar rcs` would work, but nvcc --lib is what
    // upstream cuLA builds use and we stay consistent with that.)
    if !cpp_objects.is_empty() {
        let lib = out_dir.join("libcula_cpp.a");
        let mut cmd = Command::new(&nvcc);
        cmd.arg("--lib").args(["-o", lib.to_str().unwrap()]);
        for obj in &cpp_objects {
            cmd.arg(obj);
        }
        let status = cmd.status().expect("Failed to run nvcc --lib");
        if !status.success() {
            panic!("nvcc --lib failed for libcula_cpp.a");
        }
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=cula_cpp");
    }

    // Wait for Phase 2 to finish.
    let has_dsl_kernels = dsl_handle.join().expect("DSL compilation thread panicked");

    // ── Phase 3: archive + dispatch codegen ─────────────────────────
    if has_dsl_kernels {
        let objects = archive::collect_obj_files(&kernels_dir);
        let _ = archive::archive_and_whole_link(&objects, &out_dir, "cula_dsl_kernels")
            .map_err(|e| panic!("{e}"));
    }
    generate_dispatch_code(&kernels_dir, &out_dir, has_dsl_kernels);

    // ── Link CUDA runtime ───────────────────────────────────────────
    link_cuda_runtime_static(&cuda_path);
}

// ─────────────────────────────────────────────────────────────────────
// Phase 1: C++ CUTLASS kernels
// ─────────────────────────────────────────────────────────────────────

fn compile_cpp_kernels(
    nvcc: &Path,
    cula_dir: &Path,
    cutlass_dir: &Path,
    cuda_path: &Path,
    out_dir: &Path,
    manifest_dir: &Path,
    sm100: bool,
) -> Vec<PathBuf> {
    let include_dirs: Vec<PathBuf> = vec![
        cutlass_dir.join("include"),
        cutlass_dir.join("tools/util/include"),
        cula_dir.join("csrc"),
        cula_dir.join("csrc/kerutils/include"),
        cuda_path.join("include"),
    ];

    let sm90_sources = [
        cula_dir.join("csrc/kda/sm90/kda_fwd_sm90.cu"),
        cula_dir.join("csrc/kda/sm90/kda_fwd_sm90_safe_gate.cu"),
    ];
    let wrapper_src = manifest_dir.join("src/cula_wrapper.cu");

    // Build (src, obj, gencodes, defines) tasks and compile them all
    // in parallel — the upstream kernels are big CUTLASS templates and
    // each takes ~30s of nvcc time, so serialising would cost minutes.
    #[derive(Clone)]
    struct CompileTask {
        src: PathBuf,
        obj: PathBuf,
        gencodes: Vec<String>,
        defines: Vec<String>,
    }

    let mut tasks: Vec<CompileTask> = Vec::new();

    for src in &sm90_sources {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        tasks.push(CompileTask {
            src: src.clone(),
            obj: out_dir.join(format!("{stem}.o")),
            gencodes: vec!["-gencode=arch=compute_90a,code=sm_90a".into()],
            defines: vec!["-DCULA_SM90A_ENABLED".into()],
        });
    }
    tasks.push(CompileTask {
        src: wrapper_src.clone(),
        obj: out_dir.join("cula_wrapper_sm90.o"),
        gencodes: vec!["-gencode=arch=compute_90a,code=sm_90a".into()],
        defines: vec!["-DCULA_SM90A_ENABLED".into()],
    });

    if sm100 {
        build_log!("compiling SM100 KDA kernels");
        tasks.push(CompileTask {
            src: cula_dir.join("csrc/kda/sm100/kda_fwd_sm100.cu"),
            obj: out_dir.join("kda_fwd_sm100.o"),
            gencodes: vec!["-gencode=arch=compute_100a,code=sm_100a".into()],
            defines: vec![],
        });
        tasks.push(CompileTask {
            src: wrapper_src.clone(),
            obj: out_dir.join("cula_wrapper_sm100.o"),
            gencodes: vec!["-gencode=arch=compute_100a,code=sm_100a".into()],
            defines: vec!["-DCULA_SM100_ENABLED".into()],
        });
    } else {
        build_log!("SM100 not supported by nvcc, skipping Blackwell kernels");
    }

    // Parallel compile via threads. Each thread calls into the shared
    // compile_cu_to_obj helper.
    let nvcc_arc = std::sync::Arc::new(nvcc.to_path_buf());
    let includes_arc = std::sync::Arc::new(include_dirs);

    let handles: Vec<_> = tasks
        .into_iter()
        .map(|task| {
            let nvcc_p = nvcc_arc.clone();
            let inc = includes_arc.clone();
            std::thread::spawn(move || {
                let mut opts = ObjCompile::new(&task.src, &task.obj);
                for i in inc.iter() {
                    opts = opts.include(i.clone());
                }
                for g in &task.gencodes {
                    opts = opts.gencode(g.clone());
                }
                for d in &task.defines {
                    opts = opts.define(d.clone());
                }
                compile_cu_to_obj(&nvcc_p, &opts);
                task.obj
            })
        })
        .collect();

    handles
        .into_iter()
        .map(|h| h.join().expect("nvcc compilation thread panicked"))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// Phase 2: CuTe DSL kernels (AOT via Python)
// ─────────────────────────────────────────────────────────────────────
//
// Runs `scripts/compile_kernels.py` inside a provisioned venv. The
// script itself owns all the per-kernel knowledge (variant matrix,
// placeholder tensors, `cute.compile` → `export_to_c` calls); this
// function just handles venv + caching + per-arch invocation. That
// last bit still has to stay per-crate today because the cuLA compile
// script needs specific env vars and a cuLA-specific bootstrap
// monkey-patch; step 4 of the refactor will lift that into a shared
// dsl compile driver.

fn compile_dsl_kernels(
    kernels_dir: &Path,
    manifest_dir: &Path,
    cula_dir: &Path,
    out_dir: &Path,
    target_archs: &[String],
) -> bool {
    let script = manifest_dir.join("scripts/compile_kernels.py");

    // Cache hit: if the per-arch manifest was written by a script with
    // the same hash as the one we'd run now, trust the cache.
    if let Some(hash) = file_hash(&script) {
        if kernels_dir.join("manifest.json").exists() && has_obj_files(kernels_dir) {
            let manifest_str = std::fs::read_to_string(kernels_dir.join("manifest.json"))
                .unwrap_or_default();
            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                let stored = manifest
                    .get("script_hash")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if stored == hash {
                    build_log!("[DSL] using cached kernels");
                    return true;
                }
                build_log!("[DSL] compile_kernels.py changed, recompiling...");
                clear_obj_files(kernels_dir);
            }
        }
    }

    // Sticky failure marker: if the previous build tried and failed
    // with the same script, don't re-run nvcc on every `cargo build`.
    let fail_marker = kernels_dir.join(".dsl_compile_failed");
    if let Some(hash) = file_hash(&script) {
        if fail_marker.exists() {
            let stored = std::fs::read_to_string(&fail_marker).unwrap_or_default();
            if stored.trim() == hash {
                build_log!("[DSL] skipping (previously failed, script unchanged)");
                return false;
            }
        }
    }

    build_log!("[DSL] attempting AOT compilation...");

    if !script.exists() {
        build_log!("[DSL] no compile script, skipping DSL kernels");
        return false;
    }

    let python = match ensure_cula_python_env(out_dir, cula_dir) {
        Ok(p) => p,
        Err(e) => {
            build_log!("[DSL] Python env failed: {e}, skipping DSL kernels");
            return false;
        }
    };

    build_log!("[DSL] AOT compiling for {:?}...", target_archs);

    // Force workers=1: each DSL kernel's compile step touches CUDA
    // (for SM detection, tensor setup, etc.), and forking after CUDA
    // init blows up with "Cannot re-initialize CUDA in forked
    // subprocess". Single-worker is slower but actually works.
    let workers = env::var("PRELUDE_CULA_WORKERS").unwrap_or_else(|_| "1".to_string());

    // Monkey-patch assert_blackwell/assert_hopper so AOT
    // cross-compilation works regardless of the host GPU. The target
    // arch comes from CUTE_DSL_ARCH. `runpy.run_path` launches the
    // script as if it were invoked directly so `_compile_worker` can
    // be pickled by a ProcessPool inside the script.
    let bootstrap = r#"
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

    for arch in target_archs {
        build_log!("[DSL] compiling for {arch}...");
        let arch_dir = kernels_dir.join(arch);
        let _ = std::fs::create_dir_all(&arch_dir);

        // CUTE_DSL_ENABLE_TVM_FFI=1 makes `export_to_c` emit
        // `__tvm_ffi_<name>` wrappers around the MLIR/CUTLASS kernels.
        // Without it the .o has only the raw `_cuda_init` /
        // `_cutlass_...` / `_mlir_ciface_...` entry points and there's
        // no stable callable for the dispatch table.
        let output = Command::new(&python)
            .arg("-c")
            .arg(bootstrap)
            .arg(&script)
            .arg("--output-dir")
            .arg(&arch_dir)
            .args(["-j", &workers])
            .env("CUTE_DSL_ARCH", format!("{arch}a"))
            .env("CUTE_DSL_ENABLE_TVM_FFI", "1")
            .output();

        match &output {
            Ok(o) if o.status.success() => {
                build_log!("[DSL] {arch} succeeded");
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                let stdout = String::from_utf8_lossy(&o.stdout);
                build_log!("[DSL] {arch} failed (exit code: {:?})", o.status.code());
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
                build_log!("[DSL] failed to run script: {e}");
            }
        }
    }

    let success = has_obj_files(kernels_dir);
    if !success {
        if let Some(hash) = file_hash(&script) {
            let _ = std::fs::create_dir_all(kernels_dir);
            let _ = std::fs::write(&fail_marker, &hash);
        }
    }
    success
}

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

/// Provision (or reuse) the cuLA venv with torch + flash-linear-attention
/// + cuLA (source install). Returns the path to the venv's python3.
///
/// cuLA installs in three sequential pip steps:
///
///   1. **torch first** — cuLA's setup.py imports torch at build time.
///   2. **flash-linear-attention** — provides `fla`, required by the
///      KDA kernels. Upstream's pyproject.toml doesn't declare it, so
///      we install it ourselves. Failure is tolerated (logged, not
///      fatal) — if it's missing the KDA kernels are skipped but other
///      variants still build.
///   3. **cuLA from source** with `--no-build-isolation` (torch is
///      already installed).
///
/// We also require `cula.kda.kda_decode` importable so a stale venv
/// with an older cuLA gets reinstalled from the current source tree.
fn ensure_cula_python_env(out_dir: &Path, cula_src: &Path) -> Result<PathBuf, String> {
    let venv_dir = out_dir.join("cula-venv");
    let venv = PythonVenv::ensure(&venv_dir)?;

    let check_code = "import cula; import cutlass; from cula.kda import kda_decode";
    if venv.check_import(check_code) {
        return Ok(venv.python_path().to_path_buf());
    }

    if !cula_src.join("pyproject.toml").exists() {
        return Err(format!("cuLA source not found at {}", cula_src.display()));
    }

    build_log!(
        "installing cuLA Python deps into venv from {}...",
        cula_src.display()
    );

    let torch_index = detect_torch_cuda_index();

    // Step 1: torch (must be first for cuLA's setup.py to import it).
    venv.pip_install(
        &["torch"],
        InstallOpts::new().extra_index_url(torch_index.as_deref()),
    )?;

    // Step 2: flash-linear-attention — optional but preferred.
    if let Err(e) = venv.pip_install(&["flash-linear-attention"], InstallOpts::new()) {
        build_log!("flash-linear-attention install failed ({e}); KDA kernels will be skipped");
    }

    // Step 3: cuLA from source, no-build-isolation so it sees the
    // torch we just installed.
    let cula_src_str = cula_src
        .to_str()
        .ok_or_else(|| format!("cula source path not utf-8: {}", cula_src.display()))?;
    venv.pip_install(
        &[cula_src_str],
        InstallOpts::new()
            .extra_index_url(torch_index.as_deref())
            .no_build_isolation(true),
    )?;

    if !venv.check_import(check_code) {
        return Err("cuLA not importable after install".into());
    }

    Ok(venv.python_path().to_path_buf())
}

// ─────────────────────────────────────────────────────────────────────
// Phase 3: Dispatch codegen
// ─────────────────────────────────────────────────────────────────────

fn generate_dispatch_code(kernels_dir: &Path, out_dir: &Path, has_kernels: bool) {
    let dispatch_path = out_dir.join("cula_dsl_dispatch.rs");
    let stub_signature =
        "pub(crate) fn lookup_dsl(_kernel_type: &str, _key: &str, _arch: u32) \
         -> Option<crate::dsl::TVMSafeCallFn>";

    if !has_kernels {
        std::fs::write(&dispatch_path, dispatch::stub_lookup(stub_signature))
            .expect("write dispatch stub");
        return;
    }

    let variants = dispatch::collect_manifests(kernels_dir);
    if variants.is_empty() {
        std::fs::write(&dispatch_path, dispatch::stub_lookup(stub_signature))
            .expect("write dispatch stub");
        return;
    }

    let symbols: Vec<&str> = variants
        .iter()
        .filter_map(|v| v["symbol"].as_str())
        .collect();

    let mut code = dispatch::header_comment();
    code.push_str(&dispatch::tvm_ffi_extern_block(
        symbols.iter().copied(),
        "crate::dsl::TVMFFIAny",
    ));

    // cuLA-specific lookup: key is the full variant name string, arch
    // is the SM major×10+minor the kernel was compiled for.
    writeln!(
        code,
        "pub(crate) fn lookup_dsl(_kernel_type: &str, key: &str, arch: u32) \
         -> Option<crate::dsl::TVMSafeCallFn> {{"
    )
    .unwrap();
    writeln!(code, "    match (key, arch) {{").unwrap();
    for variant in &variants {
        let name = variant["name"].as_str().unwrap();
        let symbol = variant["symbol"].as_str().unwrap();
        let arch = variant["arch"].as_u64().unwrap_or(0);
        writeln!(code, "        (\"{name}\", {arch}) => Some({symbol}),").unwrap();
    }
    writeln!(code, "        _ => None,").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}").unwrap();

    std::fs::write(&dispatch_path, &code).expect("write dispatch");
    dispatch::log_generated("cuLA", variants.len());
}

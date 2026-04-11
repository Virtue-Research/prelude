//! Build script for the `cula` crate — Rust bindings to cuLA linear attention kernels.
//!
//! Phase 1: Compile C++ CUTLASS 3.x kernels (SM90/SM100) via nvcc → static archive.
//! Phase 2: AOT compile CuTe DSL kernels via Python → .o files → static archive.
//! Phase 3: Generate Rust dispatch code from manifest.json.
//!
//! Source layout discovery:
//!   - cuLA sources:    $CULA_ROOT     (default: $CARGO_WORKSPACE/third_party/cuLA)
//!   - CUTLASS headers: $CUTLASS_ROOT  (default: $CARGO_WORKSPACE/third_party/cutlass)
//!   - CUDA toolkit:    $CUDA_PATH     (default: /usr/local/cuda or /opt/cuda)
//!
//! The workspace-root fallback keeps zero-config builds inside prelude; the env
//! vars let this crate build standalone by pointing at checkouts elsewhere.

use std::env;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

// Workspace helpers are inlined below rather than `include!`-ed from a parent
// file, so this crate can be built standalone (outside the prelude workspace).
macro_rules! build_log {
    ($($arg:tt)*) => {{
        let _msg = format!($($arg)*);
        eprintln!("  [{}] {_msg}", env!("CARGO_PKG_NAME"));
        println!("cargo:warning={}", _msg);
    }};
}

/// Watch a git submodule's HEAD so we rebuild when the submodule pointer moves.
/// Walks up from `CARGO_MANIFEST_DIR` looking for a `.git` directory; this means
/// the crate must live inside a git checkout that has `third_party/<name>` as a
/// submodule. When run outside such a workspace the function is a no-op and the
/// build just relies on explicit env vars / Cargo.toml rerun-if triggers.
fn track_submodule(name: &str) {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut dir = manifest.as_path();
    loop {
        if dir.join(".git").is_dir() {
            let head = dir.join(format!(".git/modules/third_party/{name}/HEAD"));
            if head.exists() {
                println!("cargo:rerun-if-changed={}", head.display());
            }
            return;
        }
        match dir.parent() {
            Some(p) => dir = p,
            None => return,
        }
    }
}

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
    //   - `CULA_ROOT`    overrides cuLA source path (marker: csrc/kda/sm90/prefill_kernel.hpp)
    //   - `CUTLASS_ROOT` overrides CUTLASS include path (marker: include/cutlass/cutlass.h)
    // Fall back to the prelude workspace layout
    // (crates/prelude-cuda/cula → ../../.. = workspace root) so the common case
    // is zero-config.
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
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }
    let sm100 = nvcc_supports_sm100(&nvcc);

    // ── Phase 1 (nvcc C++) and Phase 2 (Python DSL) run in parallel ──
    //
    //   Phase 1: nvcc compiles C++ CUTLASS kernels (uses GPU compiler). Always on.
    //   Phase 2: Python compiles CuTe DSL kernels (uses Python + cutlass-dsl).
    //            Gated behind the `dsl` cargo feature (on by default).
    //
    // They share no state and use different tools → safe to parallelize.

    let dsl_enabled = env::var("CARGO_FEATURE_DSL").is_ok();
    let kernels_dir = out_dir.join("dsl_kernels");

    // Build target arch list: SM90 always, plus higher archs if nvcc supports them.
    let mut dsl_archs = vec!["sm_90".to_string()];
    if sm100 { dsl_archs.push("sm_100".to_string()); }
    // Future: if sm120 { dsl_archs.push("sm_120".to_string()); }

    // Spawn Phase 2 in a background thread (it may take minutes for venv setup).
    // When the `dsl` feature is off, the thread just skips the Python build and
    // we emit a stub dispatch table later.
    let dsl_kernels_dir = kernels_dir.clone();
    let dsl_manifest_dir = manifest_dir.clone();
    let dsl_cula_dir = cula_dir.clone();
    let dsl_out_dir = out_dir.clone();
    let dsl_handle = std::thread::spawn(move || {
        if !dsl_enabled {
            build_log!("[DSL] feature disabled, skipping Python AOT compile");
            return false;
        }
        compile_dsl_kernels(&dsl_kernels_dir, &dsl_manifest_dir, &dsl_cula_dir, &dsl_out_dir, &dsl_archs)
    });

    // Phase 1: C++ CUTLASS kernels (runs on main thread concurrently)
    let cpp_objects = compile_cpp_kernels(
        &nvcc, &cula_dir, &cutlass_dir, &cuda_path, &out_dir, &manifest_dir, sm100,
    );

    // Archive C++ objects → libcula_cpp.a
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

    // Wait for Phase 2 to finish
    let has_dsl_kernels = dsl_handle.join().expect("DSL compilation thread panicked");

    if has_dsl_kernels {
        link_dsl_kernel_objects(&kernels_dir, &out_dir);
    }

    // ── Phase 3: Generate dispatch code ──────────────────────────────
    generate_dispatch_code(&kernels_dir, &out_dir, has_dsl_kernels);

    // ── Link CUDA runtime ────────────────────────────────────────────
    link_cuda_runtime(&cuda_path);
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
    let include_args: Vec<String> = vec![
        format!("-I{}", cutlass_dir.join("include").display()),
        format!("-I{}", cutlass_dir.join("tools/util/include").display()),
        format!("-I{}", cula_dir.join("csrc").display()),
        format!("-I{}", cula_dir.join("csrc/kerutils/include").display()),
        format!("-I{}/include", cuda_path.display()),
    ];

    let sm90_sources = vec![
        cula_dir.join("csrc/kda/sm90/kda_fwd_sm90.cu"),
        cula_dir.join("csrc/kda/sm90/kda_fwd_sm90_safe_gate.cu"),
    ];
    let wrapper_src = manifest_dir.join("src/cula_wrapper.cu");

    // Collect all (src, obj, arch_args, defines) tasks, then compile in parallel.
    let mut tasks: Vec<(PathBuf, PathBuf, Vec<&str>, Vec<&str>)> = Vec::new();

    // SM90 kernel sources
    for src in &sm90_sources {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let obj = out_dir.join(format!("{stem}.o"));
        tasks.push((src.clone(), obj, vec!["-gencode=arch=compute_90a,code=sm_90a"], vec!["-DCULA_SM90A_ENABLED"]));
    }

    // SM90 wrapper
    tasks.push((
        wrapper_src.clone(),
        out_dir.join("cula_wrapper_sm90.o"),
        vec!["-gencode=arch=compute_90a,code=sm_90a"],
        vec!["-DCULA_SM90A_ENABLED"],
    ));

    // SM100
    if sm100 {
        build_log!("compiling SM100 KDA kernels");
        tasks.push((
            cula_dir.join("csrc/kda/sm100/kda_fwd_sm100.cu"),
            out_dir.join("kda_fwd_sm100.o"),
            vec!["-gencode=arch=compute_100a,code=sm_100a"],
            vec![],
        ));
        tasks.push((
            wrapper_src.clone(),
            out_dir.join("cula_wrapper_sm100.o"),
            vec!["-gencode=arch=compute_100a,code=sm_100a"],
            vec!["-DCULA_SM100_ENABLED"],
        ));
    } else {
        build_log!("SM100 not supported by nvcc, skipping Blackwell kernels");
    }

    // Compile all in parallel using threads
    let include_args_shared: std::sync::Arc<Vec<String>> = std::sync::Arc::new(include_args);
    let nvcc_shared: std::sync::Arc<PathBuf> = std::sync::Arc::new(nvcc.to_path_buf());

    let handles: Vec<_> = tasks.into_iter().map(|(src, obj, arch_args, defines)| {
        let inc = include_args_shared.clone();
        let nvcc_p = nvcc_shared.clone();
        let arch_owned: Vec<String> = arch_args.into_iter().map(String::from).collect();
        let def_owned: Vec<String> = defines.into_iter().map(String::from).collect();
        std::thread::spawn(move || {
            let arch_refs: Vec<&str> = arch_owned.iter().map(|s| s.as_str()).collect();
            let def_refs: Vec<&str> = def_owned.iter().map(|s| s.as_str()).collect();
            nvcc_compile(&nvcc_p, &src, &obj, &inc, &arch_refs, &def_refs);
            obj
        })
    }).collect();

    let objects: Vec<PathBuf> = handles.into_iter()
        .map(|h| h.join().expect("nvcc compilation thread panicked"))
        .collect();

    objects
}

fn nvcc_compile(
    nvcc: &Path, src: &Path, obj: &Path,
    include_args: &[String], arch_args: &[&str], defines: &[&str],
) {
    let src_name = src.file_name().unwrap().to_str().unwrap();
    let arch_str = arch_args.first().map(|a| *a).unwrap_or("unknown");
    build_log!("[nvcc] {src_name} ({arch_str})");

    let mut cmd = Command::new(nvcc);
    cmd.args([
        "-std=c++20", "-O3",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-Xcompiler", "-fPIC",
    ]);
    for a in arch_args {
        cmd.arg(a);
    }
    for d in defines {
        cmd.arg(d);
    }
    for inc in include_args {
        cmd.arg(inc);
    }
    cmd.args(["-c", src.to_str().unwrap(), "-o", obj.to_str().unwrap()]);
    let status = cmd.status().expect("Failed to run nvcc");
    if !status.success() {
        panic!("nvcc failed for {}", src.display());
    }
    build_log!("[nvcc] {src_name} ✓");
}

// ─────────────────────────────────────────────────────────────────────
// Phase 2: CuTe DSL kernels (AOT via Python, same pattern as FA4)
// ─────────────────────────────────────────────────────────────────────

fn compile_dsl_kernels(
    kernels_dir: &Path,
    manifest_dir: &Path,
    cula_dir: &Path,
    out_dir: &Path,
    target_archs: &[String],
) -> bool {
    let script = manifest_dir.join("scripts/compile_kernels.py");

    // Check if pre-compiled kernels exist and are up-to-date
    if let Some(hash) = file_hash(&script) {
        if kernels_dir.join("manifest.json").exists() && has_obj_files(kernels_dir) {
            let manifest_str = std::fs::read_to_string(kernels_dir.join("manifest.json"))
                .unwrap_or_default();
            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                let stored = manifest.get("script_hash")
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

    // Check if a previous attempt already failed (don't retry every build)
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

    let python = match ensure_python_env(out_dir) {
        Ok(p) => p,
        Err(e) => {
            build_log!("[DSL] Python env failed: {e}, skipping DSL kernels");
            return false;
        }
    };

    build_log!("[DSL] AOT compiling for {:?}...", target_archs);

    // Force workers=1: each DSL kernel's compile step touches CUDA (for SM
    // detection, tensor setup, etc.), and forking after CUDA init blows up
    // with "Cannot re-initialize CUDA in forked subprocess". Pickling the
    // compile_worker across a ProcessPool spawn also breaks when the script
    // is executed via runpy. Single-worker is slower but actually works.
    let workers = env::var("PRELUDE_CULA_WORKERS").unwrap_or_else(|_| "1".to_string());

    // Monkey-patch assert_blackwell/assert_hopper so AOT cross-compilation works
    // regardless of the host GPU. The target arch comes from CUTE_DSL_ARCH env var.
    // The first positional arg is the absolute path to the script — we pop it so
    // argparse inside the script only sees --output-dir / -j flags, then exec
    // the script with __file__ pointed at the real path.
    // `runpy.run_path(..., run_name='__main__')` loads the script as if it
    // were executed as `python compile_kernels.py ...`, so `ProcessPoolExecutor`
    // can pickle top-level functions like `_compile_worker` — exec(compile(...))
    // with a hand-rolled globals dict breaks that lookup.
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

    // Compile for each target arch
    for arch in target_archs {
        build_log!("[DSL] compiling for {arch}...");
        let arch_dir = kernels_dir.join(arch);
        let _ = std::fs::create_dir_all(&arch_dir);

        // We used to set PYTHONPATH=third_party/cuLA to pick up local changes
        // without reinstalling the venv, but that causes `import cula` to
        // resolve to the source tree, which is missing the compiled
        // `cula.cudac` C extension (which only lives in site-packages). Drop
        // the override and rely on `ensure_python_env` to keep the venv in
        // sync with the source tree.
        //
        // CUTE_DSL_ENABLE_TVM_FFI=1 makes the AOT `export_to_c` pipeline emit
        // `__tvm_ffi_<name>` wrapper symbols around the MLIR/CUTLASS kernels.
        // Without it, the .o contains only the raw `_cuda_init` /
        // `_cutlass_...` / `_mlir_ciface_...` entry points and there's no
        // stable callable for the dispatch table to reference.
        let output = Command::new(&python)
            .arg("-c").arg(bootstrap)
            .arg(&script)
            .arg("--output-dir").arg(&arch_dir)
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
                for line in stderr.lines().rev().take(10).collect::<Vec<_>>().into_iter().rev() {
                    build_log!("[DSL]   {line}");
                }
                if !stdout.is_empty() {
                    for line in stdout.lines().rev().take(5).collect::<Vec<_>>().into_iter().rev() {
                        build_log!("[DSL]   [stdout] {line}");
                    }
                }
            }
            Err(e) => {
                build_log!("[DSL] failed to run script: {e}");
            }
        }
    }

    // Check if any arch produced kernel objects
    let success = has_obj_files(kernels_dir);
    if !success {
        // Write fail marker so we don't retry every build
        if let Some(hash) = file_hash(&script) {
            let _ = std::fs::create_dir_all(kernels_dir);
            let _ = std::fs::write(&fail_marker, &hash);
        }
    }
    success
}

fn collect_obj_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|x| x == "o") { out.push(p); }
        else if p.is_dir() { collect_obj_files(&p, out); }
    }
}

fn link_dsl_kernel_objects(kernels_dir: &Path, out_dir: &Path) {
    let mut obj_files = Vec::new();
    collect_obj_files(kernels_dir, &mut obj_files);

    if obj_files.is_empty() {
        return;
    }

    build_log!("[DSL] archiving {} kernel .o files", obj_files.len());

    let archive_path = out_dir.join("libcula_dsl_kernels.a");
    let mut cmd = Command::new("ar");
    cmd.arg("rcs").arg(&archive_path);
    for obj in &obj_files {
        cmd.arg(obj);
    }
    let status = cmd.status().expect("Failed to run ar");
    if !status.success() {
        panic!("ar failed to create libcula_dsl_kernels.a");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=cula_dsl_kernels");
}

// ─────────────────────────────────────────────────────────────────────
// Phase 3: Generate Rust dispatch code
// ─────────────────────────────────────────────────────────────────────

fn generate_dispatch_code(kernels_dir: &Path, out_dir: &Path, has_kernels: bool) {
    let dispatch_path = out_dir.join("cula_dsl_dispatch.rs");

    let stub = "pub(crate) fn lookup_dsl(_kernel_type: &str, _key: &str, _arch: u32) \
                -> Option<crate::dsl::TVMSafeCallFn> { None }\n";

    if !has_kernels {
        std::fs::write(&dispatch_path, stub).unwrap();
        return;
    }

    // Each target arch writes its own manifest at `dsl_kernels/<arch>/manifest.json`.
    // Walk the per-arch manifests and merge variants into one dispatch table
    // keyed on (name, arch).
    let mut variants: Vec<serde_json::Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(kernels_dir) {
        for entry in entries.flatten() {
            let arch_dir = entry.path();
            if !arch_dir.is_dir() { continue; }
            let manifest_path = arch_dir.join("manifest.json");
            let Ok(manifest_str) = std::fs::read_to_string(&manifest_path) else { continue };
            let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) else { continue };
            let Some(arr) = manifest["variants"].as_array() else { continue };
            variants.extend(arr.iter().cloned());
        }
    }
    // Fall back to the old single-manifest layout for compatibility.
    if variants.is_empty() {
        let manifest_path = kernels_dir.join("manifest.json");
        if let Ok(manifest_str) = std::fs::read_to_string(&manifest_path) {
            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&manifest_str) {
                if let Some(arr) = manifest["variants"].as_array() {
                    variants.extend(arr.iter().cloned());
                }
            }
        }
    }

    if variants.is_empty() {
        std::fs::write(&dispatch_path, stub).unwrap();
        return;
    }

    let mut code = String::new();
    writeln!(code, "// AUTO-GENERATED by build.rs — do not edit").unwrap();
    writeln!(code).unwrap();

    // Extern declarations — de-dup by symbol in case two manifests list the same kernel.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    writeln!(code, "unsafe extern \"C\" {{").unwrap();
    for variant in &variants {
        let symbol = variant["symbol"].as_str().unwrap();
        if !seen.insert(symbol.to_string()) { continue; }
        writeln!(
            code,
            "    fn {symbol}(handle: *mut std::ffi::c_void, args: *const crate::dsl::TVMFFIAny, \
             num_args: i32, ret: *mut crate::dsl::TVMFFIAny) -> i32;"
        ).unwrap();
    }
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Lookup by (name, arch). Arch is the SM major×10+minor the kernel was
    // compiled for. Runtime caller passes the detected GPU arch and we match
    // exact-equal.
    writeln!(
        code,
        "pub(crate) fn lookup_dsl(_kernel_type: &str, key: &str, arch: u32) \
         -> Option<crate::dsl::TVMSafeCallFn> {{"
    ).unwrap();
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

    std::fs::write(&dispatch_path, &code).unwrap();
    println!(
        "cargo:warning=cuLA DSL: generated dispatch with {} variants",
        variants.len()
    );
}

// ─────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────

fn link_cuda_runtime(cuda_path: &Path) {
    let cuda_lib = cuda_path.join("lib64");
    if cuda_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    let cuda_targets_lib = cuda_path.join("targets/x86_64-linux/lib");
    if cuda_targets_lib.exists() {
        println!("cargo:rustc-link-search=native={}", cuda_targets_lib.display());
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn nvcc_supports_sm100(nvcc: &Path) -> bool {
    Command::new(nvcc)
        .arg("--list-gpu-arch")
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("compute_100"))
        .unwrap_or(false)
}

/// Resolve a C++ dependency root from an env var override (first choice) or a
/// workspace fallback (second choice). `marker` is a relative file that must
/// exist inside the resolved directory; we use it as a sanity check and surface
/// a helpful error instead of silently compiling with broken include paths.
fn locate_source(env_var: &str, name: &str, marker: &str, fallback: &Path) -> PathBuf {
    let chosen = match env::var(env_var) {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => fallback.to_path_buf(),
    };
    if !chosen.join(marker).exists() {
        let hint = if env::var(env_var).is_ok() {
            format!(
                "{env_var}={} points at a directory that is missing `{marker}`. \
                 Check that it really is a {name} checkout.",
                chosen.display()
            )
        } else {
            format!(
                "Expected {name} at {} (missing `{marker}`). Either run \
                 `git submodule update --init third_party/{name}` inside the \
                 prelude workspace, or set {env_var}=/path/to/{name} to build \
                 this crate standalone.",
                chosen.display()
            )
        };
        panic!("{hint}");
    }
    chosen
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

fn file_hash(path: &Path) -> Option<String> {
    use sha2::Digest;
    let content = std::fs::read(path).ok()?;
    let hash = sha2::Sha256::digest(&content);
    Some(hex::encode(hash)[..16].to_string())
}

fn has_obj_files(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else { return false };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|x| x == "o") { return true; }
        if p.is_dir() && has_obj_files(&p) { return true; }
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

fn ensure_python_env(out_dir: &Path) -> Result<String, String> {
    let venv_dir = out_dir.join("cula-venv");
    let venv_python = venv_dir.join("bin/python3");

    // If venv already has cuLA installed, reuse it.
    if check_cula_importable(venv_python.to_str().unwrap_or("")) {
        return Ok(venv_python.to_string_lossy().into_owned());
    }

    // Find cuLA source (third_party/cuLA with pyproject.toml)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap().parent().unwrap();
    let cula_src = workspace_root.join("third_party/cuLA");
    if !cula_src.join("pyproject.toml").exists() {
        return Err(format!("cuLA source not found at {}", cula_src.display()));
    }

    build_log!("creating Python venv and installing cuLA from {}...", cula_src.display());

    // Create venv (prefer uv for speed, fallback to python3 -m venv)
    let has_uv = Command::new("uv").arg("--version").output()
        .map(|o| o.status.success()).unwrap_or(false);

    if has_uv {
        let s = Command::new("uv")
            .args(["venv", venv_dir.to_str().unwrap(), "--python", "3.12"])
            .status().map_err(|e| format!("uv venv: {e}"))?;
        if !s.success() {
            return Err("uv venv creation failed".into());
        }
    } else {
        let s = Command::new("python3")
            .args(["-m", "venv", venv_dir.to_str().unwrap()])
            .status().map_err(|e| format!("python3 -m venv: {e}"))?;
        if !s.success() {
            return Err("venv creation failed".into());
        }
    }

    // Step 1: Install torch first (cuLA's build imports torch at setup time).
    let venv_python_str = venv_python.to_string_lossy().to_string();
    let torch_index = detect_torch_cuda_index();

    build_log!("installing torch...");
    let torch_ok = if has_uv {
        let mut cmd = Command::new("uv");
        cmd.args(["pip", "install", "--python", &venv_python_str, "torch"]);
        if let Some(ref idx) = torch_index {
            cmd.args(["--extra-index-url", idx]);
        }
        cmd.status().map(|s| s.success()).unwrap_or(false)
    } else {
        let pip = venv_dir.join("bin/pip");
        let mut cmd = Command::new(&pip);
        cmd.args(["install", "torch"]);
        if let Some(ref idx) = torch_index {
            cmd.args(["--extra-index-url", idx]);
        }
        cmd.status().map(|s| s.success()).unwrap_or(false)
    };
    if !torch_ok {
        return Err("Failed to install torch".into());
    }

    // Step 2: Install flash-linear-attention (provides `fla` module, needed by KDA kernels).
    // cuLA's pyproject.toml doesn't declare this dep (upstream oversight).
    build_log!("installing flash-linear-attention...");
    let fla_ok = if has_uv {
        Command::new("uv").args(["pip", "install", "--python", &venv_python_str, "flash-linear-attention"])
            .status().map(|s| s.success()).unwrap_or(false)
    } else {
        let pip = venv_dir.join("bin/pip");
        Command::new(&pip).args(["install", "flash-linear-attention"])
            .status().map(|s| s.success()).unwrap_or(false)
    };
    if !fla_ok {
        build_log!("flash-linear-attention install failed (KDA kernels will be skipped)");
    }

    // Step 3: Install cuLA from source (--no-build-isolation since torch is already installed).
    build_log!("installing cuLA from source...");
    let output = if has_uv {
        let mut cmd = Command::new("uv");
        cmd.args(["pip", "install", "--python", &venv_python_str, "--no-build-isolation"]);
        if let Some(ref idx) = torch_index {
            cmd.args(["--extra-index-url", idx]);
        }
        cmd.arg(cula_src.to_str().unwrap());
        cmd.output().map_err(|e| format!("uv pip install cuLA: {e}"))?
    } else {
        let pip = venv_dir.join("bin/pip");
        let mut cmd = Command::new(&pip);
        cmd.args(["install", "--no-build-isolation"]);
        if let Some(ref idx) = torch_index {
            cmd.args(["--extra-index-url", idx]);
        }
        cmd.arg(cula_src.to_str().unwrap());
        cmd.output().map_err(|e| format!("pip install cuLA: {e}"))?
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        for line in stderr.lines().rev().take(10).collect::<Vec<_>>().into_iter().rev() {
            build_log!("[venv] {line}");
        }
        return Err(format!("pip install cuLA failed (exit {})", output.status));
    }

    if !check_cula_importable(&venv_python_str) {
        return Err("cuLA not importable after install".into());
    }

    Ok(venv_python_str)
}

fn check_cula_importable(python: &str) -> bool {
    if python.is_empty() { return false; }
    // Also require cula.kda.kda_decode (added in the KDA decode PR) — if it's
    // missing, the installed venv is stale and must be reinstalled from the
    // updated source tree.
    Command::new(python)
        .args(["-c", "import cula; import cutlass; from cula.kda import kda_decode"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn detect_torch_cuda_index() -> Option<String> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let release = text.split("release ").nth(1)?;
    let version = release.split(',').next()?.trim();
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() >= 2 {
        let cu_tag = format!("{}{}", parts[0], parts[1]);
        Some(format!("https://download.pytorch.org/whl/cu{cu_tag}"))
    } else {
        None
    }
}

//! Build script for prelude-cula: cuLA linear attention kernels.
//!
//! Phase 1: Compile C++ CUTLASS 3.x kernels (SM90/SM100) via nvcc → static archive.
//! Phase 2: AOT compile CuTe DSL kernels via Python → .o files → static archive.
//! Phase 3: Generate Rust dispatch code from manifest.json.

use std::env;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

include!("../../../build_log.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cula_wrapper.cu");
    println!("cargo:rerun-if-changed=scripts/compile_kernels.py");
    track_submodule("cuLA");
    track_submodule("cutlass");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.join("../../..");

    // Ensure submodules
    let cula_dir = workspace_root.join("third_party/cuLA");
    if !cula_dir.join("csrc/kda/sm90/prefill_kernel.hpp").exists() {
        panic!(
            "third_party/cuLA submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/cuLA"
        );
    }
    let cutlass_dir = workspace_root.join("third_party/cutlass");
    if !cutlass_dir.join("include/cutlass/cutlass.h").exists() {
        panic!(
            "third_party/cutlass submodule not found or incomplete.\n\
             Run: git submodule update --init third_party/cutlass"
        );
    }

    let cuda_path = find_cuda();
    let nvcc = cuda_path.join("bin/nvcc");
    if !nvcc.exists() {
        panic!("nvcc not found at {}", nvcc.display());
    }
    let sm100 = nvcc_supports_sm100(&nvcc);

    // ── Phase 1 (nvcc C++) and Phase 2 (Python DSL) run in parallel ──
    //
    //   Phase 1: nvcc compiles C++ CUTLASS kernels (uses GPU compiler)
    //   Phase 2: Python compiles CuTe DSL kernels (uses Python + cutlass-dsl)
    //
    // They share no state and use different tools → safe to parallelize.

    let kernels_dir = out_dir.join("dsl_kernels");

    // Build target arch list: SM90 always, plus higher archs if nvcc supports them.
    let mut dsl_archs = vec!["sm_90".to_string()];
    if sm100 { dsl_archs.push("sm_100".to_string()); }
    // Future: if sm120 { dsl_archs.push("sm_120".to_string()); }

    // Spawn Phase 2 in a background thread (it may take minutes for venv setup)
    let dsl_kernels_dir = kernels_dir.clone();
    let dsl_manifest_dir = manifest_dir.clone();
    let dsl_cula_dir = cula_dir.clone();
    let dsl_out_dir = out_dir.clone();
    let dsl_handle = std::thread::spawn(move || {
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

    let workers = env::var("PRELUDE_CULA_WORKERS").unwrap_or_else(|_| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        n.min(8).to_string()
    });

    // Monkey-patch assert_blackwell/assert_hopper so AOT cross-compilation works
    // regardless of the host GPU. The target arch comes from CUTE_DSL_ARCH env var.
    let bootstrap = r#"
import sys, os
arch = int(''.join(c for c in os.environ.get('CUTE_DSL_ARCH','sm_90a').split('_')[1] if c.isdigit()))
major, minor = arch // 10, arch % 10
import cula.utils
cula.utils.get_device_sm_version = lambda device=None: (major, minor)
cula.utils.assert_blackwell = lambda device=None: None
cula.utils.assert_hopper = lambda device=None: None
sys.argv = ['compile_kernels.py'] + sys.argv[1:]
exec(open(sys.argv[0]).read())
"#;

    // Compile for each target arch
    for arch in target_archs {
        build_log!("[DSL] compiling for {arch}...");
        let arch_dir = kernels_dir.join(arch);
        let _ = std::fs::create_dir_all(&arch_dir);

        let output = Command::new(&python)
            .arg("-c").arg(bootstrap)
            .arg(&script)
            .arg("--output-dir").arg(&arch_dir)
            .args(["-j", &workers])
            .env("PYTHONPATH", cula_dir)
            .env("CUTE_DSL_ARCH", format!("{arch}a"))
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

    if !has_kernels {
        std::fs::write(
            &dispatch_path,
            "pub(crate) fn lookup_dsl(_kernel_type: &str, _key: &str, _arch: u32) \
             -> Option<crate::dsl::TVMSafeCallFn> { None }\n",
        ).unwrap();
        return;
    }

    let manifest_path = kernels_dir.join("manifest.json");
    let manifest_str = match std::fs::read_to_string(&manifest_path) {
        Ok(s) => s,
        Err(_) => {
            std::fs::write(
                &dispatch_path,
                "pub(crate) fn lookup_dsl(_kernel_type: &str, _key: &str, _arch: u32) \
                 -> Option<crate::dsl::TVMSafeCallFn> { None }\n",
            ).unwrap();
            return;
        }
    };

    let manifest: serde_json::Value = serde_json::from_str(&manifest_str)
        .expect("Failed to parse cuLA manifest.json");
    let variants = manifest["variants"].as_array()
        .expect("manifest.json missing 'variants' array");

    let mut code = String::new();
    writeln!(code, "// AUTO-GENERATED by build.rs — do not edit").unwrap();
    writeln!(code).unwrap();

    // Extern declarations
    writeln!(code, "unsafe extern \"C\" {{").unwrap();
    for variant in variants {
        let symbol = variant["symbol"].as_str().unwrap();
        writeln!(
            code,
            "    fn {symbol}(handle: *mut std::ffi::c_void, args: *const crate::dsl::TVMFFIAny, \
             num_args: i32, ret: *mut crate::dsl::TVMFFIAny) -> i32;"
        ).unwrap();
    }
    writeln!(code, "}}").unwrap();
    writeln!(code).unwrap();

    // Lookup by name
    writeln!(
        code,
        "pub(crate) fn lookup_dsl(_kernel_type: &str, key: &str, _arch: u32) \
         -> Option<crate::dsl::TVMSafeCallFn> {{"
    ).unwrap();
    writeln!(code, "    match key {{").unwrap();
    for variant in variants {
        let name = variant["name"].as_str().unwrap();
        let symbol = variant["symbol"].as_str().unwrap();
        writeln!(code, "        \"{name}\" => Some({symbol}),").unwrap();
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
    Command::new(python)
        .args(["-c", "import cula; import cutlass"])
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

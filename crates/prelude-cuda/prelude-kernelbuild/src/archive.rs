//! Static-archive creation + whole-archive linking for AOT-compiled
//! kernel `.o` files.
//!
//! Every DSL/CUDA kernel crate that produces `.o` files needs the same
//! final step: `ar rcs libfoo.a *.o`, then a
//! `cargo:rustc-link-lib=static:+whole-archive=foo` directive so the
//! linker doesn't drop "unreferenced" TVM-FFI symbols (the Rust side
//! calls them via function pointers in a generated dispatch table, so
//! static reachability analysis can't see the references).
//!
//! This module centralises that two-step dance.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::build_log;

/// Recursively walk `dir` and return every `.o` file found at any depth.
/// Used by DSL crates that write per-arch subdirectories of object
/// files (e.g. `kernels_dir/sm_90/*.o` + `kernels_dir/sm_100/*.o`).
pub fn collect_obj_files(dir: &Path) -> Vec<PathBuf> {
    let mut objects = Vec::new();
    collect_into(dir, &mut objects);
    objects
}

fn collect_into(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_into(&p, out);
        } else if p.extension().is_some_and(|x| x == "o") {
            out.push(p);
        }
    }
}

/// Create a static archive (`lib<name>.a`) containing the given object
/// files and emit the `cargo:rustc-link-*` directives that whole-archive
/// link it into the consumer crate. No-op (returns `false`) when
/// `objects` is empty.
///
/// `lib_name` is the `<name>` in `lib<name>.a` — must match the
/// `cargo:rustc-link-lib=static=<name>` directive rustc generates
/// downstream. The archive lands at `out_dir/lib<name>.a`.
///
/// Whole-archive linking is non-negotiable for TVM-FFI kernels because
/// Rust calls them via generated extern declarations backed by fn
/// pointers inside a dispatch table — the linker's dead-code elimination
/// can't see those references and would otherwise drop every unused
/// kernel, turning the table into a list of dangling pointers.
pub fn archive_and_whole_link(
    objects: &[PathBuf],
    out_dir: &Path,
    lib_name: &str,
) -> Result<bool, String> {
    if objects.is_empty() {
        return Ok(false);
    }

    build_log!("archiving {} kernel .o files → lib{lib_name}.a", objects.len());

    let archive_path = out_dir.join(format!("lib{lib_name}.a"));
    let mut cmd = Command::new("ar");
    cmd.arg("rcs").arg(&archive_path);
    for obj in objects {
        cmd.arg(obj);
    }
    let status = cmd
        .status()
        .map_err(|e| format!("ar spawn failed: {e}"))?;
    if !status.success() {
        return Err(format!("ar failed to create lib{lib_name}.a"));
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive={lib_name}");

    Ok(true)
}

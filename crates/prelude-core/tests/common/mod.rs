#![allow(dead_code)]

use prelude_core::tensor::{Device, DType, Result, Tensor};

// -- Test device selection --

/// Devices to test. CPU always; others added via features.
pub fn test_devices() -> Vec<Device> {
    let mut devs = vec![Device::Cpu];
    #[cfg(feature = "test-cuda")]
    devs.push(Device::Cuda(0));
    // planned:
    // #[cfg(feature = "test-amd")]   devs.push(Device::Amd(0));
    // #[cfg(feature = "test-metal")] devs.push(Device::Metal(0));
    // #[cfg(feature = "test-tpu")]   devs.push(Device::Tpu(0));
    devs
}

// -- PyTorch subprocess --

pub fn pytorch_eval(script: &str) -> Option<String> {
    use std::io::Write;
    // Use stdin instead of -c to avoid OS argument length limits (128KB on Linux).
    let mut child = std::process::Command::new("python3")
        .args(["-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take().unwrap().write_all(script.as_bytes()).ok()?;
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No module named") {
            return None;
        }
        panic!("PyTorch script failed:\n{stderr}\nScript:\n{script}");
    }
    Some(
        String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .to_string(),
    )
}

pub fn parse_f32_list(json: &str) -> Vec<f32> {
    serde_json::from_str::<Vec<f64>>(json)
        .unwrap()
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

pub fn parse_f32_2d(json: &str) -> Vec<Vec<f32>> {
    serde_json::from_str::<Vec<Vec<f64>>>(json)
        .unwrap()
        .into_iter()
        .map(|row| row.into_iter().map(|v| v as f32).collect())
        .collect()
}

pub fn parse_u8_list(json: &str) -> Vec<u8> {
    serde_json::from_str::<Vec<u8>>(json).unwrap()
}

// -- Comparison helpers --

pub fn assert_close(ours: &[f32], reference: &[f32], atol: f32, context: &str) {
    assert_eq!(ours.len(), reference.len(), "{context}: length mismatch");
    let mut max_diff = 0f32;
    let mut max_idx = 0;
    for (i, (a, b)) in ours.iter().zip(reference).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    assert!(
        max_diff < atol,
        "{context}: max_diff={max_diff:.2e} at [{max_idx}] (ours={}, ref={}), atol={atol:.0e}",
        ours[max_idx],
        reference[max_idx]
    );
}

pub fn assert_close_2d(ours: &[Vec<f32>], reference: &[Vec<f32>], atol: f32, context: &str) {
    for (i, (o, r)) in ours.iter().zip(reference).enumerate() {
        assert_close(o, r, atol, &format!("{context} row {i}"));
    }
}

// -- Dtype test configs --

pub struct DTypeConfig {
    pub dtype: DType,
    pub py_dtype: &'static str,
    pub atol_default: f32,
    pub atol_matmul: f32,
    pub atol_reduction: f32,
    pub atol_chained: f32,
}

pub const F32_CONFIG: DTypeConfig = DTypeConfig {
    dtype: DType::F32,
    py_dtype: "torch.float32",
    atol_default: 1e-5,
    atol_matmul: 5e-2,      // TF32 on SM90 has ~10-bit mantissa → relative error ~1e-3
    atol_reduction: 1e-5,
    atol_chained: 5e1,  // TF32 matmul + chained ops → large outputs (~100K) have absolute error ~10-50
};

pub const BF16_CONFIG: DTypeConfig = DTypeConfig {
    dtype: DType::BF16,
    py_dtype: "torch.bfloat16",
    atol_default: 1e-2,
    atol_matmul: 1e-1,
    atol_reduction: 1e-2,
    atol_chained: 5e-1,
};

pub const F16_CONFIG: DTypeConfig = DTypeConfig {
    dtype: DType::F16,
    py_dtype: "torch.float16",
    atol_default: 1e-2,
    atol_matmul: 2e0,     // F16 has 10-bit mantissa; large outputs (~1000) have ULP ~1
    atol_reduction: 1e-3,
    atol_chained: 1e1,    // chained ops amplify errors
};

pub const ALL_DTYPES: &[&DTypeConfig] = &[&F32_CONFIG, &BF16_CONFIG, &F16_CONFIG];

// -- Data generation --

pub fn pseudo_random(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.017 + seed).sin() * 2.0))
        .collect()
}

// -- Rounding helpers --

pub fn round(v: f32, decimals: i32) -> f32 {
    let p = 10f32.powi(decimals);
    (v * p).round() / p
}

pub fn to_vec1_round(t: &Tensor, d: i32) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| round(*v, d))
        .collect())
}

pub fn to_vec2_round(t: &Tensor, d: i32) -> Result<Vec<Vec<f32>>> {
    Ok(t.to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .iter()
        .map(|row| row.iter().map(|v| round(*v, d)).collect())
        .collect())
}

// -- File-based PyTorch reference ─────────────────────────────────────
//
// Input: named f32 arrays → binary temp file.
// Python: reads inputs, runs script, writes output → binary temp file.
// Output: Vec<f32> (single flat result) or Vec<Vec<f32>> (multiple named results).
//
// Usage:
//   let reference = require_pytorch_ref!(
//       inputs: { "x" => &x_data, "w" => &w_data },
//       script: r#"
//           x = read_input("x", {py}).reshape(4, 256)
//           w = read_input("w", {py}).reshape(128, 256)
//           y = (x @ w.T).float()
//           write_output(y)
//       "#,
//       py = cfg.py_dtype
//   );

/// Run a PyTorch script with binary file I/O for data exchange.
///
/// `inputs`: named f32 arrays written to a temp file.
/// `script`: Python code that can call `read_input(name, dtype)` and `write_output(tensor)`.
/// Returns `None` if python3/torch not available, `Some(Vec<f32>)` otherwise.
pub fn pytorch_ref(inputs: &[(&str, &[f32])], script: &str) -> Option<Vec<f32>> {
    use std::io::Write;

    let id = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir();
    let input_path = dir.join(format!("prelude_ref_{id}_{ts}_in.bin"));
    let output_path = dir.join(format!("prelude_ref_{id}_{ts}_out.bin"));
    let meta_path = dir.join(format!("prelude_ref_{id}_{ts}_meta.txt"));

    // Write input arrays as concatenated raw f32 bytes
    {
        let mut f = std::fs::File::create(&input_path).unwrap();
        let mut meta = String::new();
        for (name, data) in inputs {
            let offset = f.stream_position().ok().unwrap_or(0) as usize;
            f.write_all(bytemuck::cast_slice::<f32, u8>(data)).unwrap();
            meta.push_str(&format!("{name} {offset} {}\n", data.len()));
        }
        std::fs::write(&meta_path, meta).unwrap();
    }

    // Build Python preamble: defines read_input() and write_output()
    let preamble = format!(
        r#"
import torch, struct, os
_input_path = "{input_path}"
_output_path = "{output_path}"
_meta = {{}}
with open("{meta_path}") as f:
    for line in f:
        parts = line.strip().split()
        _meta[parts[0]] = (int(parts[1]), int(parts[2]))

_raw = open(_input_path, "rb").read()

def read_input(name, dtype=torch.float32):
    offset, count = _meta[name]
    floats = struct.unpack(f'{{count}}f', _raw[offset:offset+count*4])
    return torch.tensor(floats).to(dtype)

def write_output(tensor):
    t = tensor.float().flatten()
    data = struct.pack(f'{{t.numel()}}f', *t.tolist())
    open(_output_path, "wb").write(data)

def write_outputs(**kwargs):
    import json
    result = {{}}
    for k, v in kwargs.items():
        t = v.float().flatten()
        result[k] = t.tolist()
    open(_output_path, "w").write(json.dumps(result))
"#,
        input_path = input_path.display(),
        output_path = output_path.display(),
        meta_path = meta_path.display(),
    );

    let full_script = format!("{preamble}\n{script}");

    // Run via stdin
    let mut child = std::process::Command::new("python3")
        .args(["-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take().unwrap().write_all(full_script.as_bytes()).ok()?;
    let output = child.wait_with_output().ok()?;

    // Cleanup temp files
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&meta_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = std::fs::remove_file(&output_path);
        if stderr.contains("No module named") {
            return None;
        }
        panic!("PyTorch script failed:\n{stderr}\nScript:\n{full_script}");
    }

    let out_bytes = std::fs::read(&output_path).unwrap();
    let _ = std::fs::remove_file(&output_path);
    let result: Vec<f32> = bytemuck::cast_slice(&out_bytes).to_vec();
    Some(result)
}

/// Like `pytorch_ref` but returns named outputs as a map.
pub fn pytorch_ref_multi(inputs: &[(&str, &[f32])], script: &str) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    use std::io::Write;

    let id = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir();
    let input_path = dir.join(format!("prelude_ref_{id}_{ts}_in.bin"));
    let output_path = dir.join(format!("prelude_ref_{id}_{ts}_out.json"));
    let meta_path = dir.join(format!("prelude_ref_{id}_{ts}_meta.txt"));

    {
        let mut f = std::fs::File::create(&input_path).unwrap();
        let mut meta = String::new();
        for (name, data) in inputs {
            let offset = f.stream_position().ok().unwrap_or(0) as usize;
            f.write_all(bytemuck::cast_slice::<f32, u8>(data)).unwrap();
            meta.push_str(&format!("{name} {offset} {}\n", data.len()));
        }
        std::fs::write(&meta_path, meta).unwrap();
    }

    let preamble = format!(
        r#"
import torch, struct, json
_input_path = "{input_path}"
_output_path = "{output_path}"
_meta = {{}}
with open("{meta_path}") as f:
    for line in f:
        parts = line.strip().split()
        _meta[parts[0]] = (int(parts[1]), int(parts[2]))

_raw = open(_input_path, "rb").read()

def read_input(name, dtype=torch.float32):
    offset, count = _meta[name]
    floats = struct.unpack(f'{{count}}f', _raw[offset:offset+count*4])
    return torch.tensor(floats).to(dtype)

def write_outputs(**kwargs):
    result = {{}}
    for k, v in kwargs.items():
        t = v.float().flatten()
        result[k] = t.tolist()
    open(_output_path, "w").write(json.dumps(result))
"#,
        input_path = input_path.display(),
        output_path = output_path.display(),
        meta_path = meta_path.display(),
    );

    let full_script = format!("{preamble}\n{script}");

    let mut child = std::process::Command::new("python3")
        .args(["-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take().unwrap().write_all(full_script.as_bytes()).ok()?;
    let output = child.wait_with_output().ok()?;

    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&meta_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = std::fs::remove_file(&output_path);
        if stderr.contains("No module named") {
            return None;
        }
        panic!("PyTorch script failed:\n{stderr}\nScript:\n{full_script}");
    }

    let json_str = std::fs::read_to_string(&output_path).unwrap();
    let _ = std::fs::remove_file(&output_path);
    let map: std::collections::HashMap<String, Vec<f64>> = serde_json::from_str(&json_str).unwrap();
    Some(map.into_iter().map(|(k, v)| (k, v.into_iter().map(|x| x as f32).collect())).collect())
}

use std::io::Seek;

/// Reshape a flat Vec<f32> into Vec<Vec<f32>> with the given number of columns.
pub fn unflatten(flat: &[f32], cols: usize) -> Vec<Vec<f32>> {
    flat.chunks(cols).map(|c| c.to_vec()).collect()
}

// -- Macros --

#[macro_export]
macro_rules! require_pytorch {
    ($script:expr) => {
        match common::pytorch_eval($script) {
            Some(v) => v,
            None => {
                eprintln!("SKIPPED: python3 + torch not available");
                return Ok(());
            }
        }
    };
}

#[macro_export]
macro_rules! require_pytorch_ref {
    ($inputs:expr, $script:expr) => {
        match common::pytorch_ref($inputs, $script) {
            Some(v) => v,
            None => {
                eprintln!("SKIPPED: python3 + torch not available");
                return Ok(());
            }
        }
    };
}

#[macro_export]
macro_rules! require_pytorch_ref_multi {
    ($inputs:expr, $script:expr) => {
        match common::pytorch_ref_multi($inputs, $script) {
            Some(v) => v,
            None => {
                eprintln!("SKIPPED: python3 + torch not available");
                return Ok(());
            }
        }
    };
}

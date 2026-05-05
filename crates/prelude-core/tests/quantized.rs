//! Tests for GGUF parsing and quantized tensor dequantization.

use prelude_core::tensor::quantized::gguf_file;
use prelude_core::tensor::quantized::{GgmlDType, QTensor};
use prelude_core::tensor::{Device, Result};

// ── GgmlDType ────────────────────────────────────────────────────

#[test]
fn ggml_dtype_roundtrip() {
    let dtypes = [
        (GgmlDType::F32, 0),
        (GgmlDType::F16, 1),
        (GgmlDType::Q4_0, 2),
        (GgmlDType::Q4_1, 3),
        (GgmlDType::Q5_0, 6),
        (GgmlDType::Q5_1, 7),
        (GgmlDType::Q8_0, 8),
        (GgmlDType::Q8_1, 9),
        (GgmlDType::Q2K, 10),
        (GgmlDType::Q3K, 11),
        (GgmlDType::Q4K, 12),
        (GgmlDType::Q5K, 13),
        (GgmlDType::Q6K, 14),
        (GgmlDType::Q8K, 15),
        (GgmlDType::BF16, 30),
    ];
    for (dt, expected_u32) in dtypes {
        assert_eq!(dt.to_u32(), expected_u32, "to_u32 failed for {dt}");
        assert_eq!(
            GgmlDType::from_u32(expected_u32).unwrap(),
            dt,
            "from_u32 failed for {expected_u32}"
        );
    }
}

#[test]
fn ggml_dtype_unknown() {
    assert!(GgmlDType::from_u32(99).is_err());
}

// ── Q4_0 dequantization ─────────────────────────────────────────

#[test]
fn dequant_q4_0() -> Result<()> {
    // Q4_0: 32 elements per block, 18 bytes per block
    // Construct a known block: scale = 1.0 (f16), qs = all zeros except first nibble
    let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
    let mut block = vec![0u8; 18];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // qs[0] = 0x88 means: low nibble = 8 (value = 8-8 = 0), high nibble = 8 (value = 8-8 = 0)
    // qs[0] = 0x9A means: low = 0xA (10-8=2), high = 0x9 (9-8=1)
    block[2] = 0x9A; // first pair: values 2 and 1

    let qt = QTensor::new(
        block,
        GgmlDType::Q4_0,
        prelude_core::tensor::Shape::from(vec![32]),
    );
    let t = qt.dequantize(&Device::Cpu)?;
    let v: Vec<f32> = t.to_vec1()?;
    assert_eq!(v.len(), 32);
    // First element: (0xA - 8) * 1.0 = 2.0
    assert_eq!(v[0], 2.0);
    // 17th element (second half, index 16): (0x9 - 8) * 1.0 = 1.0
    assert_eq!(v[16], 1.0);
    Ok(())
}

// ── Q8_0 dequantization ─────────────────────────────────────────

#[test]
fn dequant_q8_0() -> Result<()> {
    // Q8_0: 32 elements, scale (f16) + 32 signed bytes = 34 bytes
    let scale_bytes = half::f16::from_f32(0.5).to_le_bytes();
    let mut block = vec![0u8; 34];
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // Set quantized values: 2, -3, 4, ...
    block[2] = 2u8; // i8 = 2
    block[3] = (-3i8) as u8; // i8 = -3
    block[4] = 4u8; // i8 = 4

    let qt = QTensor::new(
        block,
        GgmlDType::Q8_0,
        prelude_core::tensor::Shape::from(vec![32]),
    );
    let t = qt.dequantize(&Device::Cpu)?;
    let v: Vec<f32> = t.to_vec1()?;
    assert_eq!(v[0], 1.0); // 2 * 0.5
    assert_eq!(v[1], -1.5); // -3 * 0.5
    assert_eq!(v[2], 2.0); // 4 * 0.5
    Ok(())
}

// ── F32 passthrough ──────────────────────────────────────────────

#[test]
fn dequant_f32() -> Result<()> {
    let values = vec![1.0f32, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let qt = QTensor::new(
        bytes,
        GgmlDType::F32,
        prelude_core::tensor::Shape::from(vec![4]),
    );
    let t = qt.dequantize(&Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

// ── F16 passthrough ──────────────────────────────────────────────

#[test]
fn dequant_f16() -> Result<()> {
    let values = [1.0f32, -2.5, 3.0, 0.0];
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let qt = QTensor::new(
        bytes,
        GgmlDType::F16,
        prelude_core::tensor::Shape::from(vec![4]),
    );
    let t = qt.dequantize(&Device::Cpu)?;
    let v: Vec<f32> = t.to_vec1()?;
    assert_eq!(v, vec![1.0, -2.5, 3.0, 0.0]);
    Ok(())
}

// ── GGUF file format parsing ─────────────────────────────────────

#[test]
fn gguf_write_read_roundtrip() -> Result<()> {
    use std::io::Cursor;

    // Build a minimal GGUF v3 file in memory
    let mut buf: Vec<u8> = Vec::new();

    // Magic: "GGUF"
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    // Version: 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // Tensor count: 1
    buf.extend_from_slice(&1u64.to_le_bytes());
    // Metadata KV count: 1
    buf.extend_from_slice(&1u64.to_le_bytes());

    // Metadata: "test.key" = u32(42)
    let key = b"test.key";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes()); // key length
    buf.extend_from_slice(key);
    buf.extend_from_slice(&4u32.to_le_bytes()); // value type: U32
    buf.extend_from_slice(&42u32.to_le_bytes()); // value

    // Tensor info: "weights" shape [4], dtype F32, offset 0
    let name = b"weights";
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name);
    buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    buf.extend_from_slice(&4u64.to_le_bytes()); // dim[0] (reversed in GGUF: innermost first)
    buf.extend_from_slice(&0u32.to_le_bytes()); // dtype: F32
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Pad to alignment (32 bytes)
    let pos = buf.len();
    let aligned = (pos + 31) / 32 * 32;
    buf.resize(aligned, 0);

    // Tensor data: 4 floats
    let tensor_data = [1.0f32, 2.0, 3.0, 4.0];
    for &v in &tensor_data {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    // Parse
    let mut cursor = Cursor::new(&buf);
    let content = gguf_file::Content::read(&mut cursor)?;

    // Verify metadata
    assert_eq!(content.metadata.get("test.key").unwrap().to_u32()?, 42);

    // Verify tensor info
    let ti = content.tensor_infos.get("weights").unwrap();
    assert_eq!(ti.ggml_dtype, GgmlDType::F32);
    assert_eq!(ti.shape.dims(), &[4]); // reversed from GGUF's innermost-first

    // Load tensor
    let mut cursor = Cursor::new(&buf);
    let qt = content.tensor(&mut cursor, "weights", &Device::Cpu)?;
    let t = qt.dequantize(&Device::Cpu)?;
    assert_eq!(t.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[test]
fn gguf_missing_tensor() -> Result<()> {
    use std::io::Cursor;

    // Minimal GGUF v3 with 0 tensors
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata count

    let mut cursor = Cursor::new(&buf);
    let content = gguf_file::Content::read(&mut cursor)?;
    assert!(content.tensor_infos.is_empty());

    let mut cursor = Cursor::new(&buf);
    let result = content.tensor(&mut cursor, "nonexistent", &Device::Cpu);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn gguf_bad_magic() {
    use std::io::Cursor;
    let buf = vec![0u8; 32]; // all zeros, wrong magic
    let mut cursor = Cursor::new(&buf);
    assert!(gguf_file::Content::read(&mut cursor).is_err());
}

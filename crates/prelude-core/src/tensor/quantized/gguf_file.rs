//! GGUF file format parser.
//!
//! Parses GGUF v2/v3 files (the standard format for quantized LLMs).
//! Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use crate::tensor::{bail, Device, Result, Shape};
use super::{GgmlDType, QTensor};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

// ── Metadata value types ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn to_u32(&self) -> Result<u32> {
        match self {
            Value::U8(v) => Ok(*v as u32),
            Value::U16(v) => Ok(*v as u32),
            Value::U32(v) => Ok(*v),
            Value::I32(v) => Ok(*v as u32),
            Value::U64(v) => Ok(*v as u32),
            _ => bail!("expected u32, got {self:?}"),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Value::F32(v) => Ok(*v),
            Value::F64(v) => Ok(*v as f32),
            _ => bail!("expected f32, got {self:?}"),
        }
    }

    pub fn to_string(&self) -> Result<&String> {
        match self {
            Value::String(s) => Ok(s),
            _ => bail!("expected string, got {self:?}"),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Value::Array(v) => Ok(v),
            _ => bail!("expected array, got {self:?}"),
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Value::Bool(v) => Ok(*v),
            Value::U8(v) => Ok(*v != 0),
            _ => bail!("expected bool, got {self:?}"),
        }
    }
}

// ── Tensor info ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: Shape,
    pub offset: u64,
}

// ── Content (parsed GGUF file) ───────────────────────────────────

#[derive(Debug)]
pub struct Content {
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

impl Content {
    /// Parse a GGUF file header (metadata + tensor index).
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        // Magic
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (magic: 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x})");
        }

        // Version
        let version = read_u32(reader)?;
        if version < 2 || version > 3 {
            bail!("unsupported GGUF version {version} (supported: 2, 3)");
        }

        // Counts
        let tensor_count = if version >= 3 { read_u64(reader)? } else { read_u32(reader)? as u64 };
        let metadata_kv_count = if version >= 3 { read_u64(reader)? } else { read_u32(reader)? as u64 };

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(reader)?;
            let value = read_value(reader)?;
            metadata.insert(key, value);
        }

        // Alignment
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(DEFAULT_ALIGNMENT as u32) as u64;

        // Parse tensor infos
        let mut tensor_infos = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(reader)?;
            let n_dims = read_u32(reader)? as usize;
            // GGUF stores dimensions in reversed order
            let mut dims = vec![0u64; n_dims];
            for d in dims.iter_mut() {
                *d = if version >= 3 { read_u64(reader)? } else { read_u32(reader)? as u64 };
            }
            dims.reverse();
            let shape = Shape::from(dims.iter().map(|&d| d as usize).collect::<Vec<_>>());
            let ggml_dtype = GgmlDType::from_u32(read_u32(reader)?)?;
            let offset = read_u64(reader)?;
            tensor_infos.insert(name, TensorInfo { ggml_dtype, shape, offset });
        }

        // Tensor data starts after header, aligned
        let pos = reader.stream_position().map_err(io_err)?;
        let tensor_data_offset = align_offset(pos, alignment);

        Ok(Content { metadata, tensor_infos, tensor_data_offset })
    }

    /// Load a single tensor by name from the GGUF file.
    pub fn tensor<R: Read + Seek>(
        &self,
        reader: &mut R,
        name: &str,
        _device: &Device,
    ) -> Result<QTensor> {
        let info = self.tensor_infos.get(name)
            .ok_or_else(|| crate::tensor::Error::Msg(format!("tensor '{name}' not found in GGUF")))?;

        let elem_count = info.shape.elem_count();
        let dtype = info.ggml_dtype;

        // Compute byte size
        let data_size = if dtype.is_float() {
            elem_count * dtype.type_size()
        } else {
            let block_size = dtype.block_size();
            let n_blocks = elem_count / block_size;
            n_blocks * dtype.type_size()
        };

        // Seek and read
        let offset = self.tensor_data_offset + info.offset;
        reader.seek(SeekFrom::Start(offset)).map_err(io_err)?;
        let mut data = vec![0u8; data_size];
        reader.read_exact(&mut data).map_err(io_err)?;

        Ok(QTensor::new(data, dtype, info.shape.clone()))
    }
}

// ── Binary reading helpers ───────────────────────────────────────

fn io_err(e: std::io::Error) -> crate::tensor::Error {
    crate::tensor::Error::Io(e)
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(io_err)?;
    String::from_utf8(buf).map_err(|e| crate::tensor::Error::Msg(format!("invalid UTF-8: {e}")))
}

fn read_value<R: Read>(r: &mut R) -> Result<Value> {
    let vtype = read_u32(r)?;
    read_value_of_type(r, vtype)
}

fn read_value_of_type<R: Read>(r: &mut R, vtype: u32) -> Result<Value> {
    Ok(match vtype {
        0 => Value::U8(read_u8(r)?),
        1 => Value::I8(read_i8(r)?),
        2 => Value::U16(read_u16(r)?),
        3 => Value::I16(read_i16(r)?),
        4 => Value::U32(read_u32(r)?),
        5 => Value::I32(read_i32(r)?),
        6 => Value::F32(read_f32(r)?),
        7 => Value::Bool(read_u8(r)? != 0),
        8 => Value::String(read_string(r)?),
        9 => {
            // Array: element_type (u32) + count (u64) + values
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_value_of_type(r, elem_type)?);
            }
            Value::Array(arr)
        }
        10 => Value::U64(read_u64(r)?),
        11 => Value::I64(read_i64(r)?),
        12 => Value::F64(read_f64(r)?),
        _ => bail!("unknown GGUF value type: {vtype}"),
    })
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) / alignment * alignment
}

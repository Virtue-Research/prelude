//! Own safetensors loading — replaces `candle_core::safetensors`.
//!
//! Uses the `safetensors` crate directly + `memmap2` for zero-copy mmap
//! + `yoke` for self-referential lifetime management.

use crate::tensor::{bail, DType, Device, Error, Result, Tensor, WithDType};
use safetensors::tensor as st;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

// ── DType conversion ─��─────────────────────────────────────────────

fn st_dtype_to_ours(d: st::Dtype) -> Result<DType> {
    match d {
        st::Dtype::U8 => Ok(DType::U8),
        st::Dtype::U32 => Ok(DType::U32),
        st::Dtype::I16 => Ok(DType::I16),
        st::Dtype::I32 => Ok(DType::I32),
        st::Dtype::I64 => Ok(DType::I64),
        st::Dtype::BF16 => Ok(DType::BF16),
        st::Dtype::F16 => Ok(DType::F16),
        st::Dtype::F32 => Ok(DType::F32),
        st::Dtype::F64 => Ok(DType::F64),
        st::Dtype::F8_E4M3 => Ok(DType::F8E4M3),
        other => bail!("unsupported safetensors dtype: {other:?}"),
    }
}

fn our_dtype_to_st(d: DType) -> st::Dtype {
    match d {
        DType::U8 => st::Dtype::U8,
        DType::U32 => st::Dtype::U32,
        DType::I16 => st::Dtype::I16,
        DType::I32 => st::Dtype::I32,
        DType::I64 => st::Dtype::I64,
        DType::BF16 => st::Dtype::BF16,
        DType::F16 => st::Dtype::F16,
        DType::F32 => st::Dtype::F32,
        DType::F64 => st::Dtype::F64,
        DType::F8E4M3 => st::Dtype::F8_E4M3,
    }
}

// ── Core tensor loading ────────────────────────────────────────────

/// Convert raw bytes to a typed slice, handling alignment.
fn convert_slice<T: WithDType>(data: &[u8], shape: &[usize], device: &Device) -> Result<Tensor> {
    let size_in_bytes = <T as WithDType>::DTYPE.size_in_bytes();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // Properly aligned — zero-copy reinterpret.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        Tensor::from_slice(data, shape, device)
    } else {
        // Misaligned — copy into aligned buffer.
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count);
        }
        Tensor::from_slice(&c, shape, device)
    }
}

/// Load a safetensors TensorView into our Tensor on the given device.
fn load_tensor_view(view: &st::TensorView<'_>, device: &Device) -> Result<Tensor> {
    let data = view.data();
    let shape = view.shape();
    match view.dtype() {
        st::Dtype::U8 => convert_slice::<u8>(&data, shape, device),
        st::Dtype::U32 => convert_slice::<u32>(&data, shape, device),
        st::Dtype::I16 => convert_slice::<i16>(&data, shape, device),
        st::Dtype::I32 => convert_slice::<i32>(&data, shape, device),
        st::Dtype::I64 => convert_slice::<i64>(&data, shape, device),
        st::Dtype::BF16 => convert_slice::<half::bf16>(&data, shape, device),
        st::Dtype::F16 => convert_slice::<half::f16>(&data, shape, device),
        st::Dtype::F32 => convert_slice::<f32>(&data, shape, device),
        st::Dtype::F64 => convert_slice::<f64>(&data, shape, device),
        st::Dtype::F8_E4M3 => bail!("F8E4M3 loading not yet supported in own safetensors"),
        // U16 in safetensors → upcast to U32
        st::Dtype::U16 => {
            let elem_count = data.len() / 2;
            let src = if (data.as_ptr() as usize) % 2 == 0 {
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, elem_count) }
                    .to_vec()
            } else {
                let mut buf: Vec<u16> = Vec::with_capacity(elem_count);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        buf.as_mut_ptr() as *mut u8,
                        data.len(),
                    );
                    buf.set_len(elem_count);
                }
                buf
            };
            let upcast: Vec<u32> = src.iter().map(|&x| x as u32).collect();
            Tensor::from_vec(upcast, shape, device)
        }
        dtype => bail!("unsupported safetensors dtype: {dtype:?}"),
    }
}

// ── Yoke helper ────────────────────────────────────────────────────

#[derive(yoke::Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

// ── MmapedSafetensors ──────────────────────────────────────────────

/// Memory-mapped safetensors file(s) — zero-copy access to tensor data.
pub struct MmapedSafetensors {
    safetensors: Vec<yoke::Yoke<SafeTensors_<'static>, memmap2::Mmap>>,
    routing: Option<HashMap<String, usize>>,
}

impl MmapedSafetensors {
    /// Open a single safetensors file via mmap.
    ///
    /// # Safety
    /// Inherited from [`memmap2::MmapOptions`].
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let file = std::fs::File::open(p)
            .map_err(|e| Error::Io(e))?;
        let mmap = memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| Error::Io(e))?;
        let yoked = yoke::Yoke::<SafeTensors_<'static>, memmap2::Mmap>::try_attach_to_cart(
            mmap,
            |data: &[u8]| {
                let st = safetensors::SafeTensors::deserialize(data)
                    .map_err(|e| Error::Msg(format!("safetensors deserialize: {e}")))?;
                Ok::<_, Error>(SafeTensors_(st))
            },
        )?;
        Ok(Self {
            safetensors: vec![yoked],
            routing: None,
        })
    }

    /// Open multiple safetensors files (sharded models).
    /// If a tensor name appears in multiple files, the last file wins.
    ///
    /// # Safety
    /// Inherited from [`memmap2::MmapOptions`].
    pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut routing = HashMap::new();
        let mut safetensors = vec![];
        for (index, p) in paths.iter().enumerate() {
            let p = p.as_ref();
            let file = std::fs::File::open(p)
                .map_err(|e| Error::Io(e))?;
            let mmap = memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| Error::Io(e))?;
            let yoked = yoke::Yoke::<SafeTensors_<'static>, memmap2::Mmap>::try_attach_to_cart(
                mmap,
                |data: &[u8]| {
                    let st = safetensors::SafeTensors::deserialize(data)
                        .map_err(|e| Error::Msg(format!("safetensors deserialize {}: {e}", p.display())))?;
                    Ok::<_, Error>(SafeTensors_(st))
                },
            )?;
            for k in yoked.get().0.names() {
                routing.insert(k.to_string(), index);
            }
            safetensors.push(yoked);
        }
        Ok(Self {
            safetensors,
            routing: Some(routing),
        })
    }

    /// Load a tensor by name onto the given device.
    pub fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        let view = self.get(name)?;
        load_tensor_view(&view, device)
    }

    /// List all tensors across all files.
    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.safetensors
            .iter()
            .flat_map(|s| s.get().0.tensors())
            .collect()
    }

    /// Get a TensorView by name (for metadata inspection).
    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        let index = match &self.routing {
            None => 0,
            Some(routing) => *routing.get(name).ok_or_else(|| {
                Error::CannotFindTensor {
                    path: name.to_string(),
                }
                .bt()
            })?,
        };
        self.safetensors[index]
            .get()
            .0
            .tensor(name)
            .map_err(|e| Error::Msg(format!("{e}")).bt())
    }
}

// ── BufferedSafetensors ────────────────────────────────────────────

/// Safetensors from an owned byte buffer (e.g. downloaded model).
pub struct BufferedSafetensors {
    safetensors: yoke::Yoke<SafeTensors_<'static>, Vec<u8>>,
}

impl BufferedSafetensors {
    pub fn new(buffer: Vec<u8>) -> Result<Self> {
        let safetensors = yoke::Yoke::<SafeTensors_<'static>, Vec<u8>>::try_attach_to_cart(
            buffer,
            |data: &[u8]| {
                let st = safetensors::SafeTensors::deserialize(data)
                    .map_err(|e| Error::Msg(format!("safetensors deserialize: {e}")))?;
                Ok::<_, Error>(SafeTensors_(st))
            },
        )?;
        Ok(Self { safetensors })
    }

    pub fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        let view = self.get(name)?;
        load_tensor_view(&view, device)
    }

    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.safetensors.get().0.tensors()
    }

    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        self.safetensors
            .get()
            .0
            .tensor(name)
            .map_err(|e| Error::Msg(format!("{e}")).bt())
    }
}

// ── Convenience functions ──────────────────────────────────────────

/// Load all tensors from a safetensors file into a HashMap.
pub fn load<P: AsRef<Path>>(filename: P, device: &Device) -> Result<HashMap<String, Tensor>> {
    let data = std::fs::read(filename.as_ref())
        .map_err(|e| Error::Io(e))?;
    load_buffer(&data, device)
}

// ── st::View impl for our Tensor (enables save) ───────────────────

impl st::View for Tensor {
    fn dtype(&self) -> st::Dtype {
        our_dtype_to_st(self.dtype())
    }
    fn shape(&self) -> &[usize] {
        self.dims()
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Owned(tensor_to_bytes(self).unwrap())
    }
    fn data_len(&self) -> usize {
        self.elem_count() * self.dtype().size_in_bytes()
    }
}

impl st::View for &Tensor {
    fn dtype(&self) -> st::Dtype {
        our_dtype_to_st((*self).dtype())
    }
    fn shape(&self) -> &[usize] {
        self.dims()
    }
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Owned(tensor_to_bytes(self).unwrap())
    }
    fn data_len(&self) -> usize {
        self.elem_count() * (*self).dtype().size_in_bytes()
    }
}

/// Save tensors to a safetensors file.
pub fn save<K: AsRef<str> + Ord + std::fmt::Display, P: AsRef<Path>>(
    tensors: &HashMap<K, Tensor>,
    filename: P,
) -> Result<()> {
    st::serialize_to_file(tensors, None, filename.as_ref())
        .map_err(|e| Error::Msg(format!("safetensors save: {e}")))
}

/// Convert a CPU tensor's data to raw bytes.
fn tensor_to_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    fn convert_back_<T: WithDType>(mut vs: Vec<T>) -> Vec<u8> {
        let size_in_bytes = <T as WithDType>::DTYPE.size_in_bytes();
        let length = vs.len() * size_in_bytes;
        let capacity = vs.capacity() * size_in_bytes;
        let ptr = vs.as_mut_ptr() as *mut u8;
        std::mem::forget(vs);
        unsafe { Vec::from_raw_parts(ptr, length, capacity) }
    }

    let tensor = tensor.flatten_all()?;
    match tensor.dtype() {
        DType::U8 => Ok(convert_back_::<u8>(tensor.to_vec1()?)),
        DType::U32 => Ok(convert_back_::<u32>(tensor.to_vec1()?)),
        DType::I16 => Ok(convert_back_::<i16>(tensor.to_vec1()?)),
        DType::I32 => Ok(convert_back_::<i32>(tensor.to_vec1()?)),
        DType::I64 => Ok(convert_back_::<i64>(tensor.to_vec1()?)),
        DType::BF16 => Ok(convert_back_::<half::bf16>(tensor.to_vec1()?)),
        DType::F16 => Ok(convert_back_::<half::f16>(tensor.to_vec1()?)),
        DType::F32 => Ok(convert_back_::<f32>(tensor.to_vec1()?)),
        DType::F64 => Ok(convert_back_::<f64>(tensor.to_vec1()?)),
        dt => bail!("safetensors save: unsupported dtype {dt:?}"),
    }
}

/// Load all tensors from a byte buffer.
pub fn load_buffer(data: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    let st = safetensors::SafeTensors::deserialize(data)
        .map_err(|e| Error::Msg(format!("safetensors: {e}")))?;
    st.tensors()
        .into_iter()
        .map(|(name, view)| {
            let tensor = load_tensor_view(&view, device)?;
            Ok((name, tensor))
        })
        .collect()
}

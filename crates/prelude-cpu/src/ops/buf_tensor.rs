//! Lightweight CPU BF16 tensor — carries shape with data, zero-cost reshape/narrow.
//!
//! Replaces raw `(*const u16, total, dim, ...)` parameter lists with a single
//! `CpuTensor` that provides shape-aware access. All operations are zero-cost
//! pointer arithmetic — no heap allocation, no reference counting.

use std::marker::PhantomData;

use prelude_core::tensor::{Device, Result, Tensor};

/// Zero-overhead CPU BF16 tensor. Borrows data, carries shape.
/// `reshape`/`narrow` are pointer arithmetic — no allocation.
#[derive(Clone, Copy)]
pub struct CpuTensor<'a> {
    data: *const u16,
    len: usize,
    shape: [usize; 4],
    ndim: u8,
    _phantom: PhantomData<&'a [u16]>,
}

// Safety: CpuTensor is a read-only view into data valid for lifetime 'a.
// The PhantomData<&'a [u16]> ensures the compiler enforces the borrow.
unsafe impl<'a> Send for CpuTensor<'a> {}
unsafe impl<'a> Sync for CpuTensor<'a> {}

impl<'a> CpuTensor<'a> {
    /// Wrap a `&[u16]` (BF16 bit patterns) with the given shape.
    ///
    /// # Panics
    /// Panics if `shape` has more than 4 dimensions or if the product of
    /// shape dimensions does not equal `data.len()`.
    #[inline]
    pub fn from_slice(data: &'a [u16], shape: &[usize]) -> Self {
        let ndim = shape.len();
        assert!(ndim <= 4, "CpuTensor: max 4 dims, got {ndim}");
        let product: usize = shape.iter().product();
        assert_eq!(
            product,
            data.len(),
            "CpuTensor: shape product {product} != data len {}",
            data.len()
        );
        let mut s = [0usize; 4];
        s[..ndim].copy_from_slice(shape);
        Self {
            data: data.as_ptr(),
            len: data.len(),
            shape: s,
            ndim: ndim as u8,
            _phantom: PhantomData,
        }
    }

    /// Wrap a raw pointer + length with shape.
    ///
    /// # Safety
    /// `data` must point to at least `len` valid u16 elements for lifetime `'a`.
    #[inline]
    pub unsafe fn from_raw(data: *const u16, len: usize, shape: &[usize]) -> Self {
        let ndim = shape.len();
        debug_assert!(ndim <= 4);
        debug_assert_eq!(shape.iter().product::<usize>(), len);
        let mut s = [0usize; 4];
        s[..ndim].copy_from_slice(shape);
        Self {
            data,
            len,
            shape: s,
            ndim: ndim as u8,
            _phantom: PhantomData,
        }
    }

    /// Extract from a contiguous CPU BF16 candle Tensor.
    ///
    /// The tensor must be contiguous (all BF16 weight tensors from safetensor
    /// loading are contiguous). The returned `CpuTensor` borrows from the
    /// tensor's underlying storage — the tensor must outlive the `CpuTensor`.
    pub fn from_candle(tensor: &'a Tensor) -> Result<Self> {
        let slice = super::tensor_as_u16_slice_pub(tensor)?;
        Ok(unsafe { Self::from_raw(slice.as_ptr(), slice.len(), tensor.dims()) })
    }

    /// Copy data into a new candle BF16 Tensor.
    pub fn to_candle(&self, device: &Device) -> Result<Tensor> {
        let data = self.as_slice();
        let bf16_vec: Vec<half::bf16> = bytemuck::cast_slice::<u16, half::bf16>(data).to_vec();
        Tensor::from_vec(bf16_vec, self.dims(), device)
    }

    /// Reshape to a new shape (must have same total elements). Zero-cost.
    ///
    /// # Panics
    /// Panics if the new shape has more than 4 dims or product != len.
    #[inline]
    pub fn reshape(&self, shape: &[usize]) -> Self {
        let ndim = shape.len();
        assert!(ndim <= 4, "CpuTensor::reshape: max 4 dims, got {ndim}");
        let product: usize = shape.iter().product();
        assert_eq!(
            product, self.len,
            "CpuTensor::reshape: product {product} != len {}",
            self.len
        );
        let mut s = [0usize; 4];
        s[..ndim].copy_from_slice(shape);
        Self {
            data: self.data,
            len: self.len,
            shape: s,
            ndim: ndim as u8,
            _phantom: PhantomData,
        }
    }

    /// Narrow along dimension 0: select `count` rows starting at `start`.
    /// Zero-cost pointer offset.
    ///
    /// # Panics
    /// Panics if `dim != 0` or out of bounds.
    #[inline]
    pub fn narrow(&self, dim: usize, start: usize, count: usize) -> Self {
        assert_eq!(dim, 0, "CpuTensor::narrow: only dim 0 supported");
        assert!(
            start + count <= self.shape[0],
            "CpuTensor::narrow: {start}+{count} > {}",
            self.shape[0]
        );
        let stride0: usize = self.shape[1..self.ndim as usize].iter().product();
        let offset = start * stride0;
        let new_len = count * stride0;
        let mut s = self.shape;
        s[0] = count;
        Self {
            data: unsafe { self.data.add(offset) },
            len: new_len,
            shape: s,
            ndim: self.ndim,
            _phantom: PhantomData,
        }
    }

    /// Size of dimension `d`.
    #[inline]
    pub fn dim(&self, d: usize) -> usize {
        debug_assert!(
            d < self.ndim as usize,
            "CpuTensor::dim({d}) but ndim={}",
            self.ndim
        );
        self.shape[d]
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.ndim as usize
    }

    /// Shape as a slice.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.ndim as usize]
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Raw pointer to data.
    #[inline]
    pub fn as_ptr(&self) -> *const u16 {
        self.data
    }

    /// View as an immutable slice.
    #[inline]
    pub fn as_slice(&self) -> &'a [u16] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}

// ── F32 variant ──────────────────────────────────────────────────────────

/// Zero-overhead CPU F32 tensor. Same design as `CpuTensor` (BF16) but
/// for `f32` data. Will be unified with `CpuTensor` via generics later.
#[derive(Clone, Copy)]
pub struct CpuTensorF32<'a> {
    data: *const f32,
    len: usize,
    shape: [usize; 4],
    ndim: u8,
    _phantom: PhantomData<&'a [f32]>,
}

unsafe impl<'a> Send for CpuTensorF32<'a> {}
unsafe impl<'a> Sync for CpuTensorF32<'a> {}

impl<'a> CpuTensorF32<'a> {
    #[inline]
    pub fn from_slice(data: &'a [f32], shape: &[usize]) -> Self {
        let ndim = shape.len();
        assert!(ndim <= 4, "CpuTensorF32: max 4 dims, got {ndim}");
        let product: usize = shape.iter().product();
        assert_eq!(product, data.len(), "CpuTensorF32: shape product {product} != data len {}", data.len());
        let mut s = [0usize; 4];
        s[..ndim].copy_from_slice(shape);
        Self { data: data.as_ptr(), len: data.len(), shape: s, ndim: ndim as u8, _phantom: PhantomData }
    }

    #[inline]
    pub unsafe fn from_raw(data: *const f32, len: usize, shape: &[usize]) -> Self {
        let ndim = shape.len();
        debug_assert!(ndim <= 4);
        debug_assert_eq!(shape.iter().product::<usize>(), len);
        let mut s = [0usize; 4];
        s[..ndim].copy_from_slice(shape);
        Self { data, len, shape: s, ndim: ndim as u8, _phantom: PhantomData }
    }

    pub fn from_candle(tensor: &'a Tensor) -> Result<Self> {
        let slice = super::tensor_as_f32_slice(tensor)?;
        Ok(unsafe { Self::from_raw(slice.as_ptr(), slice.len(), tensor.dims()) })
    }

    pub fn to_candle(&self, device: &Device) -> Result<Tensor> {
        let data = self.as_slice().to_vec();
        Tensor::from_vec(data, self.dims(), device)
    }

    #[inline]
    pub fn reshape(&self, shape: &[usize]) -> Self {
        let ndim = shape.len();
        assert!(ndim <= 4, "CpuTensorF32::reshape: max 4 dims, got {ndim}");
        let product: usize = shape.iter().product();
        assert_eq!(product, self.len, "CpuTensorF32::reshape: product {product} != len {}", self.len);
        let mut s = [0usize; 4];
        s[..ndim].copy_from_slice(shape);
        Self { data: self.data, len: self.len, shape: s, ndim: ndim as u8, _phantom: PhantomData }
    }

    #[inline]
    pub fn narrow(&self, dim: usize, start: usize, count: usize) -> Self {
        assert_eq!(dim, 0, "CpuTensorF32::narrow: only dim 0 supported");
        assert!(start + count <= self.shape[0], "CpuTensorF32::narrow: {start}+{count} > {}", self.shape[0]);
        let stride0: usize = self.shape[1..self.ndim as usize].iter().product();
        let offset = start * stride0;
        let new_len = count * stride0;
        let mut s = self.shape;
        s[0] = count;
        Self { data: unsafe { self.data.add(offset) }, len: new_len, shape: s, ndim: self.ndim, _phantom: PhantomData }
    }

    #[inline] pub fn dim(&self, d: usize) -> usize { self.shape[d] }
    #[inline] pub fn ndim(&self) -> usize { self.ndim as usize }
    #[inline] pub fn dims(&self) -> &[usize] { &self.shape[..self.ndim as usize] }
    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn as_ptr(&self) -> *const f32 { self.data }
    #[inline] pub fn as_slice(&self) -> &'a [f32] { unsafe { std::slice::from_raw_parts(self.data, self.len) } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_1d() {
        let data = vec![1u16, 2, 3, 4];
        let t = CpuTensor::from_slice(&data, &[4]);
        assert_eq!(t.ndim(), 1);
        assert_eq!(t.dim(0), 4);
        assert_eq!(t.len(), 4);
        assert_eq!(t.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn from_slice_2d() {
        let data = vec![1u16, 2, 3, 4, 5, 6];
        let t = CpuTensor::from_slice(&data, &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
        assert_eq!(t.len(), 6);
    }

    #[test]
    fn reshape() {
        let data = vec![0u16; 12];
        let t = CpuTensor::from_slice(&data, &[3, 4]);
        let t2 = t.reshape(&[4, 3]);
        assert_eq!(t2.dim(0), 4);
        assert_eq!(t2.dim(1), 3);
        assert_eq!(t2.len(), 12);
        assert_eq!(t.as_ptr(), t2.as_ptr());
    }

    #[test]
    fn narrow_dim0() {
        let data: Vec<u16> = (0..12).collect();
        let t = CpuTensor::from_slice(&data, &[4, 3]);
        let n = t.narrow(0, 1, 2);
        assert_eq!(n.dim(0), 2);
        assert_eq!(n.dim(1), 3);
        assert_eq!(n.len(), 6);
        assert_eq!(n.as_slice(), &[3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn narrow_1d() {
        let data: Vec<u16> = (0..10).collect();
        let t = CpuTensor::from_slice(&data, &[10]);
        let n = t.narrow(0, 3, 4);
        assert_eq!(n.dim(0), 4);
        assert_eq!(n.as_slice(), &[3, 4, 5, 6]);
    }

    #[test]
    fn from_slice_3d() {
        let data = vec![0u16; 24];
        let t = CpuTensor::from_slice(&data, &[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
        assert_eq!(t.dim(2), 4);
        assert_eq!(t.len(), 24);
    }

    #[test]
    fn narrow_3d() {
        let data: Vec<u16> = (0..24).collect();
        let t = CpuTensor::from_slice(&data, &[2, 3, 4]);
        let n = t.narrow(0, 1, 1);
        assert_eq!(n.dim(0), 1);
        assert_eq!(n.dim(1), 3);
        assert_eq!(n.dim(2), 4);
        assert_eq!(n.len(), 12);
        assert_eq!(n.as_slice(), &(12..24).collect::<Vec<u16>>());
    }

    #[test]
    fn from_candle_roundtrip() {
        use prelude_core::tensor::{DType, Device};
        let bf16_vals: Vec<half::bf16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let tensor = Tensor::from_vec(bf16_vals, &[2, 3], &Device::Cpu).unwrap();

        let ct = CpuTensor::from_candle(&tensor).unwrap();
        assert_eq!(ct.dims(), &[2, 3]);
        assert_eq!(ct.len(), 6);

        let back = ct.to_candle(&Device::Cpu).unwrap();
        assert_eq!(back.dims(), &[2, 3]);
        assert_eq!(back.dtype(), DType::BF16);
        let vals: Vec<f32> = back.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap().into_iter().flatten().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn from_raw_roundtrip() {
        let data = vec![10u16, 20, 30];
        let t = unsafe { CpuTensor::from_raw(data.as_ptr(), 3, &[3]) };
        assert_eq!(t.dim(0), 3);
        assert_eq!(t.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn copy_semantics() {
        let data = vec![0u16; 6];
        let t = CpuTensor::from_slice(&data, &[2, 3]);
        let t2 = t; // Copy
        // both still usable
        assert_eq!(t.as_ptr(), t2.as_ptr());
        assert_eq!(t.dims(), t2.dims());
    }

    #[test]
    #[should_panic(expected = "shape product 5 != data len 4")]
    fn mismatched_shape() {
        let data = vec![0u16; 4];
        CpuTensor::from_slice(&data, &[5]);
    }

    #[test]
    #[should_panic(expected = "max 4 dims")]
    fn too_many_dims() {
        let data = vec![0u16; 1];
        CpuTensor::from_slice(&data, &[1, 1, 1, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "only dim 0")]
    fn narrow_wrong_dim() {
        let data = vec![0u16; 6];
        let t = CpuTensor::from_slice(&data, &[2, 3]);
        t.narrow(1, 0, 1);
    }

    #[test]
    #[should_panic(expected = "product 8 != len 6")]
    fn reshape_mismatch() {
        let data = vec![0u16; 6];
        let t = CpuTensor::from_slice(&data, &[2, 3]);
        t.reshape(&[2, 4]);
    }
}

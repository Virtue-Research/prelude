//! Tensor layout: shape + strides + start_offset.
//!
//! Copied from candle-core/src/layout.rs, trimmed (removed strided_index methods).

use super::error::{Error, Result};
use super::shape::{Dim, Shape};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Layout {
    shape: Shape,
    stride: Vec<usize>,
    start_offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, stride: Vec<usize>, start_offset: usize) -> Self {
        Self { shape, stride, start_offset }
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self { shape, stride, start_offset }
    }

    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    pub fn dims(&self) -> &[usize] { self.shape.dims() }

    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(&self.shape, "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn shape(&self) -> &Shape { &self.shape }
    pub fn stride(&self) -> &[usize] { &self.stride }
    pub fn start_offset(&self) -> usize { self.start_offset }

    pub fn contiguous_offsets(&self) -> Option<(usize, usize)> {
        if self.is_contiguous() {
            let start_o = self.start_offset;
            Some((start_o, start_o + self.shape.elem_count()))
        } else {
            None
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    pub fn is_fortran_contiguous(&self) -> bool {
        self.shape.is_fortran_contiguous(&self.stride)
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dim >= dims.len() {
            Err(Error::DimOutOfRange { shape: self.shape().clone(), dim: dim as i32, op: "narrow" }.bt())?
        }
        if start + len > dims[dim] {
            Err(Error::NarrowInvalidArgs { shape: self.shape.clone(), dim, start, len, msg: "start + len > dim_len" }.bt())?
        }
        let mut dims = dims.to_vec();
        dims[dim] = len;
        Ok(Self {
            shape: Shape::from(dims),
            stride: self.stride.clone(),
            start_offset: self.start_offset + self.stride[dim] * start,
        })
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if rank <= dim1 || rank <= dim2 {
            Err(Error::UnexpectedNumberOfDims {
                expected: usize::max(dim1, dim2), got: rank, shape: self.shape().clone(),
            }.bt())?
        }
        let mut stride = self.stride().to_vec();
        let mut dims = self.shape().dims().to_vec();
        dims.swap(dim1, dim2);
        stride.swap(dim1, dim2);
        Ok(Self { shape: Shape::from(dims), stride, start_offset: self.start_offset })
    }

    pub fn permute(&self, idxs: &[usize]) -> Result<Self> {
        let is_permutation = idxs.len() == self.shape.rank() && (0..idxs.len()).all(|i| idxs.contains(&i));
        if !is_permutation {
            crate::bail!("dimension mismatch in permute, tensor {:?}, dims: {:?}", self.dims(), idxs)
        }
        let stride = self.stride();
        let dims = self.shape().dims();
        let mut perm_stride = stride.to_vec();
        let mut perm_dims = dims.to_vec();
        for (i, &idx) in idxs.iter().enumerate() {
            perm_stride[i] = stride[idx];
            perm_dims[i] = dims[idx];
        }
        Ok(Self { shape: Shape::from(perm_dims), stride: perm_stride, start_offset: self.start_offset })
    }

    /// Unsqueeze: insert a dim=1 at the given position.
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if dim > rank {
            Err(Error::DimOutOfRange { shape: self.shape.clone(), dim: dim as i32, op: "unsqueeze" }.bt())?
        }
        let mut dims = self.dims().to_vec();
        let mut stride = self.stride.clone();
        let s = if dim < rank { stride[dim] * dims[dim] } else if rank > 0 { 1 } else { 1 };
        dims.insert(dim, 1);
        stride.insert(dim, s);
        Ok(Self { shape: Shape::from(dims), stride, start_offset: self.start_offset })
    }

    /// Squeeze: remove a dim=1 at the given position.
    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            Err(Error::DimOutOfRange { shape: self.shape.clone(), dim: dim as i32, op: "squeeze" }.bt())?
        }
        if dims[dim] != 1 {
            return Ok(self.clone()); // no-op if dim != 1
        }
        let mut new_dims = dims.to_vec();
        let mut new_stride = self.stride.clone();
        new_dims.remove(dim);
        new_stride.remove(dim);
        Ok(Self { shape: Shape::from(new_dims), stride: new_stride, start_offset: self.start_offset })
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.rank() < self.shape().rank() {
            return Err(Error::BroadcastIncompatibleShapes {
                src_shape: self.shape().clone(), dst_shape: shape,
            }.bt());
        }
        let added_dims = shape.rank() - self.shape().rank();
        let mut stride = vec![0; added_dims];
        for (&dst_dim, (&src_dim, &src_stride)) in shape.dims()[added_dims..]
            .iter()
            .zip(self.dims().iter().zip(self.stride()))
        {
            let s = if dst_dim == src_dim {
                src_stride
            } else if src_dim != 1 {
                return Err(Error::BroadcastIncompatibleShapes {
                    src_shape: self.shape().clone(), dst_shape: shape,
                }.bt());
            } else {
                0
            };
            stride.push(s)
        }
        Ok(Self { shape, stride, start_offset: self.start_offset })
    }

    /// Create a StridedIndex iterator over all elements in logical order.
    pub fn strided_index(&self) -> StridedIndex<'_> {
        StridedIndex::new(self.dims(), self.stride(), self.start_offset)
    }
}

// ── StridedIndex ───────────────────────────────────────────────────

/// Iterator over physical storage indices for a strided tensor layout.
/// Yields indices in logical (row-major) order.
pub struct StridedIndex<'a> {
    dims: &'a [usize],
    stride: &'a [usize],
    offset: usize,
    multi_index: Vec<usize>,
    total: usize,
    pos: usize,
}

impl<'a> StridedIndex<'a> {
    fn new(dims: &'a [usize], stride: &'a [usize], offset: usize) -> Self {
        let total: usize = dims.iter().product();
        Self {
            dims,
            stride,
            offset,
            multi_index: vec![0; dims.len()],
            total,
            pos: 0,
        }
    }
}

impl Iterator for StridedIndex<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.pos >= self.total {
            return None;
        }
        // Compute physical index from multi_index + strides
        let mut idx = self.offset;
        for (i, &s) in self.stride.iter().enumerate() {
            idx += self.multi_index[i] * s;
        }
        self.pos += 1;

        // Advance multi_index (rightmost dimension increments first)
        if self.pos < self.total {
            for d in (0..self.dims.len()).rev() {
                self.multi_index[d] += 1;
                if self.multi_index[d] < self.dims[d] {
                    break;
                }
                self.multi_index[d] = 0;
            }
        }
        Some(idx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.total - self.pos;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for StridedIndex<'_> {}

//! Shape, dimension indexing, and shape-with-hole helpers.
//!
//! Copied from candle-core/src/shape.rs with trimming:
//! - Removed `impl crate::Tensor` blocks (those live in tensor/mod.rs)
#![allow(clippy::redundant_closure_call)]
use super::{Error, Result};

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

pub const SCALAR: Shape = Shape(vec![]);

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.dims())
    }
}

impl<const C: usize> From<&[usize; C]> for Shape {
    fn from(dims: &[usize; C]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self(vec![d1])
    }
}

macro_rules! impl_from_tuple {
    ($tuple:ty, $($index:tt),+) => {
        impl From<$tuple> for Shape {
            fn from(d: $tuple) -> Self {
                Self(vec![$(d.$index,)+])
            }
        }
    }
}

impl_from_tuple!((usize,), 0);
impl_from_tuple!((usize, usize), 0, 1);
impl_from_tuple!((usize, usize, usize), 0, 1, 2);
impl_from_tuple!((usize, usize, usize, usize), 0, 1, 2, 3);
impl_from_tuple!((usize, usize, usize, usize, usize), 0, 1, 2, 3, 4);
impl_from_tuple!((usize, usize, usize, usize, usize, usize), 0, 1, 2, 3, 4, 5);

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl Shape {
    pub fn from_dims(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn into_dims(self) -> Vec<usize> {
        self.0
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self, "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    pub fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride: Vec<_> = self
            .0
            .iter()
            .rev()
            .scan(1, |prod, u| {
                let prod_pre_mult = *prod;
                *prod *= u;
                Some(prod_pre_mult)
            })
            .collect();
        stride.reverse();
        stride
    }

    pub fn is_contiguous(&self, stride: &[usize]) -> bool {
        if self.0.len() != stride.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.0.iter()).rev() {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    pub fn is_fortran_contiguous(&self, stride: &[usize]) -> bool {
        if self.0.len() != stride.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride.iter().zip(self.0.iter()) {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim;
        }
        true
    }

    pub fn extend(mut self, additional_dims: &[usize]) -> Self {
        self.0.extend(additional_dims);
        self
    }

    pub fn broadcast_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<Shape> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = vec![0; bcast_ndims];
        for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
            let rev_idx = bcast_ndims - idx;
            let l_value = if lhs_ndims < rev_idx { 1 } else { lhs_dims[lhs_ndims - rev_idx] };
            let r_value = if rhs_ndims < rev_idx { 1 } else { rhs_dims[rhs_ndims - rev_idx] };
            *bcast_value = if l_value == r_value {
                l_value
            } else if l_value == 1 {
                r_value
            } else if r_value == 1 {
                l_value
            } else {
                Err(Error::ShapeMismatchBinaryOp { lhs: lhs.clone(), rhs: rhs.clone(), op }.bt())?
            }
        }
        Ok(Shape::from(bcast_dims))
    }

    pub fn broadcast_shape_matmul(&self, rhs: &Self) -> Result<(Shape, Shape)> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
            crate::bail!("only 2d matrixes are supported {lhs:?} {rhs:?}")
        }
        let (m, lhs_k) = (lhs_dims[lhs_dims.len() - 2], lhs_dims[lhs_dims.len() - 1]);
        let (rhs_k, n) = (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1]);
        if lhs_k != rhs_k {
            crate::bail!("different inner dimensions in broadcast matmul {lhs:?} {rhs:?}")
        }
        let lhs_b = Self::from(&lhs_dims[..lhs_dims.len() - 2]);
        let rhs_b = Self::from(&rhs_dims[..rhs_dims.len() - 2]);
        let bcast = lhs_b.broadcast_shape_binary_op(&rhs_b, "broadcast_matmul")?;
        let bcast_dims = bcast.dims();
        let bcast_lhs = [bcast_dims, &[m, lhs_k]].concat();
        let bcast_rhs = [bcast_dims, &[rhs_k, n]].concat();
        Ok((Shape::from(bcast_lhs), Shape::from(bcast_rhs)))
    }
}

// ── extract_dims (Shape methods only, Tensor methods in mod.rs) ────

macro_rules! extract_dims {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        pub fn $fn_name(dims: &[usize]) -> Result<$out_type> {
            if dims.len() != $cnt {
                Err(Error::UnexpectedNumberOfDims {
                    expected: $cnt,
                    got: dims.len(),
                    shape: Shape::from(dims),
                }.bt())
            } else {
                Ok($dims(dims))
            }
        }

        impl Shape {
            pub fn $fn_name(&self) -> Result<$out_type> {
                $fn_name(self.0.as_slice())
            }
        }
    };
}

extract_dims!(dims0, 0, |_: &[usize]| (), ());
extract_dims!(dims1, 1, |d: &[usize]| d[0], usize);
extract_dims!(dims2, 2, |d: &[usize]| (d[0], d[1]), (usize, usize));
extract_dims!(dims3, 3, |d: &[usize]| (d[0], d[1], d[2]), (usize, usize, usize));
extract_dims!(dims4, 4, |d: &[usize]| (d[0], d[1], d[2], d[3]), (usize, usize, usize, usize));
extract_dims!(dims5, 5, |d: &[usize]| (d[0], d[1], d[2], d[3], d[4]), (usize, usize, usize, usize, usize));

// ── Dim / Dims / ShapeWithOneHole traits ───────────────────────────

pub trait Dim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= shape.dims().len() {
            Err(Error::DimOutOfRange { shape: shape.clone(), dim: dim as i32, op }.bt())?
        } else {
            Ok(dim)
        }
    }
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim > shape.dims().len() {
            Err(Error::DimOutOfRange { shape: shape.clone(), dim: dim as i32, op }.bt())?
        } else {
            Ok(dim)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum D {
    Minus1,
    Minus2,
    Minus(usize),
}

impl D {
    fn out_of_range(&self, shape: &Shape, op: &'static str) -> Error {
        let dim = match self {
            Self::Minus1 => -1,
            Self::Minus2 => -2,
            Self::Minus(u) => -(*u as i32),
        };
        Error::DimOutOfRange { shape: shape.clone(), dim, op }.bt()
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 if rank >= 1 => Ok(rank - 1),
            Self::Minus2 if rank >= 2 => Ok(rank - 2),
            Self::Minus(u) if *u > 0 && rank >= *u => Ok(rank - *u),
            _ => Err(self.out_of_range(shape, op)),
        }
    }
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 => Ok(rank),
            Self::Minus2 if rank >= 1 => Ok(rank - 1),
            Self::Minus(u) if *u > 0 && rank + 1 >= *u => Ok(rank + 1 - *u),
            _ => Err(self.out_of_range(shape, op)),
        }
    }
}

pub trait Dims: Sized {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>>;

    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dims = self.to_indexes_internal(shape, op)?;
        for (i, &dim) in dims.iter().enumerate() {
            if dims[..i].contains(&dim) {
                Err(Error::DuplicateDimIndex { shape: shape.clone(), dims: dims.clone(), op }.bt())?
            }
            if dim >= shape.rank() {
                Err(Error::DimOutOfRange { shape: shape.clone(), dim: dim as i32, op }.bt())?
            }
        }
        Ok(dims)
    }
}

impl Dims for Vec<usize> {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> { Ok(self) }
}
impl<const N: usize> Dims for [usize; N] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> { Ok(self.to_vec()) }
}
impl Dims for &[usize] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> { Ok(self.to_vec()) }
}
impl Dims for () {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> { Ok(vec![]) }
}
impl<D: Dim + Sized> Dims for D {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Ok(vec![self.to_index(shape, op)?])
    }
}
impl<D: Dim> Dims for (D,) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Ok(vec![self.0.to_index(shape, op)?])
    }
}
impl<D1: Dim, D2: Dim> Dims for (D1, D2) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Ok(vec![self.0.to_index(shape, op)?, self.1.to_index(shape, op)?])
    }
}
impl<D1: Dim, D2: Dim, D3: Dim> Dims for (D1, D2, D3) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        Ok(vec![self.0.to_index(shape, op)?, self.1.to_index(shape, op)?, self.2.to_index(shape, op)?])
    }
}

pub trait ShapeWithOneHole {
    fn into_shape(self, el_count: usize) -> Result<Shape>;
}

impl<S: Into<Shape>> ShapeWithOneHole for S {
    fn into_shape(self, _el_count: usize) -> Result<Shape> {
        Ok(self.into())
    }
}

impl ShapeWithOneHole for ((),) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { Ok(el_count.into()) }
}

fn hole_size(el_count: usize, prod_d: usize, s: &dyn std::fmt::Debug) -> Result<usize> {
    if prod_d == 0 { crate::bail!("cannot reshape tensor of {el_count} elements to {s:?}") }
    if !el_count.is_multiple_of(prod_d) { crate::bail!("cannot reshape tensor with {el_count} elements to {s:?}") }
    Ok(el_count / prod_d)
}

impl ShapeWithOneHole for ((), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { let ((), d1) = self; Ok((hole_size(el_count, d1, &self)?, d1).into()) }
}
impl ShapeWithOneHole for (usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { let (d1, ()) = self; Ok((d1, hole_size(el_count, d1, &self)?).into()) }
}
impl ShapeWithOneHole for ((), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { let ((), d1, d2) = self; Ok((hole_size(el_count, d1 * d2, &self)?, d1, d2).into()) }
}
impl ShapeWithOneHole for (usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { let (d1, (), d2) = self; Ok((d1, hole_size(el_count, d1 * d2, &self)?, d2).into()) }
}
impl ShapeWithOneHole for (usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { let (d1, d2, ()) = self; Ok((d1, d2, hole_size(el_count, d1 * d2, &self)?).into()) }
}
impl ShapeWithOneHole for ((), usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { let ((), d1, d2, d3) = self; Ok((hole_size(el_count, d1 * d2 * d3, &self)?, d1, d2, d3).into()) }
}
impl ShapeWithOneHole for (usize, (), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { let (d1, (), d2, d3) = self; Ok((d1, hole_size(el_count, d1 * d2 * d3, &self)?, d2, d3).into()) }
}
impl ShapeWithOneHole for (usize, usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { let (d1, d2, (), d3) = self; Ok((d1, d2, hole_size(el_count, d1 * d2 * d3, &self)?, d3).into()) }
}
impl ShapeWithOneHole for (usize, usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> { let (d1, d2, d3, ()) = self; Ok((d1, d2, d3, hole_size(el_count, d1 * d2 * d3, &self)?).into()) }
}

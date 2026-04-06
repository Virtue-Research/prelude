//! Element type trait for tensors.

use super::DType;

/// Trait for element types that can be stored in tensors.
pub trait WithDType: Sized + Copy + Send + Sync + 'static {
    /// Our DType tag for this element type.
    const DTYPE: DType;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

macro_rules! impl_with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;
            fn from_f64(v: f64) -> Self { $from_f64(v) }
            fn to_f64(self) -> f64 { $to_f64(self) }
        }
    };
}

use half::{bf16, f16};

impl_with_dtype!(u8,  U8,  |v: f64| v as u8,  |v: u8| v as f64);
impl_with_dtype!(u32, U32, |v: f64| v as u32, |v: u32| v as f64);
impl_with_dtype!(i16, I16, |v: f64| v as i16, |v: i16| v as f64);
impl_with_dtype!(i32, I32, |v: f64| v as i32, |v: i32| v as f64);
impl_with_dtype!(i64, I64, |v: f64| v as i64, |v: i64| v as f64);
impl_with_dtype!(f16, F16, f16::from_f64, f16::to_f64);
impl_with_dtype!(bf16, BF16, bf16::from_f64, bf16::to_f64);
impl_with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
impl_with_dtype!(f64, F64, |v: f64| v, |v: f64| v);
// F8E4M3: skipped until float8 is a direct dep of prelude-core.

/// Marker trait for integer element types.
pub trait IntDType: WithDType {
    fn is_true(&self) -> bool;
    fn as_usize(&self) -> usize;
}

macro_rules! impl_int_dtype {
    ($($ty:ty),+) => {
        $(impl IntDType for $ty {
            fn is_true(&self) -> bool { *self != 0 }
            fn as_usize(&self) -> usize { *self as usize }
        })+
    };
}
impl_int_dtype!(u8, u32, i16, i32, i64);

/// Marker trait for floating-point element types.
pub trait FloatDType: WithDType {}

impl FloatDType for f16 {}
impl FloatDType for bf16 {}
impl FloatDType for f32 {}
impl FloatDType for f64 {}

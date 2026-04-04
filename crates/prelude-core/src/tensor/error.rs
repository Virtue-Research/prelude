//! Prelude error types.
//!
//! Subset of candle-core's error enum — only the variants we actually use.

use super::shape::Shape;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
    Metal { gpu_id: usize },
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

#[derive(thiserror::Error)]
pub enum Error {
    // === DType Errors ===
    #[error("{msg}, expected: {expected}, got: {got}")]
    UnexpectedDType {
        msg: &'static str,
        expected: super::DType,
        got: super::DType,
    },

    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp {
        lhs: super::DType,
        rhs: super::DType,
        op: &'static str,
    },

    #[error("unsupported dtype {0:?} for op {1}")]
    UnsupportedDTypeForOp(super::DType, &'static str),

    // === Dimension Errors ===
    #[error("{op}: dimension index {dim} out of range for shape {shape:?}")]
    DimOutOfRange { shape: Shape, dim: i32, op: &'static str },

    #[error("{op}: duplicate dim index {dims:?} for shape {shape:?}")]
    DuplicateDimIndex { shape: Shape, dims: Vec<usize>, op: &'static str },

    // === Shape Errors ===
    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims { expected: usize, got: usize, shape: Shape },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedShape { msg: String, expected: Shape, got: Shape },

    #[error("shape mismatch, got buffer of size {buffer_size} which is incompatible with shape {shape:?}")]
    ShapeMismatch { buffer_size: usize, shape: Shape },

    #[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    ShapeMismatchBinaryOp { lhs: Shape, rhs: Shape, op: &'static str },

    #[error("shape mismatch in cat for dim {dim}, shape for arg 1: {first_shape:?} shape for arg {n}: {nth_shape:?}")]
    ShapeMismatchCat { dim: usize, first_shape: Shape, n: usize, nth_shape: Shape },

    #[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
    BroadcastIncompatibleShapes { src_shape: Shape, dst_shape: Shape },

    // === Device Errors ===
    #[error("device mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DeviceMismatchBinaryOp { lhs: DeviceLocation, rhs: DeviceLocation, op: &'static str },

    // === Op Specific Errors ===
    #[error("narrow invalid args {msg}: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
    NarrowInvalidArgs { shape: Shape, dim: usize, start: usize, len: usize, msg: &'static str },

    #[error("{op} invalid index {index} with dim size {size}")]
    InvalidIndex { op: &'static str, index: usize, size: usize },

    #[error("{op} only supports contiguous tensors")]
    RequiresContiguous { op: &'static str },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneTensor { op: &'static str },

    #[error("the candle crate has not been built with cuda support")]
    NotCompiledWithCudaSupport,

    #[error("the candle crate has not been built with metal support")]
    NotCompiledWithMetalSupport,

    #[error("cannot find tensor {path}")]
    CannotFindTensor { path: String },

    // === Wrapped / Generic Errors ===
    #[error(transparent)]
    Cuda(Box<dyn std::error::Error + Send + Sync>),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("{0}")]
    Wrapped(Box<dyn std::fmt::Display + Send + Sync>),

    #[error("{inner}\n{backtrace}")]
    WithBacktrace { inner: Box<Self>, backtrace: Box<std::backtrace::Backtrace> },

    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn wrap(err: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Wrapped(Box::new(err)).bt()
    }

    pub fn msg(err: impl std::fmt::Display) -> Self {
        Self::Msg(err.to_string()).bt()
    }

    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }
}

/// Bridge: convert candle_core::Error into our Error.
impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Msg(e.to_string())
    }
}

/// Bridge: convert our Error back to candle_core::Error.
impl From<Error> for candle_core::Error {
    fn from(e: Error) -> Self {
        candle_core::Error::Msg(e.to_string())
    }
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::tensor::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::tensor::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::tensor::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}

//! Reduce operations via cubek::reduce.
//!
//! Wraps cubek's reduce API for use by CubeCLTensorOps.

use cubecl::prelude::*;
use cubecl::server::Handle;

use cubek::reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, unit::UnitStrategy},
};

use super::elementwise::dtype_to_storage;
use crate::ops::traits::ReduceOp;
use crate::tensor::{DType, Shape};

/// Map Prelude ReduceOp to cubek ReduceOperationConfig.
fn to_cubek_config(op: ReduceOp) -> ReduceOperationConfig {
    match op {
        ReduceOp::Sum => ReduceOperationConfig::Sum,
        ReduceOp::Max => ReduceOperationConfig::Max,
        ReduceOp::Min => ReduceOperationConfig::Min,
        ReduceOp::ArgMax => ReduceOperationConfig::ArgMax,
        ReduceOp::ArgMin => ReduceOperationConfig::ArgMin,
    }
}

/// Compute output shape for reduce (dim set to 1).
fn reduce_output_shape(input_shape: &Shape, dim: usize) -> Shape {
    let mut dims = input_shape.dims().to_vec();
    dims[dim] = 1;
    Shape::from(dims)
}

/// Launch a cubek reduce operation.
pub fn launch_reduce<R: Runtime>(
    client: &cubecl::client::ComputeClient<R>,
    input_handle: Handle,
    input_shape: &Shape,
    input_dtype: DType,
    dim: usize,
    op: ReduceOp,
) -> std::result::Result<(Handle, Shape, DType), cubek::reduce::ReduceError> {
    let config = to_cubek_config(op);
    let out_shape = reduce_output_shape(input_shape, dim);

    let is_arg = matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin);
    let out_dtype = if is_arg { DType::U32 } else { input_dtype };
    let out_size = out_shape.elem_count() * out_dtype.size_in_bytes();
    let out_handle = client.empty(out_size);

    let in_th = cubecl::std::tensor::TensorHandle::new_contiguous(
        input_shape.dims().to_vec(), input_handle, dtype_to_storage(input_dtype),
    );
    let out_th = cubecl::std::tensor::TensorHandle::new_contiguous(
        out_shape.dims().to_vec(), out_handle.clone(), dtype_to_storage(out_dtype),
    );

    let in_elem = match dtype_to_storage(input_dtype) {
        cubecl::ir::StorageType::Scalar(e) => e,
        _ => unreachable!(),
    };
    let out_elem = if is_arg {
        Some(match dtype_to_storage(out_dtype) {
            cubecl::ir::StorageType::Scalar(e) => e,
            _ => unreachable!(),
        })
    } else {
        None
    };
    let dtypes = config.precision(in_elem, out_elem);

    let strategy = ReduceStrategy {
        routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        vectorization: VectorizationStrategy {
            parallel_output_vectorization: false,
        },
    };

    cubek::reduce::reduce::<R>(
        client,
        in_th.binding(),
        out_th.binding(),
        dim,
        strategy,
        config,
        dtypes,
    )?;

    Ok((out_handle, out_shape, out_dtype))
}

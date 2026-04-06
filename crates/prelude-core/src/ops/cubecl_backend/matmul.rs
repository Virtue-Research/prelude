//! Matmul via cubek::matmul.

use cubecl::prelude::*;
use cubecl::server::Handle;

use cubek::matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek::std::InputBinding;

use super::elementwise::dtype_to_storage;
use crate::tensor::{DType, Shape};

pub fn launch_matmul<R: Runtime>(
    client: &cubecl::client::ComputeClient<R>,
    a_handle: Handle,
    a_shape: &Shape,
    b_handle: Handle,
    b_shape: &Shape,
    dtype: DType,
) -> std::result::Result<(Handle, Shape), String> {
    let a_dims = a_shape.dims();
    let b_dims = b_shape.dims();
    let m = a_dims[a_dims.len() - 2];
    let n = b_dims[b_dims.len() - 1];

    let mut out_dims = a_dims[..a_dims.len() - 2].to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let storage = dtype_to_storage(dtype);
    let out_handle = client.empty(out_shape.elem_count() * dtype.size_in_bytes());

    let a_th = cubecl::std::tensor::TensorHandle::new_contiguous(a_dims.to_vec(), a_handle, storage);
    let b_th = cubecl::std::tensor::TensorHandle::new_contiguous(b_dims.to_vec(), b_handle, storage);
    let out_th = cubecl::std::tensor::TensorHandle::new_contiguous(out_shape.dims().to_vec(), out_handle.clone(), storage);

    let lhs = InputBinding::Normal(a_th.binding(), storage);
    let rhs = InputBinding::Normal(b_th.binding(), storage);

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: storage,
        rhs: storage,
        out: storage,
    });

    cubek::matmul::launch::launch_naive::launch_ref::<R>(
        client, lhs, rhs, out_th.binding(), &mut dtypes,
    ).map_err(|e| format!("{e:?}"))?;

    Ok((out_handle, out_shape))
}

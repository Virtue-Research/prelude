use crate::tensor::{Result, Tensor};

pub trait ConvOps: Send + Sync {
    fn conv1d(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor>;

    fn conv_transpose1d(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Result<Tensor>;

    fn conv2d(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Tensor>;
}

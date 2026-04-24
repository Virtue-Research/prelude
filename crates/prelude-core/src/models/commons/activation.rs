use crate::tensor::{Module, Result, Tensor, D};

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    Mish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}

impl Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu()?.sqr(),
            Self::Relu6 => xs.clamp(0f32, 6f32),
            Self::Silu => xs.silu(),
            Self::Sigmoid => sigmoid(xs),
            Self::HardSigmoid => hard_sigmoid(xs),
            Self::Swiglu => swiglu(xs),
            Self::Swish => xs * sigmoid(xs)?,
            Self::HardSwish => xs * hard_sigmoid(xs)?,
            Self::Mish => mish(xs),
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => leaky_relu(xs, negative_slope),
            Self::GeluPytorchTanh => xs.gelu(),
        }
    }
}

fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    (xs.neg()?.exp()? + 1.0)?.recip()
}

fn hard_sigmoid(xs: &Tensor) -> Result<Tensor> {
    ((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32)
}

fn mish(xs: &Tensor) -> Result<Tensor> {
    let softplus = (xs.exp()? + 1.0)?.log()?;
    xs * softplus.tanh()?
}

fn leaky_relu(xs: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = xs.zeros_like()?;
    let pos = xs.maximum(&zeros)?;
    let neg = (xs.minimum(&zeros)? * negative_slope)?;
    pos + neg
}

fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let chunks = xs.chunk(2, D::Minus1)?;
    &chunks[0].silu()? * &chunks[1]
}

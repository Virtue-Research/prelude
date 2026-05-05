use crate::loading::var_builder::VarBuilder;
use crate::tensor::{Module, Result, Tensor};

/// Wraps a weight tensor of shape `(vocab_size, hidden_size)`.
/// `forward` performs `index_select` on dimension 0.
#[derive(Clone, Debug)]
pub struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for Embedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        values.reshape(final_dims)
    }
}

/// Load an embedding layer from a `VarBuilder` (reads `"weight"` tensor).
pub fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let weight = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(weight, hidden_size))
}

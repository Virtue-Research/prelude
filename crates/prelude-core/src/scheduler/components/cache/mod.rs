pub mod block_manager;
pub mod deltanet_pool;
pub mod kv_buf;
pub mod manager;
pub mod prefix_cache;
pub mod prefix_index;
pub mod prefix_plan;

pub(crate) mod prefix_ops;

use crate::tensor::DType;

/// Declares what cache a layer needs. Models return one spec per layer.
/// The engine groups layers by spec and allocates accordingly.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerCacheSpec {
    /// Standard KV cache (softmax attention). Paged.
    Attention {
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
    },
    /// Recurrent state (Mamba, DeltaNet, RWKV). Fixed size per request.
    Recurrent {
        state_shapes: Vec<Vec<usize>>,
        state_dtypes: Vec<DType>,
    },
    /// No cache (diffusion, embedding, encoder layers).
    None,
}

#[cfg(test)]
mod layer_cache_spec_tests {
    use super::*;

    #[test]
    fn attention_spec() {
        let spec = LayerCacheSpec::Attention {
            num_kv_heads: 8,
            head_dim: 128,
            sliding_window: Some(4096),
        };
        match &spec {
            LayerCacheSpec::Attention {
                num_kv_heads,
                head_dim,
                sliding_window,
            } => {
                assert_eq!(*num_kv_heads, 8);
                assert_eq!(*head_dim, 128);
                assert_eq!(*sliding_window, Some(4096));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn recurrent_spec() {
        let spec = LayerCacheSpec::Recurrent {
            state_shapes: vec![vec![16, 256], vec![64, 256]],
            state_dtypes: vec![DType::BF16, DType::BF16],
        };
        match &spec {
            LayerCacheSpec::Recurrent {
                state_shapes,
                state_dtypes,
            } => {
                assert_eq!(state_shapes.len(), 2);
                assert_eq!(state_dtypes.len(), 2);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn none_spec() {
        assert_eq!(LayerCacheSpec::None, LayerCacheSpec::None);
    }

    #[test]
    fn mixed_model_specs() {
        // Simulate a hybrid model: 24 attention layers + 4 recurrent layers
        let mut specs = Vec::new();
        for _ in 0..24 {
            specs.push(LayerCacheSpec::Attention {
                num_kv_heads: 8,
                head_dim: 128,
                sliding_window: None,
            });
        }
        for _ in 0..4 {
            specs.push(LayerCacheSpec::Recurrent {
                state_shapes: vec![vec![4, 512], vec![64, 512]],
                state_dtypes: vec![DType::F32, DType::BF16],
            });
        }
        assert_eq!(specs.len(), 28);
        assert_eq!(
            specs
                .iter()
                .filter(|s| matches!(s, LayerCacheSpec::Attention { .. }))
                .count(),
            24
        );
        assert_eq!(
            specs
                .iter()
                .filter(|s| matches!(s, LayerCacheSpec::Recurrent { .. }))
                .count(),
            4
        );
    }
}

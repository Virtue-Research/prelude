use super::{
    Gemma3ClassifierConfig, Gemma3Config, Gemma3EmbeddingDenseLayer, Gemma3ForCausalLM,
    Gemma3ForEmbedding, Gemma3ForSequenceClassification,
};
use candle_nn::VarBuilder;
use crate::engine::EngineError;
use crate::engine::{CommonModelConfig, EmbeddingSemantics, RuntimeCaps, TaskKind, WeightsBackend};
use crate::models::architectures::meta::{
    ArchSpec, AuxiliaryVarBuilder, ParsedModelConfig, candle_model_err,
    inject_num_labels_if_missing, parse_value,
};

const ARCHITECTURE_ALIASES: &[&str] = &["Gemma3", "Gemma3Text"];
const MODEL_TYPE_ALIASES: &[&str] = &["gemma3", "gemma3_text", "gemma2", "gemma"];
const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate, TaskKind::Classify, TaskKind::Embed];

#[derive(Debug, Clone, Copy)]
enum Gemma3WeightLayout {
    FlatText,
    NestedLanguageModel,
}

/// Opaque config stored in `ParsedModelConfig.arch_config` for Gemma3.
enum Gemma3ArchConfig {
    Dense {
        cfg: Gemma3Config,
        layout: Gemma3WeightLayout,
    },
    Classifier {
        cfg: Gemma3ClassifierConfig,
        layout: Gemma3WeightLayout,
    },
    Embedding {
        cfg: Gemma3Config,
        layout: Gemma3WeightLayout,
    },
}

fn common_from_gemma3(cfg: &Gemma3Config) -> CommonModelConfig {
    CommonModelConfig {
        vocab_size: cfg.vocab_size,
        num_hidden_layers: cfg.num_hidden_layers,
        max_position_embeddings: cfg.max_position_embeddings,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
    }
}

fn infer_weight_layout(raw: &serde_json::Value) -> Gemma3WeightLayout {
    if raw.get("text_config").is_some() {
        Gemma3WeightLayout::NestedLanguageModel
    } else {
        Gemma3WeightLayout::FlatText
    }
}

fn parse_gemma_text_config(
    raw: &serde_json::Value,
    description: &str,
) -> Result<Gemma3Config, EngineError> {
    if let Some(text_config) = raw.get("text_config") {
        parse_value(text_config.clone(), description)
    } else {
        parse_value(raw.clone(), description)
    }
}

pub(crate) struct Gemma3ModelBuildContext<'a> {
    pub main_vb: VarBuilder<'a>,
    pub embedding: Option<&'a EmbeddingSemantics>,
    pub auxiliary: &'a [AuxiliaryVarBuilder],
}

impl<'a> Gemma3ModelBuildContext<'a> {
    fn auxiliary_vb(&self, module_path: &str) -> Option<VarBuilder<'static>> {
        self.auxiliary
            .iter()
            .find(|aux| aux.module_path == module_path)
            .map(|aux| aux.vb.clone())
    }
}

fn embedding_semantics_or_default(ctx: &Gemma3ModelBuildContext<'_>) -> EmbeddingSemantics {
    ctx.embedding.cloned().unwrap_or_default()
}

pub(crate) fn build_gemma3_model_with_context(
    arch_config: &dyn std::any::Any,
    ctx: &Gemma3ModelBuildContext<'_>,
) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
    let cfg = arch_config
        .downcast_ref::<Gemma3ArchConfig>()
        .ok_or_else(|| EngineError::Internal("unexpected arch config type for Gemma3".into()))?;

    let backbone_vb = |layout| match layout {
        Gemma3WeightLayout::FlatText => ctx.main_vb.clone().pp("model"),
        Gemma3WeightLayout::NestedLanguageModel => ctx
            .main_vb
            .clone()
            .pp("model")
            .pp("language_model")
            .pp("model"),
    };

    match cfg {
        Gemma3ArchConfig::Dense { cfg, layout } => Ok(Box::new(
            Gemma3ForCausalLM::new_with_parts(cfg, backbone_vb(*layout), ctx.main_vb.clone())
                .map_err(candle_model_err)?,
        )),
        Gemma3ArchConfig::Classifier { cfg, layout } => Ok(Box::new(
            Gemma3ForSequenceClassification::new_with_parts(
                cfg,
                backbone_vb(*layout),
                ctx.main_vb.clone(),
            )
            .map_err(candle_model_err)?,
        )),
        Gemma3ArchConfig::Embedding { cfg, layout } => {
            let semantics = embedding_semantics_or_default(ctx);
            let mut dense_layers = Vec::with_capacity(semantics.dense_layers.len());
            for dense in &semantics.dense_layers {
                let vb = ctx.auxiliary_vb(&dense.module_path).ok_or_else(|| {
                    EngineError::Internal(format!(
                        "missing embedding weights for Gemma3 dense module {}",
                        dense.module_path
                    ))
                })?;
                dense_layers
                    .push(Gemma3EmbeddingDenseLayer::new(dense, vb).map_err(candle_model_err)?);
            }
            Ok(Box::new(
                Gemma3ForEmbedding::new(cfg, backbone_vb(*layout), &semantics, &dense_layers)
                    .map_err(candle_model_err)?,
            ))
        }
    }
}

pub(crate) struct Gemma3ArchSpec;

pub(crate) static GEMMA3_ARCH_SPEC: Gemma3ArchSpec = Gemma3ArchSpec;

impl ArchSpec for Gemma3ArchSpec {
    fn name(&self) -> &'static str {
        "gemma3"
    }

    fn architecture_aliases(&self) -> &'static [&'static str] {
        ARCHITECTURE_ALIASES
    }

    fn model_type_aliases(&self) -> &'static [&'static str] {
        MODEL_TYPE_ALIASES
    }

    fn supported_tasks(&self) -> &'static [TaskKind] {
        SUPPORTED_TASKS
    }

    fn parse_config(
        &self,
        task: TaskKind,
        raw: &serde_json::Value,
        _content: &str,
    ) -> Result<ParsedModelConfig, EngineError> {
        let layout = infer_weight_layout(raw);
        match task {
            TaskKind::Generate => {
                let cfg = parse_gemma_text_config(raw, "Gemma3 config")?;
                let common = common_from_gemma3(&cfg);
                Ok(ParsedModelConfig {
                    common,
                    deltanet: None,
                    arch_config: Box::new(Gemma3ArchConfig::Dense { cfg, layout }),
                })
            }
            TaskKind::Classify => {
                let json = inject_num_labels_if_missing(raw);
                let base = parse_gemma_text_config(&json, "Gemma3 classifier text config")?;
                let num_labels = json
                    .get("num_labels")
                    .and_then(|value| value.as_u64())
                    .ok_or_else(|| {
                        EngineError::InvalidRequest(
                            "Gemma3 classifier config is missing `num_labels`".into(),
                        )
                    })? as usize;
                let cfg = Gemma3ClassifierConfig {
                    base,
                    num_labels,
                    label2id: json
                        .get("label2id")
                        .and_then(|value| serde_json::from_value(value.clone()).ok()),
                    id2label: json
                        .get("id2label")
                        .and_then(|value| serde_json::from_value(value.clone()).ok()),
                };
                let common = common_from_gemma3(&cfg.base);
                Ok(ParsedModelConfig {
                    common,
                    deltanet: None,
                    arch_config: Box::new(Gemma3ArchConfig::Classifier { cfg, layout }),
                })
            }
            TaskKind::Embed => {
                let cfg = parse_gemma_text_config(raw, "Gemma3 embedding config")?;
                let common = common_from_gemma3(&cfg);
                Ok(ParsedModelConfig {
                    common,
                    deltanet: None,
                    arch_config: Box::new(Gemma3ArchConfig::Embedding { cfg, layout }),
                })
            }
        }
    }

    fn build_model(
        &self,
        arch_config: &dyn std::any::Any,
        vb: VarBuilder<'_>,
    ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
        let ctx = Gemma3ModelBuildContext {
            main_vb: vb,
            embedding: None,
            auxiliary: &[],
        };
        build_gemma3_model_with_context(arch_config, &ctx)
    }

    fn runtime_caps(
        &self,
        task: TaskKind,
        backend: WeightsBackend,
        device: &candle_core::Device,
    ) -> RuntimeCaps {
        let is_safetensors = backend == WeightsBackend::Safetensors;
        let is_generate = task == TaskKind::Generate;

        RuntimeCaps {
            supports_kv_cache: is_safetensors && is_generate,
            supports_prefix_cache: false,
            supports_paged_attn: false,
            supports_varlen: cfg!(feature = "flash-attn-v3") && device.is_cuda() && is_safetensors,
            supports_deltanet: false,
            supports_cuda_graph: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::embedding_semantics_or_default;
    use crate::engine::{
        EmbeddingDenseLayerSpec, EmbeddingNormalization, EmbeddingPooling, EmbeddingSemantics,
    };
    use crate::models::architectures::gemma3::meta::Gemma3ModelBuildContext;
    use candle_nn::VarBuilder;

    #[test]
    fn gemma3_embedding_defaults_to_last_token_without_modules_metadata() {
        let ctx = Gemma3ModelBuildContext {
            main_vb: VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu),
            embedding: None,
            auxiliary: &[],
        };

        let semantics = embedding_semantics_or_default(&ctx);

        assert_eq!(semantics.pooling, EmbeddingPooling::LastToken);
        assert_eq!(semantics.normalization, EmbeddingNormalization::None);
        assert!(semantics.dense_layers.is_empty());
    }

    #[test]
    fn gemma3_embedding_uses_provided_semantics_when_available() {
        let provided = EmbeddingSemantics {
            pooling: EmbeddingPooling::Mean,
            normalization: EmbeddingNormalization::L2,
            dense_layers: vec![EmbeddingDenseLayerSpec {
                module_path: "2_Dense".into(),
                in_features: 10,
                out_features: 12,
                bias: true,
                activation: Default::default(),
            }],
        };
        let ctx = Gemma3ModelBuildContext {
            main_vb: VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu),
            embedding: Some(&provided),
            auxiliary: &[],
        };

        let semantics = embedding_semantics_or_default(&ctx);

        assert_eq!(semantics, provided);
    }
}

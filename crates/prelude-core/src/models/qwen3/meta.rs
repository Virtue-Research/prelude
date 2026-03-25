use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config as Qwen3Config;

use super::{
    Qwen3ClassifierConfig, Qwen3ForEmbedding, Qwen3ForSequenceClassification, Qwen3ModelForCausalLM,
};
use crate::engine::EngineError;
use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
use crate::models::registry::{
    ArchSpec, ParsedModelConfig, candle_model_err, inject_num_labels_if_missing, parse_json,
    parse_value,
};

const ARCHITECTURE_ALIASES: &[&str] = &["Qwen3", "Qwen3Model"];
const MODEL_TYPE_ALIASES: &[&str] = &["qwen3"];
const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate, TaskKind::Classify, TaskKind::Embed];

/// Opaque config stored in `ParsedModelConfig.arch_config` for Qwen3.
/// Differentiates the 3 task-specific config types.
enum Qwen3ArchConfig {
    Dense(Qwen3Config),
    Classifier(Qwen3ClassifierConfig),
    Embedding(Qwen3Config),
}

fn common_from_qwen3(cfg: &Qwen3Config) -> CommonModelConfig {
    CommonModelConfig {
        vocab_size: cfg.vocab_size,
        num_hidden_layers: cfg.num_hidden_layers,
        max_position_embeddings: cfg.max_position_embeddings,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
    }
}

pub(crate) struct Qwen3ArchSpec;

pub(crate) static QWEN3_ARCH_SPEC: Qwen3ArchSpec = Qwen3ArchSpec;

impl ArchSpec for Qwen3ArchSpec {
    fn name(&self) -> &'static str {
        "qwen3"
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
        content: &str,
    ) -> Result<ParsedModelConfig, EngineError> {
        match task {
            TaskKind::Generate => {
                let cfg = parse_json::<Qwen3Config>(content, "Qwen3 config")?;
                let common = common_from_qwen3(&cfg);
                Ok(ParsedModelConfig {
                    common,
                    deltanet: None,
                    arch_config: Box::new(Qwen3ArchConfig::Dense(cfg)),
                })
            }
            TaskKind::Classify => {
                let json = inject_num_labels_if_missing(raw);
                let cfg = parse_value::<Qwen3ClassifierConfig>(json, "Qwen3 classifier config")?;
                let common = common_from_qwen3(&cfg.base);
                Ok(ParsedModelConfig {
                    common,
                    deltanet: None,
                    arch_config: Box::new(Qwen3ArchConfig::Classifier(cfg)),
                })
            }
            TaskKind::Embed => {
                let cfg = parse_json::<Qwen3Config>(content, "Qwen3 embedding config")?;
                let common = common_from_qwen3(&cfg);
                Ok(ParsedModelConfig {
                    common,
                    deltanet: None,
                    arch_config: Box::new(Qwen3ArchConfig::Embedding(cfg)),
                })
            }
        }
    }

    fn build_model(
        &self,
        arch_config: &dyn std::any::Any,
        vb: VarBuilder<'_>,
    ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
        let cfg = arch_config
            .downcast_ref::<Qwen3ArchConfig>()
            .ok_or_else(|| EngineError::Internal("unexpected arch config type for Qwen3".into()))?;
        match cfg {
            Qwen3ArchConfig::Dense(c) => Ok(Box::new(
                Qwen3ModelForCausalLM::new(c, vb).map_err(candle_model_err)?,
            )),
            Qwen3ArchConfig::Classifier(c) => Ok(Box::new(
                Qwen3ForSequenceClassification::new(c, vb).map_err(candle_model_err)?,
            )),
            Qwen3ArchConfig::Embedding(c) => Ok(Box::new(
                Qwen3ForEmbedding::new(c, vb).map_err(candle_model_err)?,
            )),
        }
    }

    fn runtime_caps(
        &self,
        task: TaskKind,
        backend: WeightsBackend,
        device: &candle_core::Device,
    ) -> RuntimeCaps {
        let is_safetensors = backend == WeightsBackend::Safetensors;
        let supports_cuda_varlen = (cfg!(feature = "cuda")
            || cfg!(feature = "flash-attn-v4")
            || cfg!(feature = "flashinfer"))
            && device.is_cuda()
            && is_safetensors;
        RuntimeCaps {
            supports_kv_cache: is_safetensors && task == TaskKind::Generate,
            supports_prefix_cache: is_safetensors
                && cfg!(feature = "cuda")
                && device.is_cuda(),
            supports_paged_attn: cfg!(feature = "cuda") && device.is_cuda() && is_safetensors,
            supports_varlen: supports_cuda_varlen,
            supports_deltanet: false,
            supports_cuda_graph: supports_cuda_varlen && task == TaskKind::Generate,
        }
    }
}

use candle_nn::VarBuilder;

use super::{Qwen3MoeConfig, Qwen3MoeModelForCausalLM};
use crate::engine::EngineError;
use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
use crate::models::architectures::meta::{
    candle_model_err, parse_json, ArchSpec, ParsedModelConfig,
};

const ARCHITECTURE_ALIASES: &[&str] = &["Qwen3Moe", "Qwen3MoeModel"];
const MODEL_TYPE_ALIASES: &[&str] = &["qwen3_moe"];
const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate];

pub(crate) struct Qwen3MoeArchSpec;

pub(crate) static QWEN3_MOE_ARCH_SPEC: Qwen3MoeArchSpec = Qwen3MoeArchSpec;

impl ArchSpec for Qwen3MoeArchSpec {
    fn name(&self) -> &'static str {
        "qwen3_moe"
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
        _task: TaskKind,
        _raw: &serde_json::Value,
        content: &str,
    ) -> Result<ParsedModelConfig, EngineError> {
        let cfg = parse_json::<Qwen3MoeConfig>(content, "Qwen3 MoE config")?;
        let common = CommonModelConfig {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            max_position_embeddings: cfg.max_position_embeddings,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        };
        Ok(ParsedModelConfig {
            common,
            deltanet: None,
            arch_config: Box::new(cfg),
        })
    }

    fn build_model(
        &self,
        arch_config: &dyn std::any::Any,
        vb: VarBuilder<'_>,
    ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
        let cfg = arch_config
            .downcast_ref::<Qwen3MoeConfig>()
            .ok_or_else(|| {
                EngineError::Internal("unexpected arch config type for Qwen3Moe".into())
            })?;
        Ok(Box::new(
            Qwen3MoeModelForCausalLM::new(cfg, vb).map_err(candle_model_err)?,
        ))
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
            supports_prefix_cache: is_safetensors
                && cfg!(feature = "cuda")
                && device.is_cuda(),
            supports_paged_attn: cfg!(feature = "cuda")
                && device.is_cuda()
                && is_safetensors,
            supports_varlen: cfg!(feature = "cuda") && device.is_cuda() && is_safetensors,
            supports_deltanet: false,
            supports_cuda_graph: false,
        }
    }
}

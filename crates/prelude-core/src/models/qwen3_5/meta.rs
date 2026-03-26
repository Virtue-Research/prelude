use crate::loading::var_builder::VarBuilder;

use super::{Qwen3_5Config, Qwen3_5ForCausalLM};
use crate::cache::deltanet_pool::DeltaNetPoolConfig;
use crate::engine::EngineError;
use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};
use crate::models::registry::{
    candle_model_err, parse_json, ArchSpec, ParsedModelConfig,
};

const ARCHITECTURE_ALIASES: &[&str] = &["Qwen3_5", "Qwen35", "Qwen3_5Moe", "Qwen35Moe"];
const MODEL_TYPE_ALIASES: &[&str] = &["qwen3_5_text", "qwen3_5", "qwen3_5_moe_text", "qwen3_5_moe"];
const SUPPORTED_TASKS: &[TaskKind] = &[TaskKind::Generate];

fn deltanet_config_from(cfg: &Qwen3_5Config) -> DeltaNetPoolConfig {
    let num_deltanet_layers = (0..cfg.num_hidden_layers)
        .filter(|i| (i + 1) % cfg.full_attention_interval != 0)
        .count();
    DeltaNetPoolConfig {
        num_deltanet_layers,
        num_v_heads: cfg.linear_num_value_heads,
        head_k_dim: cfg.linear_key_head_dim,
        head_v_dim: cfg.linear_value_head_dim,
        conv_dim: cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim,
        conv_kernel: cfg.linear_conv_kernel_dim,
    }
}

pub(crate) struct Qwen3_5ArchSpec;

pub(crate) static QWEN3_5_ARCH_SPEC: Qwen3_5ArchSpec = Qwen3_5ArchSpec;
inventory::submit!(crate::models::registry::ArchSpecEntry::new(&QWEN3_5_ARCH_SPEC));

impl ArchSpec for Qwen3_5ArchSpec {
    fn name(&self) -> &'static str {
        "qwen3_5"
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
        let cfg = parse_json::<Qwen3_5Config>(content, "Qwen3.5 config")?;
        let common = CommonModelConfig {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            max_position_embeddings: cfg.max_position_embeddings,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        };
        let deltanet = Some(deltanet_config_from(&cfg));
        Ok(ParsedModelConfig {
            common,
            deltanet,
            arch_config: Box::new(cfg),
        })
    }

    fn build_model(
        &self,
        arch_config: &dyn std::any::Any,
        vb: VarBuilder<'_>,
    ) -> Result<Box<dyn crate::models::ModelForward>, EngineError> {
        let cfg = arch_config.downcast_ref::<Qwen3_5Config>().ok_or_else(|| {
            EngineError::Internal("unexpected arch config type for Qwen3.5".into())
        })?;
        Ok(Box::new(
            Qwen3_5ForCausalLM::new(cfg, vb).map_err(candle_model_err)?,
        ))
    }

    fn runtime_caps(
        &self,
        task: TaskKind,
        backend: WeightsBackend,
        device: &candle_core::Device,
    ) -> RuntimeCaps {
        let is_safetensors = backend == WeightsBackend::Safetensors;
        let _is_generate = task == TaskKind::Generate;

        RuntimeCaps {
            supports_kv_cache: false,
            supports_prefix_cache: false,
            supports_paged_attn: cfg!(feature = "cuda")
                && device.is_cuda()
                && is_safetensors,
            supports_varlen: cfg!(feature = "cuda") && device.is_cuda() && is_safetensors,
            supports_deltanet: true,
            supports_cuda_graph: false,
        }
    }
}

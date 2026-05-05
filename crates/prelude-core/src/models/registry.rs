use crate::loading::var_builder::VarBuilder;
use crate::tensor::Device;
use serde::de::DeserializeOwned;

use crate::cache::deltanet_pool::DeltaNetPoolConfig;
use crate::engine::EngineError;
use crate::engine::{CommonModelConfig, RuntimeCaps, TaskKind, WeightsBackend};

/// Parsed model configuration returned by `ArchSpec::parse_config()`.
/// Contains common fields needed by the engine plus an opaque arch-specific config.
pub(crate) struct ParsedModelConfig {
    pub common: CommonModelConfig,
    pub deltanet: Option<DeltaNetPoolConfig>,
    pub arch_config: Box<dyn std::any::Any + Send>,
}

#[derive(Clone)]
/// Additional weight bundle stored outside the main model checkpoint.
/// Currently used for sentence-transformers embedding heads such as Dense modules.
pub(crate) struct AuxiliaryVarBuilder {
    pub module_path: String,
    pub vb: VarBuilder<'static>,
}

pub(crate) trait ArchSpec: Sync {
    /// Human-readable architecture name for logging/display.
    fn name(&self) -> &'static str;
    fn architecture_aliases(&self) -> &'static [&'static str];
    fn model_type_aliases(&self) -> &'static [&'static str];
    fn supported_tasks(&self) -> &'static [TaskKind];
    fn parse_config(
        &self,
        task: TaskKind,
        raw: &serde_json::Value,
        content: &str,
    ) -> Result<ParsedModelConfig, EngineError>;
    fn build_model(
        &self,
        arch_config: &dyn std::any::Any,
        vb: VarBuilder<'_>,
    ) -> Result<Box<dyn crate::models::ModelForward>, EngineError>;
    fn runtime_caps(&self, task: TaskKind, backend: WeightsBackend, device: &Device)
    -> RuntimeCaps;

    fn supports_task(&self, task: TaskKind) -> bool {
        self.supported_tasks().contains(&task)
    }

    /// GGUF architecture aliases (e.g. `["qwen3"]`, `["qwen35", "qwen35moe"]`).
    /// Default: empty (model does not support GGUF loading).
    fn gguf_aliases(&self) -> &'static [&'static str] {
        &[]
    }

    /// Build a model from GGUF content. Returns the model, common config,
    /// and EOS token IDs extracted from GGUF metadata.
    ///
    /// Default: returns an error (model does not support GGUF).
    fn load_gguf(
        &self,
        _ct: crate::tensor::quantized::gguf_file::Content,
        _reader: &mut std::fs::File,
        _device: &Device,
    ) -> Result<GgufLoadResult, EngineError> {
        Err(EngineError::Unavailable(format!(
            "GGUF loading not supported for architecture '{}'",
            self.name()
        )))
    }
}

/// Result of loading a model from GGUF format via `ArchSpec::load_gguf()`.
pub(crate) struct GgufLoadResult {
    pub model: Box<dyn crate::models::ModelForward>,
    pub common: CommonModelConfig,
    pub deltanet: Option<DeltaNetPoolConfig>,
    pub eos_token_ids: Vec<u32>,
}

pub(crate) fn resolve_architecture_name(name: &str) -> Option<(&'static dyn ArchSpec, TaskKind)> {
    let (prefix, suffix) = name.split_once("For")?;
    Some((
        find_arch_spec_by_architecture_prefix(prefix)?,
        task_from_architecture_suffix(suffix)?,
    ))
}

pub(crate) fn find_arch_spec_by_architecture_prefix(prefix: &str) -> Option<&'static dyn ArchSpec> {
    all_arch_specs()
        .iter()
        .copied()
        .find(|spec| matches_any_alias(prefix, spec.architecture_aliases()))
}

pub(crate) fn find_arch_spec_by_gguf_arch(arch: &str) -> Option<&'static dyn ArchSpec> {
    all_arch_specs()
        .iter()
        .copied()
        .find(|spec| matches_any_alias(arch, spec.gguf_aliases()))
}

pub(crate) fn find_arch_spec_by_model_type(model_type: &str) -> Option<&'static dyn ArchSpec> {
    all_arch_specs()
        .iter()
        .copied()
        .find(|spec| matches_any_alias(model_type, spec.model_type_aliases()))
}

pub(crate) fn parse_json<T: DeserializeOwned>(
    content: &str,
    description: &str,
) -> Result<T, EngineError> {
    serde_json::from_str(content)
        .map_err(|e| EngineError::Internal(format!("failed to parse {description}: {e}")))
}

pub(crate) fn parse_value<T: DeserializeOwned>(
    value: serde_json::Value,
    description: &str,
) -> Result<T, EngineError> {
    serde_json::from_value(value)
        .map_err(|e| EngineError::Internal(format!("failed to parse {description}: {e}")))
}

pub(crate) fn inject_num_labels_if_missing(raw: &serde_json::Value) -> serde_json::Value {
    let mut json = raw.clone();
    if json.get("num_labels").is_none() || json["num_labels"].is_null() {
        let n = json
            .get("id2label")
            .and_then(|v| v.as_object())
            .map(|m| m.len())
            .unwrap_or(2);
        json["num_labels"] = serde_json::Value::Number(serde_json::Number::from(n));
    }
    json
}

pub(crate) fn unsupported_task_error(arch_name: &str, task: TaskKind) -> EngineError {
    EngineError::InvalidRequest(format!(
        "task {:?} is not supported for architecture {:?}",
        task, arch_name
    ))
}

pub(crate) fn candle_model_err(e: crate::tensor::Error) -> EngineError {
    EngineError::Internal(format!("candle error: {e}"))
}

/// Wrapper for `inventory` auto-registration.
///
/// Each model's `meta.rs` calls `inventory::submit!(ArchSpecEntry::new(&MY_ARCH_SPEC))`
/// to register itself. No need to edit this file when adding a new model.
pub(crate) struct ArchSpecEntry {
    pub spec: &'static dyn ArchSpec,
}

impl ArchSpecEntry {
    pub const fn new(spec: &'static dyn ArchSpec) -> Self {
        Self { spec }
    }
}

inventory::collect!(ArchSpecEntry);

fn all_arch_specs() -> Vec<&'static dyn ArchSpec> {
    inventory::iter::<ArchSpecEntry>()
        .map(|entry| entry.spec)
        .collect()
}

fn task_from_architecture_suffix(suffix: &str) -> Option<TaskKind> {
    let normalized = normalize_identifier(suffix);
    match normalized.as_str() {
        "causallm" | "conditionalgeneration" | "textgeneration" | "generation" => {
            Some(TaskKind::Generate)
        }
        "sequenceclassification" => Some(TaskKind::Classify),
        "embedding" | "textembedding" => Some(TaskKind::Embed),
        _ => None,
    }
}

fn matches_any_alias(input: &str, aliases: &'static [&'static str]) -> bool {
    let normalized = normalize_identifier(input);
    aliases
        .iter()
        .any(|alias| normalize_identifier(alias) == normalized)
}

fn normalize_identifier(input: &str) -> String {
    input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Registry lookup tests ──────────────────────────────────────

    #[test]
    fn find_gguf_arch_qwen3_unsupported() {
        // Qwen3 deliberately does NOT register a GGUF alias — there's no
        // loader yet (see qwen3.rs comment). This test guards against
        // accidentally re-adding the alias without also wiring the
        // loader, which would surface a misleading "not supported for
        // architecture" error instead of the clearer "unsupported GGUF
        // architecture" from the registry-not-found path.
        let spec = find_arch_spec_by_gguf_arch("qwen3");
        assert!(
            spec.is_none(),
            "qwen3 GGUF must not be registered until a loader exists"
        );
    }

    #[test]
    fn find_gguf_arch_qwen35() {
        let spec = find_arch_spec_by_gguf_arch("qwen35");
        assert!(spec.is_some(), "qwen35 should be registered as a GGUF arch");
        assert_eq!(spec.unwrap().name(), "qwen3_5");
    }

    #[test]
    fn find_gguf_arch_qwen35moe() {
        let spec = find_arch_spec_by_gguf_arch("qwen35moe");
        assert!(
            spec.is_some(),
            "qwen35moe should be registered as a GGUF arch"
        );
        assert_eq!(spec.unwrap().name(), "qwen3_5");
    }

    #[test]
    fn find_gguf_arch_unknown() {
        let spec = find_arch_spec_by_gguf_arch("unknown_arch_xyz");
        assert!(spec.is_none(), "unknown arch should not match");
    }

    // ── Architecture alias tests ───────────────────────────────────

    #[test]
    fn find_arch_by_prefix_qwen3() {
        let spec = find_arch_spec_by_architecture_prefix("Qwen3");
        assert!(spec.is_some());
        assert_eq!(spec.unwrap().name(), "qwen3");
    }

    #[test]
    fn find_arch_by_model_type_qwen3() {
        let spec = find_arch_spec_by_model_type("qwen3");
        assert!(spec.is_some());
        assert_eq!(spec.unwrap().name(), "qwen3");
    }

    #[test]
    fn find_arch_by_model_type_unknown() {
        let spec = find_arch_spec_by_model_type("nonexistent_model_type");
        assert!(spec.is_none());
    }

    // ── Resolution tests ───────────────────────────────────────────

    #[test]
    fn resolve_qwen3_for_causal_lm() {
        let result = resolve_architecture_name("Qwen3ForCausalLM");
        assert!(result.is_some());
        let (spec, task) = result.unwrap();
        assert_eq!(spec.name(), "qwen3");
        assert_eq!(task, TaskKind::Generate);
    }

    #[test]
    fn resolve_qwen3_for_classification() {
        let result = resolve_architecture_name("Qwen3ForSequenceClassification");
        assert!(result.is_some());
        let (spec, task) = result.unwrap();
        assert_eq!(spec.name(), "qwen3");
        assert_eq!(task, TaskKind::Classify);
    }

    // ── Normalize identifier tests ─────────────────────────────────

    #[test]
    fn normalize_strips_underscores() {
        assert_eq!(normalize_identifier("Qwen3_5"), "qwen35");
    }

    #[test]
    fn normalize_case_insensitive() {
        assert_eq!(normalize_identifier("QWEN3"), "qwen3");
        assert_eq!(normalize_identifier("qwen3"), "qwen3");
    }
}

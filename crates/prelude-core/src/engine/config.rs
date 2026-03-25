use super::*;
use crate::models::registry::{
    find_arch_spec_by_model_type, resolve_architecture_name, unsupported_task_error, ArchSpec,
    ParsedModelConfig,
};
use tracing::info;

pub(crate) struct ResolvedModelConfig {
    pub parsed: ParsedModelConfig,
    pub spec: &'static dyn ArchSpec,
    pub task: TaskKind,
    pub eos_token_ids: Vec<u32>,
}

pub(crate) fn load_model_config(
    model_path: &Path,
    task_override: TaskOverride,
) -> Result<ResolvedModelConfig, EngineError> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path).map_err(|e| {
        EngineError::Internal(format!("failed to read {}: {e}", config_path.display()))
    })?;
    let has_st_config = model_path
        .join("config_sentence_transformers.json")
        .exists();
    parse_model_config(content, task_override, has_st_config)
}

pub(crate) fn parse_model_config_for_source(
    content: &str,
    task_override: TaskOverride,
    has_sentence_transformers_config: bool,
) -> Result<ResolvedModelConfig, EngineError> {
    parse_model_config(content.to_string(), task_override, has_sentence_transformers_config)
}

fn parse_model_config(
    content: impl AsRef<str>,
    task_override: TaskOverride,
    has_sentence_transformers_config: bool,
) -> Result<ResolvedModelConfig, EngineError> {
    let content = content.as_ref();
    let raw: serde_json::Value = serde_json::from_str(content)
        .map_err(|e| EngineError::Internal(format!("failed to parse config.json: {e}")))?;
    let architecture_names = raw
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|value| value.as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let model_type = raw.get("model_type").and_then(|v| v.as_str());
    let (spec, task) = resolve_model_type(
        &architecture_names,
        model_type,
        task_override,
        has_sentence_transformers_config,
    )?;
    info!(
        architectures = ?architecture_names,
        model_type,
        resolved_arch = spec.name(),
        resolved_task = ?task,
        ?task_override,
        has_sentence_transformers_config,
        "resolved model metadata"
    );

    let parsed = spec.parse_config(task, &raw, content)?;
    let eos_token_ids = extract_eos_token_ids(&raw, task)?;
    Ok(ResolvedModelConfig {
        parsed,
        spec,
        task,
        eos_token_ids,
    })
}

fn resolve_model_type(
    architecture_names: &[String],
    model_type: Option<&str>,
    task_override: TaskOverride,
    has_sentence_transformers_config: bool,
) -> Result<(&'static dyn ArchSpec, TaskKind), EngineError> {
    let architecture_match = architecture_names
        .iter()
        .map(String::as_str)
        .find_map(resolve_architecture_name);
    let spec = architecture_match
        .map(|(spec, _)| spec)
        .or_else(|| model_type.and_then(find_arch_spec_by_model_type))
        .ok_or_else(|| {
            EngineError::InvalidRequest(format!(
                "unable to resolve model architecture from config metadata: architectures={architecture_names:?}, model_type={model_type:?}"
            ))
        })?;
    let detected_task = match architecture_match.map(|(_, task)| task) {
        Some(TaskKind::Classify) | Some(TaskKind::Embed) => architecture_match.unwrap().1,
        Some(TaskKind::Generate) | None => {
            if has_sentence_transformers_config {
                TaskKind::Embed
            } else {
                architecture_match
                    .map(|(_, task)| task)
                    .unwrap_or(TaskKind::Generate)
            }
        }
    };
    let task = task_override.resolve(detected_task);
    if !spec.supports_task(task) {
        return Err(unsupported_task_error(spec.name(), task));
    }
    Ok((spec, task))
}

fn extract_eos_token_ids(raw: &serde_json::Value, task: TaskKind) -> Result<Vec<u32>, EngineError> {
    if task != TaskKind::Generate {
        return Ok(Vec::new());
    }

    let value = raw.get("eos_token_id").ok_or_else(|| {
        EngineError::InvalidRequest(
            "generation model config is missing required field `eos_token_id`".into(),
        )
    })?;

    let ids = match value {
        serde_json::Value::Number(n) => n.as_u64().map(|id| vec![id as u32]).ok_or_else(|| {
            EngineError::InvalidRequest(
                "generation model config has invalid non-integer `eos_token_id`".into(),
            )
        })?,
        serde_json::Value::Array(values) => {
            let mut ids = Vec::with_capacity(values.len());
            for value in values {
                let id = value.as_u64().ok_or_else(|| {
                    EngineError::InvalidRequest(
                        "generation model config has non-integer entry in `eos_token_id`".into(),
                    )
                })?;
                ids.push(id as u32);
            }
            ids
        }
        _ => {
            return Err(EngineError::InvalidRequest(
                "generation model config has invalid `eos_token_id` type".into(),
            ))
        }
    };

    if ids.is_empty() {
        return Err(EngineError::InvalidRequest(
            "generation model config has empty `eos_token_id`".into(),
        ));
    }

    Ok(ids)
}

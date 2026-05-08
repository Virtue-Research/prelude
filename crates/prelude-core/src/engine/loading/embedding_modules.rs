use std::path::{Path, PathBuf};

use crate::engine::{
    EmbeddingActivation, EmbeddingDenseLayerSpec, EmbeddingNormalization, EmbeddingPooling,
    EmbeddingSemantics, EngineError, TaskKind, load_var_builder_from_filenames, tensor_err,
};
use crate::models::registry::AuxiliaryVarBuilder;
use crate::tensor::{DType, Device};

#[derive(Debug, serde::Deserialize)]
struct SentenceTransformerModuleEntry {
    idx: usize,
    path: String,
    #[serde(rename = "type")]
    module_type: String,
}

#[derive(Debug, serde::Deserialize)]
struct SentenceTransformerPoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
}

#[derive(Debug, serde::Deserialize)]
struct SentenceTransformerDenseConfig {
    in_features: usize,
    out_features: usize,
    #[serde(default)]
    bias: Option<bool>,
    #[serde(default)]
    activation_function: Option<String>,
    #[serde(default)]
    activation: Option<String>,
}

pub(crate) struct LoadedEmbeddingModules {
    pub(super) spec: EmbeddingSemantics,
    pub(super) auxiliary: Vec<AuxiliaryVarBuilder>,
}

pub(super) fn load_embedding_modules_from_dir(
    model_path: &Path,
    arch_name: &str,
    task: TaskKind,
    dtype: DType,
    device: &Device,
) -> Result<Option<LoadedEmbeddingModules>, EngineError> {
    if task != TaskKind::Embed {
        return Ok(None);
    }

    let modules_path = model_path.join("modules.json");
    if !modules_path.exists() {
        tracing::warn!(
            path = %modules_path.display(),
            "embedding modules.json not found; using default last-token semantics"
        );
        return Ok(None);
    }

    load_embedding_modules_from_file(
        &modules_path,
        |relative| Ok(model_path.join(relative)),
        dtype,
        device,
        arch_name == "gemma3",
    )
}

pub(super) async fn load_embedding_modules_from_repo(
    repo: &hf_hub::api::tokio::ApiRepo,
    arch_name: &str,
    task: TaskKind,
    dtype: DType,
    device: &Device,
) -> Result<Option<LoadedEmbeddingModules>, EngineError> {
    if task != TaskKind::Embed {
        return Ok(None);
    }

    let modules_path = match repo.get("modules.json").await {
        Ok(path) => path,
        Err(err) => {
            tracing::warn!(error = %err, "failed to resolve embedding modules.json");
            return Ok(None);
        }
    };

    // Walk modules.json once to enumerate every relative path the sync
    // loader will ask for, await-download them all up front, then hand
    // the sync loader a closure that just looks each path up in the
    // resulting map.
    //
    // We can't `block_in_place` + `Handle::current().block_on` from
    // inside the sync closure: tokio panics that combo on a
    // current_thread runtime, which `#[tokio::test]` defaults to —
    // meaning embed-loading would crash any test that builds an Engine.
    let load_dense_auxiliary = arch_name == "gemma3";
    let needed = enumerate_embedding_module_files(&modules_path, load_dense_auxiliary)?;
    let mut path_map: std::collections::HashMap<String, PathBuf> =
        std::collections::HashMap::with_capacity(needed.len());
    for relative in needed {
        let local = repo.get(&relative).await.map_err(|err| {
            EngineError::Internal(format!("failed to download {relative}: {err}"))
        })?;
        path_map.insert(relative, local);
    }

    load_embedding_modules_from_file(
        &modules_path,
        |relative| {
            path_map.get(relative).cloned().ok_or_else(|| {
                EngineError::Internal(format!(
                    "embedding module file not pre-downloaded: {relative} \
                     (enumerate_embedding_module_files missed it)"
                ))
            })
        },
        dtype,
        device,
        load_dense_auxiliary,
    )
}

/// Read `modules.json` and return every relative file path the sync
/// loader will subsequently ask for via its `resolve_path` callback.
/// Mirrors the matching arms in `load_embedding_modules_from_file` —
/// keep the two in sync.
fn enumerate_embedding_module_files(
    modules_path: &Path,
    load_dense_auxiliary: bool,
) -> Result<Vec<String>, EngineError> {
    let content = std::fs::read_to_string(modules_path).map_err(|err| {
        EngineError::Internal(format!(
            "failed to read embedding modules.json {}: {err}",
            modules_path.display()
        ))
    })?;
    let modules: Vec<SentenceTransformerModuleEntry> =
        serde_json::from_str(&content).map_err(|err| {
            EngineError::Internal(format!(
                "failed to parse embedding modules.json {}: {err}",
                modules_path.display()
            ))
        })?;

    let mut needed = Vec::new();
    for entry in &modules {
        if entry.module_type.ends_with(".Pooling") {
            needed.push(module_relative_path(&entry.path, "config.json"));
        } else if entry.module_type.ends_with(".Dense") {
            needed.push(module_relative_path(&entry.path, "config.json"));
            if load_dense_auxiliary {
                needed.push(module_relative_path(&entry.path, "model.safetensors"));
            }
        }
    }
    Ok(needed)
}

fn load_embedding_modules_from_file<F>(
    modules_path: &Path,
    mut resolve_path: F,
    dtype: DType,
    device: &Device,
    load_dense_auxiliary: bool,
) -> Result<Option<LoadedEmbeddingModules>, EngineError>
where
    F: FnMut(&str) -> Result<PathBuf, EngineError>,
{
    let content = std::fs::read_to_string(modules_path).map_err(|err| {
        EngineError::Internal(format!(
            "failed to read embedding modules.json {}: {err}",
            modules_path.display()
        ))
    })?;

    let mut modules: Vec<SentenceTransformerModuleEntry> =
        serde_json::from_str(&content).map_err(|err| {
            EngineError::Internal(format!(
                "failed to parse embedding modules.json {}: {err}",
                modules_path.display()
            ))
        })?;
    modules.sort_by_key(|entry| entry.idx);

    let mut spec = EmbeddingSemantics::default();
    let mut auxiliary = Vec::new();

    for entry in modules {
        let is_normalize = entry.module_type.ends_with(".Normalize");
        if spec.normalization == EmbeddingNormalization::L2 && !is_normalize {
            return Err(EngineError::InvalidRequest(format!(
                "unsupported sentence-transformers module order in {}: Normalize must be the final module",
                modules_path.display()
            )));
        }

        if entry.module_type.ends_with(".Transformer") {
            continue;
        }

        if entry.module_type.ends_with(".Pooling") {
            let config_path = resolve_path(&module_relative_path(&entry.path, "config.json"))?;
            let pooling_cfg: SentenceTransformerPoolingConfig =
                read_json_file(&config_path, "sentence-transformers pooling config")?;
            spec.pooling = pooling_from_config(&pooling_cfg, &config_path)?;
            continue;
        }

        if entry.module_type.ends_with(".Dense") {
            let config_path = resolve_path(&module_relative_path(&entry.path, "config.json"))?;
            let dense_cfg: SentenceTransformerDenseConfig =
                read_json_file(&config_path, "sentence-transformers dense config")?;
            let activation = resolve_dense_activation(&dense_cfg)?;
            let bias = if load_dense_auxiliary {
                let weight_path =
                    resolve_path(&module_relative_path(&entry.path, "model.safetensors"))?;
                let vb = load_var_builder_from_filenames(&[weight_path.clone()], dtype, device)?;
                auxiliary.push(AuxiliaryVarBuilder {
                    module_path: entry.path.clone(),
                    vb,
                });
                dense_bias_from_config_or_weights(dense_cfg.bias, &weight_path)?
            } else {
                dense_cfg.bias.unwrap_or(false)
            };

            spec.dense_layers.push(EmbeddingDenseLayerSpec {
                module_path: entry.path.clone(),
                in_features: dense_cfg.in_features,
                out_features: dense_cfg.out_features,
                bias,
                activation,
            });
            continue;
        }

        if entry.module_type.ends_with(".Normalize") {
            if spec.normalization == EmbeddingNormalization::L2 {
                return Err(EngineError::InvalidRequest(format!(
                    "unsupported sentence-transformers module order in {}: multiple Normalize modules are not supported",
                    modules_path.display()
                )));
            }
            spec.normalization = EmbeddingNormalization::L2;
            continue;
        }

        return Err(EngineError::InvalidRequest(format!(
            "unsupported sentence-transformers module type for embeddings: {}",
            entry.module_type
        )));
    }

    Ok(Some(LoadedEmbeddingModules { spec, auxiliary }))
}

fn module_relative_path(module_path: &str, filename: &str) -> String {
    if module_path.is_empty() {
        filename.to_string()
    } else {
        format!("{module_path}/{filename}")
    }
}

fn dense_bias_from_config_or_weights(
    config_bias: Option<bool>,
    weight_path: &Path,
) -> Result<bool, EngineError> {
    if let Some(bias) = config_bias {
        return Ok(bias);
    }

    // Some sentence-transformers dense configs omit `bias`; use the safetensor
    // payload as the source of truth so we don't silently drop a present bias.
    let weights = unsafe { crate::tensor::safetensors::MmapedSafetensors::new(weight_path) }
        .map_err(tensor_err)?;
    Ok(weights
        .tensors()
        .into_iter()
        .any(|(name, _)| name == "linear.bias"))
}

fn pooling_from_config(
    cfg: &SentenceTransformerPoolingConfig,
    path: &Path,
) -> Result<EmbeddingPooling, EngineError> {
    let supported = [
        (cfg.pooling_mode_lasttoken, EmbeddingPooling::LastToken),
        (cfg.pooling_mode_mean_tokens, EmbeddingPooling::Mean),
        (cfg.pooling_mode_cls_token, EmbeddingPooling::Cls),
    ];
    let enabled: Vec<_> = supported
        .into_iter()
        .filter_map(|(enabled, pooling): (bool, EmbeddingPooling)| enabled.then_some(pooling))
        .collect();

    if enabled.len() == 1
        && !cfg.pooling_mode_max_tokens
        && !cfg.pooling_mode_mean_sqrt_len_tokens
        && !cfg.pooling_mode_weightedmean_tokens
    {
        return Ok(enabled[0]);
    }

    Err(EngineError::InvalidRequest(format!(
        "unsupported sentence-transformers pooling config {}",
        path.display()
    )))
}

fn parse_dense_activation(value: Option<&str>) -> Result<EmbeddingActivation, EngineError> {
    let normalized = value.unwrap_or("").trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "torch.nn.modules.linear.identity" | "torch.nn.identity" | "identity" => {
            Ok(EmbeddingActivation::Identity)
        }
        _ => Err(EngineError::InvalidRequest(format!(
            "unsupported sentence-transformers dense activation: {}",
            value.unwrap_or("")
        ))),
    }
}

fn resolve_dense_activation(
    cfg: &SentenceTransformerDenseConfig,
) -> Result<EmbeddingActivation, EngineError> {
    let activation_function = cfg.activation_function.as_deref();
    let activation = cfg.activation.as_deref();

    if let (Some(lhs), Some(rhs)) = (activation_function, activation) {
        if !lhs.trim().eq_ignore_ascii_case(rhs.trim()) {
            return Err(EngineError::InvalidRequest(format!(
                "conflicting sentence-transformers dense activation values: activation_function={lhs}, activation={rhs}"
            )));
        }
    }

    parse_dense_activation(activation_function.or(activation))
}

fn read_json_file<T: serde::de::DeserializeOwned>(
    path: &Path,
    description: &str,
) -> Result<T, EngineError> {
    let content = std::fs::read_to_string(path).map_err(|err| {
        EngineError::Internal(format!(
            "failed to read {description} {}: {err}",
            path.display()
        ))
    })?;
    serde_json::from_str(&content).map_err(|err| {
        EngineError::Internal(format!(
            "failed to parse {description} {}: {err}",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::safetensors::save as save_safetensors;
    use crate::tensor::{Device, Tensor};
    use serde_json::json;
    use std::collections::HashMap;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    const TEST_SENTENCE_TRANSFORMER_MODEL_DIM: usize = 768;
    const TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM: usize = 3072;
    const TEST_SENTENCE_TRANSFORMER_DENSE_ACTIVATION: &str = "torch.nn.Identity";

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> Self {
            let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("prelude-{name}-{}-{}", std::process::id(), id));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn write_json(path: &Path, value: &serde_json::Value) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    }

    fn write_dense_weights_with_bias(
        path: &Path,
        in_features: usize,
        out_features: usize,
        include_bias: bool,
    ) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let weight = Tensor::from_vec(
            vec![0f32; in_features * out_features],
            (out_features, in_features),
            &Device::Cpu,
        )
        .unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("linear.weight".to_string(), weight);
        if include_bias {
            let bias =
                Tensor::from_vec(vec![0f32; out_features], out_features, &Device::Cpu).unwrap();
            tensors.insert("linear.bias".to_string(), bias);
        }
        save_safetensors(&tensors, path).unwrap();
    }

    fn write_dense_weights(path: &Path, in_features: usize, out_features: usize) {
        write_dense_weights_with_bias(path, in_features, out_features, false);
    }

    fn sentence_transformers_pooling_config() -> serde_json::Value {
        json!({
            "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": true,
            "pooling_mode_lasttoken": false,
            "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false,
            "pooling_mode_weightedmean_tokens": false
        })
    }

    fn sentence_transformers_dense_config(
        in_features: usize,
        out_features: usize,
        bias: Option<bool>,
        activation: Option<&str>,
    ) -> serde_json::Value {
        let mut config = json!({
            "in_features": in_features,
            "out_features": out_features
        });
        if let Some(bias) = bias {
            config["bias"] = json!(bias);
        }
        if let Some(activation) = activation {
            config["activation_function"] = json!(activation);
        }
        config
    }

    fn sentence_transformers_dense_config_with_activation_alias(
        in_features: usize,
        out_features: usize,
        activation: &str,
    ) -> serde_json::Value {
        json!({
            "in_features": in_features,
            "out_features": out_features,
            "activation": activation
        })
    }

    #[test]
    fn parses_sentence_transformers_embedding_modules() {
        let dir = TestDir::new("sentence-transformers-modules");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling"
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Dense",
                    "type": "sentence_transformers.models.Dense"
                },
                {
                    "idx": 3,
                    "name": "3",
                    "path": "3_Dense",
                    "type": "sentence_transformers.models.Dense"
                },
                {
                    "idx": 4,
                    "name": "4",
                    "path": "4_Normalize",
                    "type": "sentence_transformers.models.Normalize"
                }
            ]),
        );

        write_json(
            &dir.path.join("1_Pooling/config.json"),
            &sentence_transformers_pooling_config(),
        );
        write_json(
            &dir.path.join("2_Dense/config.json"),
            &sentence_transformers_dense_config(
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                Some(false),
                Some(TEST_SENTENCE_TRANSFORMER_DENSE_ACTIVATION),
            ),
        );
        write_json(
            &dir.path.join("3_Dense/config.json"),
            &sentence_transformers_dense_config(
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                Some(false),
                None,
            ),
        );
        write_dense_weights(
            &dir.path.join("2_Dense/model.safetensors"),
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
        );
        write_dense_weights(
            &dir.path.join("3_Dense/model.safetensors"),
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
        );

        let loaded = load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        )
        .unwrap()
        .unwrap();

        assert_eq!(loaded.spec.pooling, EmbeddingPooling::Mean);
        assert_eq!(loaded.spec.normalization, EmbeddingNormalization::L2);
        assert_eq!(loaded.spec.dense_layers.len(), 2);
        assert_eq!(loaded.spec.dense_layers[0].module_path, "2_Dense");
        assert_eq!(
            loaded.spec.dense_layers[0].in_features,
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM
        );
        assert_eq!(
            loaded.spec.dense_layers[0].out_features,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM
        );
        assert_eq!(
            loaded.spec.dense_layers[0].activation,
            EmbeddingActivation::Identity
        );
        assert_eq!(loaded.spec.dense_layers[1].module_path, "3_Dense");
        assert_eq!(
            loaded.spec.dense_layers[1].in_features,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM
        );
        assert_eq!(
            loaded.spec.dense_layers[1].out_features,
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM
        );
        assert_eq!(loaded.auxiliary.len(), 2);
        assert_eq!(loaded.auxiliary[0].module_path, "2_Dense");
        assert_eq!(loaded.auxiliary[1].module_path, "3_Dense");
    }

    #[test]
    fn infers_embedding_dense_bias_from_weights_when_config_omits_flag() {
        let dir = TestDir::new("sentence-transformers-dense-bias-inference");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense"
                }
            ]),
        );
        write_json(
            &dir.path.join("1_Dense/config.json"),
            &sentence_transformers_dense_config(
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                None,
                None,
            ),
        );
        write_dense_weights_with_bias(
            &dir.path.join("1_Dense/model.safetensors"),
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
            true,
        );

        let loaded = load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        )
        .unwrap()
        .unwrap();

        assert_eq!(loaded.spec.dense_layers.len(), 1);
        assert!(loaded.spec.dense_layers[0].bias);
    }

    #[test]
    fn parses_embedding_dense_activation_alias_key() {
        let dir = TestDir::new("sentence-transformers-dense-activation-alias");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense"
                }
            ]),
        );
        write_json(
            &dir.path.join("1_Dense/config.json"),
            &sentence_transformers_dense_config_with_activation_alias(
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                TEST_SENTENCE_TRANSFORMER_DENSE_ACTIVATION,
            ),
        );
        write_dense_weights(
            &dir.path.join("1_Dense/model.safetensors"),
            TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
            TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
        );

        let loaded = load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        )
        .unwrap()
        .unwrap();

        assert_eq!(loaded.spec.dense_layers.len(), 1);
        assert_eq!(
            loaded.spec.dense_layers[0].activation,
            EmbeddingActivation::Identity
        );
    }

    #[test]
    fn rejects_non_terminal_sentence_transformers_normalize_module() {
        let dir = TestDir::new("sentence-transformers-normalize-non-terminal");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Normalize",
                    "type": "sentence_transformers.models.Normalize"
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Dense",
                    "type": "sentence_transformers.models.Dense"
                }
            ]),
        );

        let err = match load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        ) {
            Ok(_) => panic!("expected non-terminal Normalize module order to be rejected"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("Normalize must be the final module"),
            "{err}"
        );
    }

    #[test]
    fn rejects_multiple_sentence_transformers_normalize_modules() {
        let dir = TestDir::new("sentence-transformers-normalize-duplicate");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Normalize",
                    "type": "sentence_transformers.models.Normalize"
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Normalize",
                    "type": "sentence_transformers.models.Normalize"
                }
            ]),
        );

        let err = match load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            true,
        ) {
            Ok(_) => panic!("expected duplicate Normalize modules to be rejected"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("multiple Normalize modules are not supported"),
            "{err}"
        );
    }

    #[test]
    fn skips_dense_weight_loading_for_non_gemma_embeddings() {
        let dir = TestDir::new("sentence-transformers-dense-metadata-only");
        let modules_path = dir.path.join("modules.json");

        write_json(
            &modules_path,
            &json!([
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer"
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Dense",
                    "type": "sentence_transformers.models.Dense"
                }
            ]),
        );
        write_json(
            &dir.path.join("1_Dense/config.json"),
            &sentence_transformers_dense_config(
                TEST_SENTENCE_TRANSFORMER_MODEL_DIM,
                TEST_SENTENCE_TRANSFORMER_EXPANDED_DIM,
                Some(true),
                Some(TEST_SENTENCE_TRANSFORMER_DENSE_ACTIVATION),
            ),
        );

        let loaded = load_embedding_modules_from_file(
            &modules_path,
            |relative| Ok(dir.path.join(relative)),
            DType::F32,
            &Device::Cpu,
            false,
        )
        .unwrap()
        .unwrap();

        assert_eq!(loaded.spec.dense_layers.len(), 1);
        assert!(loaded.spec.dense_layers[0].bias);
        assert!(loaded.auxiliary.is_empty());
    }
}

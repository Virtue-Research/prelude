mod forward;
pub mod gemma3;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_moe;
pub mod qwen3_next;
pub(crate) mod registry;



pub(crate) use forward::{
    ClassifierModel, EmbeddingModel, KvCacheModel, LogitsSplitModel, ModelForward,
};

/// Resolve an optional config field, warning if falling back to a default value.
///
/// Use this when deserializing model config.json — fields should be loaded from
/// the JSON, but if missing we fall back with a visible warning so operators know
/// the model may be using incorrect defaults.
macro_rules! resolve_or_warn {
    ($opt:expr, $default:expr, $field:literal, $model:expr) => {
        match $opt {
            Some(v) => v,
            None => {
                tracing::warn!(
                    "{}: '{}' not found in config.json, using default: {:?}",
                    $model,
                    $field,
                    $default
                );
                $default
            }
        }
    };
    // Also accept stringify!() output (used by model_config! macro).
    ($opt:expr, $default:expr, $field:expr, $model:expr) => {
        match $opt {
            Some(v) => v,
            None => {
                tracing::warn!(
                    "{}: '{}' not found in config.json, using default: {:?}",
                    $model,
                    $field,
                    $default
                );
                $default
            }
        }
    };
}
pub(crate) use resolve_or_warn;

/// Generate a model config struct with a forgiving `Deserialize` impl.
///
/// Three field categories:
///   - **`required`** — must exist in JSON, deserialization fails if missing.
///   - **`serde_default`** — uses serde `#[serde(default)]` (0 / false / None).
///   - **`warn_default`** — `Option<T>` in the raw parse; resolved via `resolve_or_warn!`.
///
/// ```ignore
/// model_config! {
///     pub struct FooConfig("Foo") {
///         required {
///             hidden_size: usize,
///         }
///         serde_default {
///             attention_bias: bool,
///         }
///         warn_default {
///             rms_norm_eps: f64 = 1e-6,
///         }
///     }
/// }
/// ```
macro_rules! model_config {
    (
        $(#[$outer:meta])*
        pub struct $Config:ident ($model_name:literal) {
            required {
                $( $req_field:ident : $req_ty:ty ),* $(,)?
            }
            $( serde_default {
                $( $sd_field:ident : $sd_ty:ty ),* $(,)?
            } )?
            $( warn_default {
                $( $wd_field:ident : $wd_ty:ty = $wd_default:expr ),* $(,)?
            } )?
        }
    ) => {
        $(#[$outer])*
        #[derive(Debug, Clone)]
        pub struct $Config {
            $( pub $req_field : $req_ty, )*
            $( $( pub $sd_field : $sd_ty, )* )?
            $( $( pub $wd_field : $wd_ty, )* )?
        }

        impl<'de> serde::Deserialize<'de> for $Config {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                #[derive(serde::Deserialize)]
                struct Raw {
                    $( $req_field : $req_ty, )*
                    $( $(
                        #[serde(default)]
                        $sd_field : $sd_ty,
                    )* )?
                    $( $(
                        #[serde(default)]
                        $wd_field : Option<$wd_ty>,
                    )* )?
                }

                let r = Raw::deserialize(deserializer)?;
                const MODEL: &str = $model_name;
                Ok($Config {
                    $( $req_field : r.$req_field, )*
                    $( $( $sd_field : r.$sd_field, )* )?
                    $( $( $wd_field : $crate::models::resolve_or_warn!(
                        r.$wd_field, $wd_default, stringify!($wd_field), MODEL
                    ), )* )?
                })
            }
        }
    };
}
pub(crate) use model_config;

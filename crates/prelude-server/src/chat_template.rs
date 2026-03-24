use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Local;
use hf_hub::api::sync::{Api, ApiRepo};
use minijinja::{context, Environment, Error, ErrorKind};
use minijinja_contrib::pycompat::unknown_method_callback;
use prelude_core::ChatMessage;
use serde::Deserialize;
use serde_json::json;

#[derive(Clone, Debug)]
pub struct ModelChatTemplate {
    template: String,
    bos_token: String,
    eos_token: String,
    pad_token: String,
    unk_token: String,
}

impl ModelChatTemplate {
    /// Create a chat template from a raw Jinja template string (for testing).
    pub fn from_template_string(template: String) -> Self {
        Self {
            template,
            bos_token: String::new(),
            eos_token: String::new(),
            pad_token: String::new(),
            unk_token: String::new(),
        }
    }

    pub fn from_local_path(model_path: impl AsRef<Path>) -> Result<Option<Self>> {
        let model_dir = resolve_model_dir(model_path.as_ref());
        let template_file = read_optional_file(&model_dir.join("chat_template.jinja"))?;
        let tokenizer_config = load_tokenizer_config(read_optional_file(
            &model_dir.join("tokenizer_config.json"),
        )?)?;
        Self::from_sources(template_file, tokenizer_config)
    }

    pub fn from_hf_hub(repo_id: &str) -> Result<Option<Self>> {
        let api = Api::new().context("failed to initialize hf-hub api")?;
        let repo = api.model(repo_id.to_string());
        let template_file = read_optional_repo_file(&repo, "chat_template.jinja")?;
        let tokenizer_config =
            load_tokenizer_config(read_optional_repo_file(&repo, "tokenizer_config.json")?)?;
        Self::from_sources(template_file, tokenizer_config)
    }

    pub fn render(&self, messages: &[ChatMessage]) -> Result<String> {
        let serialized_messages: Vec<_> = messages
            .iter()
            .map(|message| {
                let content = message.text_content().unwrap_or_default();
                json!({
                    "role": message.role.as_str(),
                    "content": content,
                    "tool_calls": serde_json::Value::Null,
                    "reasoning_content": serde_json::Value::Null,
                    "tool_call_id": serde_json::Value::Null,
                    "name": serde_json::Value::Null,
                })
            })
            .collect();

        let env = build_environment();
        let template = env
            .template_from_str(&self.template)
            .context("failed to compile chat template")?;
        template
            .render(context! {
                messages => serialized_messages,
                add_generation_prompt => true,
                bos_token => self.bos_token.as_str(),
                eos_token => self.eos_token.as_str(),
                pad_token => self.pad_token.as_str(),
                unk_token => self.unk_token.as_str(),
                tools => Option::<Vec<serde_json::Value>>::None,
                documents => Option::<Vec<serde_json::Value>>::None,
            })
            .context("failed to render chat template")
    }

    fn from_sources(
        template_file: Option<String>,
        tokenizer_config: Option<TokenizerConfig>,
    ) -> Result<Option<Self>> {
        let template = match template_file {
            Some(template) => Some(template),
            None => tokenizer_config
                .as_ref()
                .map(TokenizerConfig::resolve_chat_template)
                .transpose()?
                .flatten(),
        };

        let Some(template) = template else {
            return Ok(None);
        };

        let tokenizer_config = tokenizer_config.unwrap_or_default();
        Ok(Some(Self {
            template,
            bos_token: tokenizer_config.bos_token(),
            eos_token: tokenizer_config.eos_token(),
            pad_token: tokenizer_config.pad_token(),
            unk_token: tokenizer_config.unk_token(),
        }))
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    chat_template: Option<ChatTemplateField>,
    #[serde(default)]
    bos_token: Option<SpecialTokenField>,
    #[serde(default)]
    eos_token: Option<SpecialTokenField>,
    #[serde(default)]
    pad_token: Option<SpecialTokenField>,
    #[serde(default)]
    unk_token: Option<SpecialTokenField>,
}

impl TokenizerConfig {
    fn resolve_chat_template(&self) -> Result<Option<String>> {
        let Some(chat_template) = &self.chat_template else {
            return Ok(None);
        };

        match chat_template {
            ChatTemplateField::Single(template) => Ok(Some(template.clone())),
            ChatTemplateField::Named(templates) => {
                if let Some(template) = templates.get("default") {
                    return Ok(Some(template.clone()));
                }
                if templates.len() == 1 {
                    return Ok(templates.values().next().cloned());
                }
                anyhow::bail!(
                    "multiple named chat templates found but no `default` template is defined"
                );
            }
        }
    }

    fn bos_token(&self) -> String {
        self.bos_token
            .as_ref()
            .map(SpecialTokenField::content)
            .unwrap_or_default()
            .to_string()
    }

    fn eos_token(&self) -> String {
        self.eos_token
            .as_ref()
            .map(SpecialTokenField::content)
            .unwrap_or_default()
            .to_string()
    }

    fn pad_token(&self) -> String {
        self.pad_token
            .as_ref()
            .map(SpecialTokenField::content)
            .unwrap_or_default()
            .to_string()
    }

    fn unk_token(&self) -> String {
        self.unk_token
            .as_ref()
            .map(SpecialTokenField::content)
            .unwrap_or_default()
            .to_string()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum ChatTemplateField {
    Single(String),
    Named(BTreeMap<String, String>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum SpecialTokenField {
    Plain(String),
    AddedToken { content: String },
}

impl SpecialTokenField {
    fn content(&self) -> &str {
        match self {
            Self::Plain(value) => value,
            Self::AddedToken { content } => content,
        }
    }
}

fn build_environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);
    env.set_unknown_method_callback(unknown_method_callback);
    env
}

fn raise_exception(message: String) -> Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

fn strftime_now(format: String) -> String {
    Local::now().format(&format).to_string()
}

fn resolve_model_dir(path: &Path) -> PathBuf {
    if path.extension().is_some_and(|ext| ext == "gguf") {
        path.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.to_path_buf())
    } else {
        path.to_path_buf()
    }
}

fn load_tokenizer_config(contents: Option<String>) -> Result<Option<TokenizerConfig>> {
    contents
        .map(|content| {
            serde_json::from_str(&content).context("failed to parse tokenizer_config.json")
        })
        .transpose()
}

fn read_optional_file(path: &Path) -> Result<Option<String>> {
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(fs::read_to_string(path).with_context(|| {
        format!("failed to read {}", path.display())
    })?))
}

fn read_optional_repo_file(repo: &ApiRepo, filename: &str) -> Result<Option<String>> {
    let Ok(path) = repo.get(filename) else {
        return Ok(None);
    };
    Ok(Some(fs::read_to_string(&path).with_context(|| {
        format!("failed to read downloaded {filename}")
    })?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude_core::{ChatMessage, ChatMessageContent};

    #[test]
    fn resolves_default_named_chat_template() {
        let config: TokenizerConfig = serde_json::from_str(
            r#"{
                "chat_template": {
                    "default": "DEFAULT_TEMPLATE",
                    "tool_use": "TOOL_TEMPLATE"
                }
            }"#,
        )
        .unwrap();
        assert_eq!(
            config.resolve_chat_template().unwrap().as_deref(),
            Some("DEFAULT_TEMPLATE")
        );
    }

    #[test]
    fn renders_qwen_style_template_from_tokenizer_config() {
        let config: TokenizerConfig = serde_json::from_str(
            r#"{
                "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
                "eos_token": "<|im_end|>"
            }"#,
        )
        .unwrap();
        let template = ModelChatTemplate::from_sources(None, Some(config))
            .unwrap()
            .unwrap();
        let rendered = template
            .render(&[
                ChatMessage {
                    role: "system".to_string(),
                    content: ChatMessageContent::Text("You are helpful".to_string()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: ChatMessageContent::Text("Hello".to_string()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
            ])
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    #[ignore = "debug helper for real HF tokenizer templates"]
    fn renders_real_qwen3_template_smoke() {
        let template = ModelChatTemplate::from_hf_hub("Qwen/Qwen3-0.6B")
            .unwrap()
            .expect("Qwen3 template should exist");
        let rendered = template.render(&[
            ChatMessage {
                role: "system".to_string(),
                content: ChatMessageContent::Text("You are a helpful assistant.".to_string()),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: ChatMessageContent::Text("Say hello".to_string()),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            },
        ]);
        match rendered {
            Ok(text) => {
                assert!(text.contains("<|im_start|>assistant"));
            }
            Err(err) => panic!("{err:#}"),
        }
    }
}

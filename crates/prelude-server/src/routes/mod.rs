mod chat_completions;
mod classify;
mod completions;
mod embeddings;
mod generation_common;
mod health;
mod models;

pub use chat_completions::chat_completions;
pub use classify::classify;
pub use completions::completions;
pub use embeddings::embeddings;
pub use health::health;
pub use models::{get_model, list_models};

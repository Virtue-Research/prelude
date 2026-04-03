//! Sampling orchestration.
//!
//! Contains the `LogitsProcessor` for token sampling from raw logits.
//! Future: `GrammarManager` for constrained decoding, penalty processors.

mod grammar;
mod logits_processor;
pub use logits_processor::{LogitsProcessor, Sampling};

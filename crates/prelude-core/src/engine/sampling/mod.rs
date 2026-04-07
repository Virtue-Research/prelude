//! Sampling orchestration.
//!
//! Contains the `LogitsProcessor` for token sampling from raw logits,
//! and `GrammarBackend`/`GrammarMatcher` traits for constrained decoding.

pub mod grammar;
mod logits_processor;
pub use logits_processor::{LogitsProcessor, Sampling};
pub use grammar::{ConstraintSpec, GrammarBackend, GrammarMatcher};

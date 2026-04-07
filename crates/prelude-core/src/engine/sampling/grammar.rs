//! Grammar-constrained decoding.
//!
//! Traits for pluggable grammar engines that constrain token sampling via bitmasks.
//! The default backend will be llguidance (pure Rust, MIT, Earley parser + derivative regex).
//!
//! Flow:
//!   1. `GrammarBackend::compile(spec)` → `Box<dyn GrammarMatcher>` (async, on thread pool)
//!   2. `GrammarMatcher::fill_bitmask(bitmask, idx)` — after model.forward(), before sampling
//!   3. `GrammarMatcher::accept_token(id)` — after sampling, advance FSM
//!   4. `GrammarMatcher::rollback(k)` — on speculative decoding rejection

use crate::tensor::{Result, Tensor};

/// What the user specifies in the request.
#[derive(Debug, Clone)]
pub enum ConstraintSpec {
    /// JSON schema (string or parsed). Most common for API serving.
    JsonSchema(String),
    /// Regex pattern. For structured fields (phone, email, etc.).
    Regex(String),
    /// EBNF grammar. For programming languages, custom formats.
    Grammar(String),
    /// Fixed choice list. Simplest constraint.
    Choice(Vec<String>),
}

/// Pluggable grammar engine. Compiles constraints into token-level matchers.
pub trait GrammarBackend: Send + Sync {
    /// Compile a constraint specification into a grammar matcher.
    /// Can be expensive (10-100ms for complex JSON schemas).
    fn compile(&self, spec: &ConstraintSpec) -> Result<Box<dyn GrammarMatcher>>;

    /// Allocate a bitmask tensor for batch_size requests.
    /// Shape: [batch_size, ceil(vocab_size / 32)] as u32 (packed bits).
    fn allocate_bitmask(&self, batch_size: usize, vocab_size: usize) -> Tensor;
}

/// Per-request grammar state. Tracks FSM position, fills bitmask.
pub trait GrammarMatcher: Send {
    /// Fill bitmask row at `batch_index` with allowed next tokens.
    fn fill_bitmask(&self, bitmask: &mut Tensor, batch_index: usize);

    /// Accept token and advance FSM state. Returns false if token is invalid.
    fn accept_token(&mut self, token_id: u32) -> bool;

    /// Rollback FSM state by k tokens (for speculative decoding rejection).
    fn rollback(&mut self, k: usize);

    /// True if grammar has reached an accept state (output is complete).
    fn is_terminated(&self) -> bool;

    /// Reset to initial state (for reuse across requests).
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constraint_spec_variants() {
        let specs = vec![
            ConstraintSpec::JsonSchema(r#"{"type":"object"}"#.into()),
            ConstraintSpec::Regex(r"^\d{3}-\d{4}$".into()),
            ConstraintSpec::Grammar("expr ::= term (('+' | '-') term)*".into()),
            ConstraintSpec::Choice(vec!["yes".into(), "no".into()]),
        ];
        assert_eq!(specs.len(), 4);
        assert!(matches!(&specs[0], ConstraintSpec::JsonSchema(_)));
        assert!(matches!(&specs[3], ConstraintSpec::Choice(v) if v.len() == 2));
    }

    /// Verify that a trivial GrammarMatcher impl works with the trait interface.
    struct AllowAll;
    impl GrammarMatcher for AllowAll {
        fn fill_bitmask(&self, _bitmask: &mut Tensor, _batch_index: usize) {}
        fn accept_token(&mut self, _token_id: u32) -> bool { true }
        fn rollback(&mut self, _k: usize) {}
        fn is_terminated(&self) -> bool { false }
        fn reset(&mut self) {}
    }

    #[test]
    fn allow_all_matcher() {
        let mut m = AllowAll;
        assert!(m.accept_token(42));
        assert!(!m.is_terminated());
        m.rollback(3);
        m.reset();
        assert!(m.accept_token(0));
    }
}

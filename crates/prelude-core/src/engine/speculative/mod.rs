//! Speculative decoding — draft → verify → accept loop.
//!
//! `SpecDecodeRunner` orchestrates speculative decoding: a small draft model
//! proposes K tokens, then the target model verifies them in a single forward
//! pass. Accepted tokens skip full decode steps, improving throughput.
//!
//! Supports multiple proposer strategies:
//! - EAGLE (hidden-state draft model)
//! - DraftModel (separate smaller LLM)
//! - Ngram (simple n-gram cache lookup)
//! - Medusa (multi-head parallel token proposal)
//!
//! Not yet implemented — placeholder for speculative decoding support.

mod proposer;
mod rejection;
mod tree;

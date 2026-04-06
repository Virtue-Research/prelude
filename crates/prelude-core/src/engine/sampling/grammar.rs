//! Grammar-constrained decoding.
//!
//! `GrammarManager` handles async compilation of grammar specifications
//! (JSON schema, regex, CFG) into bitmasks that constrain the sampling
//! distribution at each decode step.
//!
//! Backed by llguidance (pure Rust, MIT license, Earley parser + derivative regex).
//! ~50μs per token for 128k tokenizers, zero startup overhead.
//!
//! Not yet implemented — placeholder for constrained decoding support.

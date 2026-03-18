//! Global constants for Prelude configuration.

/// Default random seed for sampling when not specified in the request.
pub const DEFAULT_SEED: u64 = 42;

// ---------------------------------------------------------------------------
// GGUF metadata fallback defaults
// ---------------------------------------------------------------------------

/// Default vocabulary size when `token_embd.weight` tensor is missing from GGUF.
/// 151936 is the Qwen3 tokenizer vocabulary size.
pub const GGUF_DEFAULT_VOCAB_SIZE: usize = 151936;

/// Default intermediate-size multiplier when `feed_forward_length` is missing.
pub const GGUF_INTERMEDIATE_SIZE_MULTIPLIER: usize = 3;

// ---------------------------------------------------------------------------
// PseudoEngine (mock) limits
// ---------------------------------------------------------------------------

/// Maximum tokens the mock PseudoEngine will generate per request.
pub const PSEUDO_ENGINE_MAX_TOKENS: u32 = 256;

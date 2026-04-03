//! Grammar-constrained decoding.
//!
//! `GrammarManager` handles async compilation of grammar specifications
//! (JSON schema, regex, CFG) into bitmasks that constrain the sampling
//! distribution at each decode step.
//!
//! Backed by the `prelude-xgrammar` plugin crate (C++ FFI).
//!
//! Not yet implemented — placeholder for constrained decoding support.

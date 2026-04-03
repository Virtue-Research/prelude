//! Scheduling-paradigm loops (device-agnostic).
//!
//! Each loop calls `Executor::submit/collect` — no device-specific code here.
//! Device crates implement `Executor` with their own execution strategy.

pub mod ar;
mod dllm;
mod diffusion;
mod tts;

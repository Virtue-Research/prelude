//! Scheduling decisions (pure CPU, no GPU).
//!
//! Per-paradigm schedulers decide what to run each step:
//! - AR (autoregressive LLM): continuous batching with prefill/decode separation
//! - DLLM (diffusion LLM): iterative demasking
//! - Diffusion (image/video): denoising loop
//! - TTS: multi-stage pipeline
//! - OneShot: embed, classify, prefill-only
//!
//! The current implementation covers the AR scheduler. Other paradigms are stubs.
//!
//! AR scheduler inspired by SGLang:
//! - Scheduler: <https://github.com/sgl-project/sglang/blob/78ddf05a/python/sglang/srt/managers/scheduler.py>
//! - Schedule policy: <https://github.com/sgl-project/sglang/blob/78ddf05a/python/sglang/srt/managers/schedule_policy.py>
//! SGLang is licensed under the Apache License, Version 2.0.

// ── AR scheduler (current implementation) ──
mod admission;
mod preemption;
mod state;
pub(crate) mod adaptive;

// ── Paradigm scheduler stubs ──
mod dllm;
mod diffusion;
mod tts;
mod oneshot;

pub mod components;

#[cfg(test)]
mod tests;

pub use components::cache::block_manager::BlockManager;
pub use components::cache::prefix_index::{PrefixInsertPlan, PrefixMatch, PrefixMatchIndex};
pub use state::{
    FinishReason, ForwardMode, SamplingParams, SchedulePolicy, Scheduler, SchedulerConfig,
    SchedulerOutput, SchedulerStep, SeqFinishReason, Sequence, SequenceStatus,
};

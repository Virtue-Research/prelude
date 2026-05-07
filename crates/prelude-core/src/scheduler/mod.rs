//! Scheduling decisions (pure CPU, no GPU).
//!
//! Per-paradigm schedulers decide what to run each step:
//! - AR (autoregressive LLM): continuous batching with prefill/decode separation
//!
//! The current implementation covers the AR scheduler. One-shot tasks
//! (classification/embedding) are batched through the executor rather than a
//! separate scheduler.
//!
//! AR scheduler inspired by SGLang:
//! - Scheduler: <https://github.com/sgl-project/sglang/blob/78ddf05a/python/sglang/srt/managers/scheduler.py>
//! - Schedule policy: <https://github.com/sgl-project/sglang/blob/78ddf05a/python/sglang/srt/managers/schedule_policy.py>
//! SGLang is licensed under the Apache License, Version 2.0.

// ── AR scheduler (current implementation) ──
mod admission;
mod preemption;
mod state;

pub mod components;

#[cfg(test)]
mod tests;

pub use components::cache::block_manager::BlockManager;
pub use components::cache::prefix_index::{PrefixInsertPlan, PrefixMatch, PrefixMatchIndex};
pub use components::cache::prefix_plan::{PrefixBoundary, PrefixResources, SharedPrefixPlanner};
pub use state::{
    FinishReason, ForwardMode, SamplingParams, SchedulePolicy, Scheduler, SchedulerConfig,
    SchedulerOutput, SchedulerStep, SeqFinishReason, Sequence, SequenceStatus,
};

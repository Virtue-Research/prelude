//! Continuous-batching LLM scheduler — zero dependencies, engine-agnostic.
//!
//! Inspired by SGLang:
//! - Scheduler: <https://github.com/sgl-project/sglang/blob/78ddf05a/python/sglang/srt/managers/scheduler.py>
//! - Schedule policy: <https://github.com/sgl-project/sglang/blob/78ddf05a/python/sglang/srt/managers/schedule_policy.py>
//! SGLang is licensed under the Apache License, Version 2.0.

mod admission;
mod preemption;
mod state;
pub(crate) mod adaptive;
pub mod scheduled_engine;

#[cfg(test)]
mod tests;

pub use crate::cache::block_manager::BlockManager;
pub use crate::cache::prefix_index::{PrefixInsertPlan, PrefixMatch, PrefixMatchIndex};
pub use state::{
    FinishReason, ForwardMode, SamplingParams, SchedulePolicy, Scheduler, SchedulerConfig,
    SchedulerOutput, SchedulerStep, SeqFinishReason, Sequence, SequenceStatus,
};
pub use scheduled_engine::ScheduledEngine;

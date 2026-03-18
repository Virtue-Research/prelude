//! Runtime module — event loops and GPU queue that drive inference.
//!
//! Scheduling decisions live in `scheduler/`; this module provides the
//! loops that consume those decisions and execute GPU work.

pub(crate) mod batch;
pub(crate) mod batch_common;
pub(crate) mod continuous;
pub(crate) mod cpu_batch_runtime;
pub(crate) mod cpu_continuous;
pub(crate) mod gpu_queue;
pub(crate) mod request_state;

// Re-export all scheduler types so existing `use crate::runtime::*` still works.
pub use crate::scheduler::{
    FinishReason, ForwardMode, SamplingParams, SchedulePolicy, Scheduler, SchedulerConfig,
    SchedulerOutput, SchedulerStep, SeqFinishReason, Sequence, SequenceStatus,
};
pub use crate::scheduler::ScheduledEngine;

pub(crate) use batch::batch_runtime_loop;
pub(crate) use continuous::continuous_generation_loop;
pub(crate) use cpu_batch_runtime::cpu_batch_runtime_loop;
pub(crate) use cpu_continuous::cpu_continuous_generation_loop;

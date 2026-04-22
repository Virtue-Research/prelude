//! Runtime module — event loops and GPU queue that drive inference.
//!
//! Scheduling decisions live in `scheduler/`; this module provides the
//! loops that consume those decisions and execute GPU work.

pub(crate) mod gpu_batch;
pub(crate) mod batch_common;
pub(crate) mod gpu_continuous;
pub(crate) mod cpu_batch;
pub(crate) mod cpu_continuous;
pub(crate) mod gpu_queue;
pub(crate) mod request_state;
#[cfg(all(feature = "cuda", any(feature = "flash-attn-v3", feature = "flash-attn-v4", feature = "flashinfer")))]
pub(crate) mod cuda_graph;

// Re-export all scheduler types so existing `use crate::runtime::*` still works.
pub use crate::scheduler::{
    FinishReason, ForwardMode, SamplingParams, SchedulePolicy, Scheduler, SchedulerConfig,
    SchedulerOutput, SchedulerStep, SeqFinishReason, Sequence, SequenceStatus,
};
pub use crate::engine::ScheduledEngine;

pub(crate) use gpu_batch::batch_runtime_loop;
pub(crate) use gpu_continuous::continuous_generation_loop;
pub(crate) use cpu_batch::cpu_batch_runtime_loop;
pub(crate) use cpu_continuous::{cpu_continuous_generation_loop, spawn_cpu_continuous_worker};

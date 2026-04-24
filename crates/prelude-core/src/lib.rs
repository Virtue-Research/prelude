pub mod tensor;

// Re-export candle's bail! macro at crate root for `crate::bail!()` usage.
pub use candle_core::bail;
pub mod config;
pub mod constants;
pub mod engine;
pub mod models;
pub mod ops;
pub mod profiling;
pub mod scheduler;
pub mod types;

/// Backward-compatible re-export: `cache` now lives at `scheduler::components::cache`.
pub use scheduler::components::cache;

/// Backward-compatible re-export: `loading` now lives under `engine`.
pub mod loading {
    pub use crate::engine::weight_loader as var_builder;
    pub use crate::engine::weights;
}

pub use cache::deltanet_pool;
pub use cache::prefix_cache;
pub use cache::prefix_index;
pub use config::EngineConfig;
pub use engine::{Engine, TaskOverride};
pub use engine::{EngineError, InferenceEngine, PseudoEngine};
pub use engine::scheduled as scheduled_engine;
pub use engine::ScheduledEngine;
pub use scheduler::SchedulerConfig;
pub use types::*;

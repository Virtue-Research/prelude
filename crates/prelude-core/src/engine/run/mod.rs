//! Scheduling-paradigm loops (device-agnostic).
//!
//! The AR loop calls `Executor::submit/collect`; device crates implement
//! `Executor` with their own execution strategy.

pub mod ar;

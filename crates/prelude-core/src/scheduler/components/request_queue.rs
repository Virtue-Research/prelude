//! Request queue — FCFS / priority / cache-aware ordering.
//!
//! Manages the waiting queue of inference requests before they're
//! admitted to running state by the scheduler. Supports multiple
//! ordering strategies.
//!
//! Not yet extracted — currently inline in scheduler state.

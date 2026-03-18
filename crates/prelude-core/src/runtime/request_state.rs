//! Request state types and scheduler message enums.
//!
//! Defines how requests flow through the scheduler: response channels,
//! per-request lifecycle state, and the message types that connect
//! ScheduledEngine to the batch and continuous runtimes.

use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

use crate::engine::{EngineError, PreparedGenerateRequest};
use crate::types::{
    ClassifyRequest, ClassifyResult, EmbedRequest, EmbedResult, GenerateRequest, GenerateResult,
    StreamEvent,
};

// ── Response channels ───────────────────────────────────────────────────

/// How generation output is delivered back to the caller.
pub(crate) enum GenerationResponseChannel {
    /// Non-streaming: send a complete GenerateResult when done.
    Complete(oneshot::Sender<Result<GenerateResult, EngineError>>),
    /// Streaming: tokens are sent per-token through this channel.
    Stream(mpsc::UnboundedSender<StreamEvent>),
}

/// Classification response channel.
pub(crate) type ClassifyResponseChannel = oneshot::Sender<Result<ClassifyResult, EngineError>>;

/// Embedding response channel.
pub(crate) type EmbedResponseChannel = oneshot::Sender<Result<EmbedResult, EngineError>>;

// ── Batch runtime request state ─────────────────────────────────────────

/// Unified generation request state used by the batch runtime.
pub(crate) struct GenerationRequestState {
    pub(crate) request: GenerateRequest,
    pub(crate) response: GenerationResponseChannel,
    /// Cached prepared request from routing-time tokenization.
    /// Avoids re-tokenizing in the batch runtime.
    pub(crate) cached_prepared: Option<PreparedGenerateRequest>,
    enqueued_at: Instant,
}

impl GenerationRequestState {
    pub(crate) fn new_complete(
        request: GenerateRequest,
        tx: oneshot::Sender<Result<GenerateResult, EngineError>>,
    ) -> Self {
        Self {
            request,
            response: GenerationResponseChannel::Complete(tx),
            cached_prepared: None,
            enqueued_at: Instant::now(),
        }
    }

    pub(crate) fn new_stream(
        request: GenerateRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> Self {
        Self {
            request,
            response: GenerationResponseChannel::Stream(tx),
            cached_prepared: None,
            enqueued_at: Instant::now(),
        }
    }

    pub(crate) fn with_prepared(mut self, prepared: PreparedGenerateRequest) -> Self {
        self.cached_prepared = Some(prepared);
        self
    }

    pub(crate) fn request(&self) -> &GenerateRequest {
        &self.request
    }

    pub(crate) fn request_id(&self) -> &str {
        &self.request.request_id
    }

    pub(crate) fn is_streaming(&self) -> bool {
        matches!(self.response, GenerationResponseChannel::Stream(_))
    }

    pub(crate) fn queue_age_ms(&self) -> f64 {
        self.enqueued_at.elapsed().as_secs_f64() * 1000.0
    }

    pub(crate) fn abort(self) {
        self.fail(EngineError::Internal("aborted".into()));
    }

    pub(crate) fn fail(self, error: EngineError) {
        tracing::trace!(
            request_id = self.request_id(),
            queue_age_ms = format!("{:.2}", self.queue_age_ms()),
            streaming = self.is_streaming(),
            "generation request failed in batch runtime"
        );
        match self.response {
            GenerationResponseChannel::Complete(tx) => {
                let _ = tx.send(Err(error));
            }
            GenerationResponseChannel::Stream(_) => {
                // Dropping sender closes the stream.
            }
        }
    }

    pub(crate) fn finish(self, run_result: Result<GenerateResult, EngineError>) {
        tracing::trace!(
            request_id = self.request_id(),
            queue_age_ms = format!("{:.2}", self.queue_age_ms()),
            streaming = self.is_streaming(),
            "generation request finished in batch runtime"
        );
        match run_result {
            Ok(result) => match self.response {
                GenerationResponseChannel::Complete(tx) => {
                    let _ = tx.send(Ok(result));
                }
                GenerationResponseChannel::Stream(tx) => {
                    let _ = tx.send(StreamEvent::Started);
                    let _ = tx.send(StreamEvent::Token {
                        text: result.output_text,
                        logprobs: None,
                    });
                    let _ = tx.send(StreamEvent::Finished {
                        finish_reason: result.finish_reason,
                        usage: result.usage,
                        metrics: result.metrics,
                    });
                }
            },
            Err(error) => self.fail(error),
        }
    }
}

// ── Continuous runtime request state ────────────────────────────────────

/// Prepared multi-token generation request routed to the continuous runtime.
pub(crate) struct ContinuousGenerationRequestState {
    pub(crate) prepared: PreparedGenerateRequest,
    pub(crate) response: GenerationResponseChannel,
    enqueued_at: Instant,
}

impl ContinuousGenerationRequestState {
    pub(crate) fn new_complete(
        prepared: PreparedGenerateRequest,
        tx: oneshot::Sender<Result<GenerateResult, EngineError>>,
    ) -> Self {
        Self {
            prepared,
            response: GenerationResponseChannel::Complete(tx),
            enqueued_at: Instant::now(),
        }
    }

    pub(crate) fn new_stream(
        prepared: PreparedGenerateRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> Self {
        Self {
            prepared,
            response: GenerationResponseChannel::Stream(tx),
            enqueued_at: Instant::now(),
        }
    }

    pub(crate) fn request_id(&self) -> &str {
        &self.prepared.request.request_id
    }

    pub(crate) fn is_streaming(&self) -> bool {
        matches!(self.response, GenerationResponseChannel::Stream(_))
    }

    pub(crate) fn queue_age_ms(&self) -> f64 {
        self.enqueued_at.elapsed().as_secs_f64() * 1000.0
    }

    pub(crate) fn fail(self, error: EngineError) {
        tracing::trace!(
            request_id = self.request_id(),
            queue_age_ms = format!("{:.2}", self.queue_age_ms()),
            streaming = self.is_streaming(),
            "generation request failed before entering continuous runtime"
        );
        match self.response {
            GenerationResponseChannel::Complete(tx) => {
                let _ = tx.send(Err(error));
            }
            GenerationResponseChannel::Stream(_) => {
                // Dropping sender closes the stream.
            }
        }
    }
}

// ── In-flight request wrappers ──────────────────────────────────────────

pub(crate) struct InFlightClassifyRequest {
    pub(crate) request: ClassifyRequest,
    pub(crate) response: ClassifyResponseChannel,
}

pub(crate) struct InFlightEmbedRequest {
    pub(crate) request: EmbedRequest,
    pub(crate) response: EmbedResponseChannel,
}

// ── Scheduler messages ──────────────────────────────────────────────────

pub(crate) enum SchedulerMsg {
    NewRequest(GenerationRequestState),
    NewClassifyRequest(InFlightClassifyRequest),
    NewEmbedRequest(InFlightEmbedRequest),
    Abort(String),
}

pub(crate) enum ContinuousSchedulerMsg {
    NewRequest(ContinuousGenerationRequestState),
    Abort(String),
}

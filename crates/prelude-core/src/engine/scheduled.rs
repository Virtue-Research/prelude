use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::{mpsc, oneshot};

use crate::engine::{Engine, EngineError, InferenceEngine, PreparedGenerateRequest};
use crate::engine::executor::{self, Executor, ForwardBatch};
use crate::engine::run::ar::{ArMessage, ResponseChannel, ar_loop};
use crate::scheduler::SchedulerConfig;
use crate::types::{
    ClassifyRequest, ClassifyResult, EmbedRequest, EmbedResult, GenerateRequest, GenerateResult,
    ModelInfo, StreamEvent,
};


// ---------------------------------------------------------------------------
// ScheduledEngine — Executor-based
// ---------------------------------------------------------------------------

pub struct ScheduledEngine {
    ar_tx: mpsc::UnboundedSender<ArMessage>,
    executor: Arc<dyn Executor>,
    model_info: ModelInfo,
    engine: Arc<Engine>,
    _ar_loop_handle: tokio::task::JoinHandle<()>,
}

impl ScheduledEngine {
    pub fn new(engine: Engine, config: SchedulerConfig) -> Self {
        let engine = Arc::new(engine);
        let model_info = ModelInfo {
            id: engine.model_id().to_string(),
            created: Utc::now().timestamp(),
            owned_by: "prelude".to_string(),
        };

        // Create the device executor (registered by device crate via ctor).
        let executor: Arc<dyn Executor> = Arc::from(
            executor::create_executor(Arc::clone(&engine))
                .unwrap_or_else(|| {
                    tracing::warn!("no device executor registered, using stub");
                    Box::new(StubExecutor)
                })
        );

        // Spawn the unified AR scheduling loop.
        let (ar_tx, ar_rx) = mpsc::unbounded_channel();
        let loop_engine = Arc::clone(&engine);
        let loop_executor = Arc::clone(&executor);
        let ar_loop_handle = tokio::spawn(async move {
            ar_loop(loop_engine, loop_executor.as_ref(), config, ar_rx).await;
        });

        tracing::info!("ScheduledEngine started (Executor-based)");

        Self {
            ar_tx,
            executor,
            model_info,
            engine,
            _ar_loop_handle: ar_loop_handle,
        }
    }

    fn prepare_and_enqueue(
        &self,
        request: GenerateRequest,
        response: ResponseChannel,
    ) -> Result<(), EngineError> {
        let prepared = tokio::task::block_in_place(|| {
            self.engine.prepare_generate_request(&request, 0)
        })?;
        self.ar_tx
            .send(ArMessage::NewRequest { prepared, response })
            .map_err(|_| EngineError::Unavailable("AR loop stopped".into()))
    }
}

/// Stub executor returned when no device crate registers one.
struct StubExecutor;

impl Executor for StubExecutor {
    fn submit(&self, _batch: ForwardBatch) -> Result<executor::ExecutionHandle, EngineError> {
        Err(EngineError::Unavailable("no device executor registered".into()))
    }
    fn collect(&self, _handle: executor::ExecutionHandle) -> Result<executor::ModelOutput, EngineError> {
        Err(EngineError::Unavailable("no device executor registered".into()))
    }
}

// ---------------------------------------------------------------------------
// InferenceEngine trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl InferenceEngine for ScheduledEngine {
    async fn model_info(&self) -> Result<ModelInfo, EngineError> {
        Ok(self.model_info.clone())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError> {
        Ok(vec![self.model_info.clone()])
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResult, EngineError> {
        let (result_tx, result_rx) = oneshot::channel();
        self.prepare_and_enqueue(request, ResponseChannel::Complete(result_tx))?;
        result_rx
            .await
            .map_err(|_| EngineError::Internal("AR loop dropped result channel".into()))?
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<(), EngineError> {
        self.prepare_and_enqueue(request, ResponseChannel::Stream(tx))
    }

    async fn cancel(&self, request_id: &str) -> Result<bool, EngineError> {
        let _ = self.ar_tx.send(ArMessage::Abort(request_id.to_string()));
        Ok(true)
    }

    async fn classify(&self, _request: ClassifyRequest) -> Result<ClassifyResult, EngineError> {
        // TODO: tokenize → executor.submit(Prefill) → collect → postprocess
        Err(EngineError::Unavailable("classify via Executor not yet wired".into()))
    }

    async fn embed(&self, _request: EmbedRequest) -> Result<EmbedResult, EngineError> {
        // TODO: tokenize → executor.submit(Prefill) → collect → postprocess
        Err(EngineError::Unavailable("embed via Executor not yet wired".into()))
    }
}

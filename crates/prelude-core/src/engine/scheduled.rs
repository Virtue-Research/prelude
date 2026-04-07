use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::{mpsc, oneshot};

use crate::engine::{Engine, EngineError, InferenceEngine, PreparedGenerateRequest};
use crate::engine::executor::{self, Executor, ForwardBatch};
use crate::engine::run::ar::{ArMessage, ResponseChannel, ar_loop};
use crate::engine::types::TaskKind;
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

        // Create the device executor (registered by device crate at startup).
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

    async fn classify(&self, request: ClassifyRequest) -> Result<ClassifyResult, EngineError> {
        let engine = Arc::clone(&self.engine);
        let executor = Arc::clone(&self.executor);
        tokio::task::spawn_blocking(move || {
            classify_via_executor(&engine, executor.as_ref(), request)
        })
        .await
        .map_err(|e| EngineError::Internal(format!("classify task panicked: {e}")))?
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResult, EngineError> {
        let engine = Arc::clone(&self.engine);
        let executor = Arc::clone(&self.executor);
        tokio::task::spawn_blocking(move || {
            embed_via_executor(&engine, executor.as_ref(), request)
        })
        .await
        .map_err(|e| EngineError::Internal(format!("embed task panicked: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// One-shot classify/embed via Executor
// ---------------------------------------------------------------------------

fn classify_via_executor(
    engine: &Engine,
    executor: &dyn Executor,
    request: ClassifyRequest,
) -> Result<ClassifyResult, EngineError> {
    use crate::engine::tokenize_batch_inputs;
    use crate::tensor::DType;
    use crate::types::{ClassificationResult, ClassificationInputs};

    let (token_ids, total_tokens) = tokenize_batch_inputs(&engine.tokenizer, &request.inputs)?;

    // Submit one-shot forward through Executor
    let batch = ForwardBatch::OneShot {
        token_groups: vec![token_ids],
        task: TaskKind::Classify,
    };
    let handle = executor.submit(batch)?;
    let output = executor.collect(handle)?;

    // Get model metadata
    let (num_labels, label_map) = engine.classify_metadata()?;

    // Postprocess: logits → per-sequence class probabilities
    let logits_f32 = output.logits
        .to_dtype(DType::F32)
        .map_err(|e| EngineError::Internal(format!("classify to_dtype: {e}")))?;
    let rows: Vec<Vec<f32>> = logits_f32
        .to_vec2()
        .map_err(|e| EngineError::Internal(format!("classify to_vec2: {e}")))?;

    let mut results = Vec::with_capacity(rows.len());
    for (idx, probs) in rows.into_iter().enumerate() {
        let (max_idx, _) = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let label = label_map.get(max_idx).cloned().flatten()
            .or_else(|| Some(format!("LABEL_{}", max_idx)));

        results.push(ClassificationResult {
            index: idx as u32,
            label,
            probs,
            num_classes: num_labels as u32,
        });
    }

    Ok(ClassifyResult {
        model: request.model,
        results,
        prompt_tokens: total_tokens,
    })
}

fn embed_via_executor(
    engine: &Engine,
    executor: &dyn Executor,
    request: EmbedRequest,
) -> Result<EmbedResult, EngineError> {
    use crate::engine::tokenize_batch_inputs;
    use crate::engine::types::EmbeddingNormalization;
    use crate::tensor::DType;
    use crate::types::EmbeddingData;

    let (token_ids, total_tokens) = tokenize_batch_inputs(&engine.tokenizer, &request.inputs)?;

    // Submit one-shot forward through Executor
    let batch = ForwardBatch::OneShot {
        token_groups: vec![token_ids],
        task: TaskKind::Embed,
    };
    let handle = executor.submit(batch)?;
    let output = executor.collect(handle)?;

    // Get model metadata
    let (dimensions, normalization) = engine.embed_metadata()?;

    // Postprocess: output → per-sequence embeddings with optional L2 normalization
    let output_f32 = output.logits
        .to_dtype(DType::F32)
        .map_err(|e| EngineError::Internal(format!("embed to_dtype: {e}")))?;
    let rows: Vec<Vec<f32>> = output_f32
        .to_vec2()
        .map_err(|e| EngineError::Internal(format!("embed to_vec2: {e}")))?;

    let mut data = Vec::with_capacity(rows.len());
    for (idx, mut embedding) in rows.into_iter().enumerate() {
        if normalization == EmbeddingNormalization::L2 {
            let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut embedding { *v /= norm; }
            }
        }
        data.push(EmbeddingData {
            index: idx as u32,
            embedding,
        });
    }

    Ok(EmbedResult {
        model: request.model,
        data,
        prompt_tokens: total_tokens,
        dimensions,
    })
}

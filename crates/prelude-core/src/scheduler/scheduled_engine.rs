use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::{mpsc, oneshot};

use crate::engine::{Engine, EngineError, InferenceEngine, PreparedGenerateRequest};
use crate::runtime::gpu_queue::spawn_gpu_worker;
use crate::runtime::{
    SchedulerConfig, batch_runtime_loop, continuous_generation_loop,
    cpu_batch_runtime_loop, cpu_continuous_generation_loop,
};
use crate::types::{
    ClassifyRequest, ClassifyResult, EmbedRequest, EmbedResult, GenerateRequest, GenerateResult,
    ModelInfo, StreamEvent,
};

pub(crate) use crate::runtime::request_state::{
    ClassifyResponseChannel, ContinuousGenerationRequestState, ContinuousSchedulerMsg,
    EmbedResponseChannel, GenerationRequestState, GenerationResponseChannel,
    InFlightClassifyRequest, InFlightEmbedRequest, SchedulerMsg,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerationSchedulerRoute {
    Batch,
    Continuous,
}

// ---------------------------------------------------------------------------
// ScheduledEngine — public API
// ---------------------------------------------------------------------------

pub struct ScheduledEngine {
    batch_tx: mpsc::UnboundedSender<SchedulerMsg>,
    continuous_tx: mpsc::UnboundedSender<ContinuousSchedulerMsg>,
    model_info: ModelInfo,
    #[allow(dead_code)]
    engine: Arc<Engine>,
    _batch_loop_handle: tokio::task::JoinHandle<()>,
    _continuous_loop_handle: Option<tokio::task::JoinHandle<()>>,
    _gpu_worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl ScheduledEngine {
    pub fn new(engine: Engine, config: SchedulerConfig) -> Self {
        let is_cpu = engine.device().is_cpu();
        let engine = Arc::new(engine);
        let model_info = ModelInfo {
            id: engine.model_id().to_string(),
            created: Utc::now().timestamp(),
            owned_by: "prelude".to_string(),
        };

        if is_cpu {
            return Self::new_cpu(engine, config, model_info);
        }

        let continuous_supported = engine.runtime_caps().supports_paged_attn;
        tracing::info!(
            classify_scheduler = "batch-runtime",
            embed_scheduler = "batch-runtime",
            prefill_only_generation_scheduler = "batch-runtime",
            multi_token_generation_scheduler = if continuous_supported {
                "continuous-runtime"
            } else {
                "unsupported"
            },
            "starting GPU scheduler runtimes"
        );

        // GPU work queue — all generation GPU work is serialized through this.
        let (gpu_tx, gpu_rx) = mpsc::unbounded_channel();
        let gpu_worker_handle = spawn_gpu_worker(gpu_rx, Arc::clone(&engine));

        let (batch_tx, batch_rx) = mpsc::unbounded_channel();
        let batch_handle = tokio::spawn(batch_runtime_loop(
            Arc::clone(&engine),
            config.clone(),
            batch_rx,
            gpu_tx.clone(),
        ));

        let (continuous_tx, continuous_rx) = mpsc::unbounded_channel();
        let continuous_handle = tokio::spawn(continuous_generation_loop(
            Arc::clone(&engine),
            config,
            continuous_rx,
            gpu_tx,
        ));

        Self {
            batch_tx,
            continuous_tx,
            model_info,
            engine,
            _batch_loop_handle: batch_handle,
            _continuous_loop_handle: Some(continuous_handle),
            _gpu_worker_handle: Some(gpu_worker_handle),
        }
    }

    /// CPU path: batch runtime for prefill-only + classify + embed,
    /// continuous runtime for multi-token decode with per-token streaming.
    fn new_cpu(engine: Arc<Engine>, config: SchedulerConfig, model_info: ModelInfo) -> Self {
        let supports_kv_cache = engine.runtime_caps().supports_kv_cache;
        tracing::info!(
            batch = "cpu-batch-runtime",
            continuous = if supports_kv_cache { "cpu-continuous-runtime" } else { "none" },
            "starting CPU scheduler runtimes"
        );

        let (batch_tx, batch_rx) = mpsc::unbounded_channel();
        let batch_handle = tokio::spawn(cpu_batch_runtime_loop(
            Arc::clone(&engine),
            config,
            batch_rx,
        ));

        let (continuous_tx, continuous_rx) = mpsc::unbounded_channel();
        let continuous_handle = if supports_kv_cache {
            Some(tokio::spawn(cpu_continuous_generation_loop(
                Arc::clone(&engine),
                continuous_rx,
            )))
        } else {
            None
        };

        Self {
            batch_tx,
            continuous_tx,
            model_info,
            engine,
            _batch_loop_handle: batch_handle,
            _continuous_loop_handle: continuous_handle,
            _gpu_worker_handle: None,
        }
    }

    fn prepare_generation_for_routing(
        &self,
        request: &GenerateRequest,
    ) -> Result<PreparedGenerateRequest, EngineError> {
        tokio::task::block_in_place(|| self.engine.prepare_generate_request(request, 0))
    }

    fn enqueue_request(
        &self,
        request: GenerateRequest,
    ) -> Result<oneshot::Receiver<Result<GenerateResult, EngineError>>, EngineError> {
        let (result_tx, result_rx) = oneshot::channel();
        let max_new = request.max_new_tokens as usize;

        if max_new <= 1 {
            // Prefill-only: route to batch without tokenizing (batch runtime tokenizes)
            let state = GenerationRequestState::new_complete(request, result_tx);
            self.batch_tx
                .send(SchedulerMsg::NewRequest(state))
                .map_err(|_| EngineError::Unavailable("batch runtime loop stopped".into()))?;
        } else {
            // Multi-token: must tokenize to check decode plan
            let prepared = self.prepare_generation_for_routing(&request)?;
            match generation_scheduler_route(self.engine.as_ref(), &request, Some(&prepared)) {
                Some(GenerationSchedulerRoute::Batch) => {
                    let state = GenerationRequestState::new_complete(request, result_tx);
                    self.batch_tx
                        .send(SchedulerMsg::NewRequest(state))
                        .map_err(|_| EngineError::Unavailable("batch runtime loop stopped".into()))?;
                }
                Some(GenerationSchedulerRoute::Continuous) => {
                    let state = ContinuousGenerationRequestState::new_complete(prepared, result_tx);
                    self.continuous_tx
                        .send(ContinuousSchedulerMsg::NewRequest(state))
                        .map_err(|_| {
                            EngineError::Unavailable("continuous generation loop stopped".into())
                        })?;
                }
                None => {
                    return Err(EngineError::Unavailable(
                        "multi-token generation requires paged attention support".into(),
                    ));
                }
            }
        }
        Ok(result_rx)
    }

    fn enqueue_stream_request(
        &self,
        request: GenerateRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<(), EngineError> {
        let max_new = request.max_new_tokens as usize;

        if max_new <= 1 {
            let state = GenerationRequestState::new_stream(request, tx);
            self.batch_tx
                .send(SchedulerMsg::NewRequest(state))
                .map_err(|_| EngineError::Unavailable("batch runtime loop stopped".into()))?;
        } else {
            let prepared = self.prepare_generation_for_routing(&request)?;
            match generation_scheduler_route(self.engine.as_ref(), &request, Some(&prepared)) {
                Some(GenerationSchedulerRoute::Batch) => {
                    let state = GenerationRequestState::new_stream(request, tx);
                    self.batch_tx
                        .send(SchedulerMsg::NewRequest(state))
                        .map_err(|_| EngineError::Unavailable("batch runtime loop stopped".into()))?;
                }
                Some(GenerationSchedulerRoute::Continuous) => {
                    let state = ContinuousGenerationRequestState::new_stream(prepared, tx);
                    self.continuous_tx
                        .send(ContinuousSchedulerMsg::NewRequest(state))
                        .map_err(|_| {
                            EngineError::Unavailable("continuous generation loop stopped".into())
                        })?;
                }
                None => {
                    return Err(EngineError::Unavailable(
                        "multi-token generation requires paged attention support".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn enqueue_classify_request(
        &self,
        request: ClassifyRequest,
    ) -> Result<oneshot::Receiver<Result<ClassifyResult, EngineError>>, EngineError> {
        let (result_tx, result_rx) = oneshot::channel();
        self.batch_tx
            .send(SchedulerMsg::NewClassifyRequest(InFlightClassifyRequest {
                request,
                response: result_tx,
            }))
            .map_err(|_| EngineError::Unavailable("batch runtime loop stopped".into()))?;
        Ok(result_rx)
    }

    fn enqueue_embed_request(
        &self,
        request: EmbedRequest,
    ) -> Result<oneshot::Receiver<Result<EmbedResult, EngineError>>, EngineError> {
        let (result_tx, result_rx) = oneshot::channel();
        self.batch_tx
            .send(SchedulerMsg::NewEmbedRequest(InFlightEmbedRequest {
                request,
                response: result_tx,
            }))
            .map_err(|_| EngineError::Unavailable("batch runtime loop stopped".into()))?;
        Ok(result_rx)
    }
}

fn select_generation_scheduler_route(
    max_new: usize,
    multi_token_decode_ready: bool,
    supports_paged_attn: bool,
) -> Option<GenerationSchedulerRoute> {
    if max_new <= 1 {
        return Some(GenerationSchedulerRoute::Batch);
    }
    if !multi_token_decode_ready {
        return None;
    }
    if supports_paged_attn {
        Some(GenerationSchedulerRoute::Continuous)
    } else {
        Some(GenerationSchedulerRoute::Batch)
    }
}

/// Route a generation request without tokenizing.
///
/// For `max_new_tokens <= 1` (prefill-only), routes to Batch immediately.
/// For `max_new_tokens > 1`, the caller must tokenize first and pass the
/// prepared request so we can check `build_decode_plan`.
fn generation_scheduler_route(
    engine: &Engine,
    request: &GenerateRequest,
    prepared: Option<&PreparedGenerateRequest>,
) -> Option<GenerationSchedulerRoute> {
    // CPU: max_new=1 → Batch, max_new>1 → Continuous (requires KV cache).
    if engine.device().is_cpu() {
        let max_new = request.max_new_tokens as usize;
        if max_new <= 1 {
            return Some(GenerationSchedulerRoute::Batch);
        }
        return if engine.runtime_caps().supports_kv_cache {
            Some(GenerationSchedulerRoute::Continuous)
        } else {
            None // model doesn't support multi-token decode
        };
    }
    let max_new = request.max_new_tokens as usize;
    if max_new <= 1 {
        return Some(GenerationSchedulerRoute::Batch);
    }
    let multi_token_decode_ready = match prepared {
        Some(p) => engine.build_decode_plan(std::slice::from_ref(p)).is_ok(),
        None => false,
    };
    select_generation_scheduler_route(
        max_new,
        multi_token_decode_ready,
        engine.runtime_caps().supports_paged_attn,
    )
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
        let result_rx = self.enqueue_request(request)?;
        result_rx
            .await
            .map_err(|_| EngineError::Internal("scheduler dropped result channel".into()))?
    }

    async fn generate_batch(
        &self,
        requests: Vec<GenerateRequest>,
    ) -> Result<Vec<GenerateResult>, EngineError> {
        let mut result_rxs = Vec::with_capacity(requests.len());
        for request in requests {
            result_rxs.push(self.enqueue_request(request)?);
        }

        let mut out = Vec::with_capacity(result_rxs.len());
        for result_rx in result_rxs {
            out.push(
                result_rx.await.map_err(|_| {
                    EngineError::Internal("scheduler dropped result channel".into())
                })??,
            );
        }
        Ok(out)
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<(), EngineError> {
        self.enqueue_stream_request(request, tx)
    }

    async fn cancel(&self, request_id: &str) -> Result<bool, EngineError> {
        let request_id = request_id.to_string();
        let _ = self.batch_tx.send(SchedulerMsg::Abort(request_id.clone()));
        let _ = self
            .continuous_tx
            .send(ContinuousSchedulerMsg::Abort(request_id));
        Ok(true)
    }

    async fn classify(&self, request: ClassifyRequest) -> Result<ClassifyResult, EngineError> {
        let result_rx = self.enqueue_classify_request(request)?;
        result_rx
            .await
            .map_err(|_| EngineError::Internal("scheduler dropped result channel".into()))?
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResult, EngineError> {
        let result_rx = self.enqueue_embed_request(request)?;
        result_rx
            .await
            .map_err(|_| EngineError::Internal("scheduler dropped result channel".into()))?
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PromptInput, SamplingParams, StopConfig};

    fn make_generate_request(request_id: &str) -> GenerateRequest {
        GenerateRequest {
            request_id: request_id.to_string(),
            model: "test-model".to_string(),
            input: PromptInput::Text("hello".to_string()),
            sampling: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            max_new_tokens: 16,
            stop: StopConfig::default(),
            seed: Some(42),
            deadline_ms: None,
            logprobs: None,
        }
    }

    #[test]
    fn generation_request_state_is_streaming() {
        let req = make_generate_request("test-complete");
        let state = GenerationRequestState::new_complete(req, oneshot::channel().0);
        assert!(!state.is_streaming());

        let req2 = make_generate_request("test-stream");
        let (tx, _rx) = mpsc::unbounded_channel();
        let state2 = GenerationRequestState::new_stream(req2, tx);
        assert!(state2.is_streaming());
    }

    #[test]
    fn selects_batch_route_for_prefill_only_generation() {
        assert_eq!(
            select_generation_scheduler_route(1, false, false),
            Some(GenerationSchedulerRoute::Batch)
        );
    }

    #[test]
    fn selects_continuous_route_for_paged_multi_token_generation() {
        assert_eq!(
            select_generation_scheduler_route(4, true, true),
            Some(GenerationSchedulerRoute::Continuous)
        );
    }

    #[test]
    fn selects_batch_route_for_nonpaged_multi_token_fallback() {
        assert_eq!(
            select_generation_scheduler_route(4, true, false),
            Some(GenerationSchedulerRoute::Batch)
        );
    }

    #[test]
    fn rejects_multi_token_generation_when_decode_is_unavailable() {
        assert_eq!(select_generation_scheduler_route(4, false, false), None);
    }
}

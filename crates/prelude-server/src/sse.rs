use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::{stream, StreamExt};
use prelude_core::{GenerateRequest, InferenceEngine, StreamEvent};

pub fn stream_sse(
    engine: Arc<dyn InferenceEngine>,
    request: GenerateRequest,
    mapper: impl Fn(StreamEvent) -> Vec<Result<Event, Infallible>> + Send + 'static,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let rid = request.request_id.clone();
    let cancel_engine = engine.clone();
    let cancel_rid = rid.clone();

    tokio::task::spawn(async move {
        if let Err(e) = engine.generate_stream(request, tx).await {
            tracing::error!(rid = %rid, error = %e, "stream generation failed");
        }
    });

    let event_stream = futures_util::stream::unfold(rx, |mut rx| async {
        rx.recv().await.map(|event| (event, rx))
    });

    let sse_stream = event_stream.flat_map(move |event| stream::iter(mapper(event)));

    // Wrap with a finalizer that cancels generation when the client disconnects.
    // Without this, a disconnected client leaves the engine spending GPU/CPU
    // compute on a request nobody is reading.
    let cancel_stream = CancelOnDrop {
        inner: Box::pin(sse_stream),
        engine: cancel_engine,
        request_id: cancel_rid,
    };

    Sse::new(cancel_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Stream wrapper that cancels the engine request when dropped (client disconnect).
struct CancelOnDrop<S> {
    inner: std::pin::Pin<Box<S>>,
    engine: Arc<dyn InferenceEngine>,
    request_id: String,
}

impl<S: futures_util::Stream> futures_util::Stream for CancelOnDrop<S> {
    type Item = S::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl<S> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        let engine = self.engine.clone();
        let rid = self.request_id.clone();
        tokio::task::spawn(async move {
            if let Err(e) = engine.cancel(&rid).await {
                tracing::debug!(rid = %rid, error = %e, "cancel on disconnect failed");
            }
        });
    }
}

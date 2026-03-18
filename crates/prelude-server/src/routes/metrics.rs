// TODO: Implement /metrics (Prometheus metrics endpoint)
//
// Standard metrics for monitoring: request count, latency histogram,
// tokens/sec, queue depth, GPU memory usage, etc.
//
// Response: Prometheus text format (text/plain)

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

pub async fn metrics() -> Response {
    (StatusCode::NOT_IMPLEMENTED, "# metrics not yet implemented\n").into_response()
}

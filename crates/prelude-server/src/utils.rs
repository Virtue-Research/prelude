use prelude_core::{DecodeMetrics, GenerateResult, Usage};
use tracing::info;

pub fn log_generation_metrics(
    usage: &Usage,
    metrics: &DecodeMetrics,
    finish_reason: &str,
    message: &str,
) {
    let decode_tps = if metrics.decode_ms > 0.0 {
        usage.completion_tokens as f32 / (metrics.decode_ms / 1000.0)
    } else {
        0.0
    };
    info!(
        prompt_tokens = usage.prompt_tokens,
        completion_tokens = usage.completion_tokens,
        ttft_ms = format!("{:.1}", metrics.ttft_ms),
        decode_ms = format!("{:.1}", metrics.decode_ms),
        total_ms = format!("{:.1}", metrics.total_ms),
        decode_tps = format!("{:.1}", decode_tps),
        finish_reason = %finish_reason,
        message,
    );
}

pub fn aggregate_usage(results: &[GenerateResult]) -> Usage {
    let mut usage = Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    };
    for result in results {
        usage.prompt_tokens = usage
            .prompt_tokens
            .saturating_add(result.usage.prompt_tokens);
        usage.completion_tokens = usage
            .completion_tokens
            .saturating_add(result.usage.completion_tokens);
        usage.total_tokens = usage.total_tokens.saturating_add(result.usage.total_tokens);
    }
    usage
}

//! Adaptive batch scheduler — EWMA-based batch size and wait time tuning.
//!
//! Tracks arrival rate and GPU time to adaptively choose batch size and wait
//! time for both generation and classification/embedding requests. The
//! `max_batch_size` and `max_batch_wait` from [`SchedulerConfig`] act as
//! hard caps; the adaptive logic operates within them.

use std::time::{Duration, Instant};

use crate::config::AdaptiveConfig;

/// Tracks arrival rate (EWMA) and GPU time to adaptively choose batch size
/// and wait time. One instance is used for generation requests, another for
/// classify/embed requests.
pub(crate) struct AdaptiveSchedulerState {
    /// EWMA of arrival rate (requests per second).
    lambda_hat: f64,
    /// Per-request GPU time at each batch size (indexed by batch size, 0 unused).
    /// `s_hat[b]` = EWMA of (total_gpu_ms / b) observed when batch size was b.
    s_hat: Vec<f64>,
    /// Whether each s_hat bucket has been observed at least once.
    s_hat_observed: Vec<bool>,
    /// Maximum waiting time cap (ms) — from config.max_batch_wait_ms.
    w_cap: f64,
    /// Hardware max batch size — from config.max_batch_size.
    b_max: usize,
    /// Timestamp of the last request arrival.
    last_arrival: Option<Instant>,
    /// EWMA smoothing factor for arrival rate (higher = more responsive).
    arrival_alpha: f64,
    /// EWMA smoothing factor for GPU time (lower = more stable).
    gpu_alpha: f64,
    /// Maximum instantaneous rate before clamping.
    max_instant_rate: f64,
}

impl AdaptiveSchedulerState {
    pub(crate) fn new(b_max: usize, w_cap_ms: f64, adaptive_config: &AdaptiveConfig) -> Self {
        Self {
            lambda_hat: adaptive_config.initial_lambda,
            s_hat: vec![0.0; b_max + 1],
            s_hat_observed: vec![false; b_max + 1],
            w_cap: w_cap_ms,
            b_max,
            last_arrival: None,
            arrival_alpha: adaptive_config.arrival_alpha,
            gpu_alpha: adaptive_config.gpu_alpha,
            max_instant_rate: adaptive_config.max_instant_rate,
        }
    }

    /// Record multiple request arrivals at once (batch counting).
    /// This prevents EWMA explosion when many requests arrive in a burst.
    pub(crate) fn record_arrivals(&mut self, count: usize, now: Instant) {
        if count == 0 {
            return;
        }
        if let Some(last) = self.last_arrival {
            let delta_s = now.duration_since(last).as_secs_f64();
            if delta_s > 0.0 && delta_s < 10.0 {
                // Use batch count to get more accurate rate
                let instant_rate = (count as f64 / delta_s).min(self.max_instant_rate);
                if self.lambda_hat <= 0.0 {
                    self.lambda_hat = instant_rate;
                } else {
                    self.lambda_hat = self.arrival_alpha * instant_rate
                        + (1.0 - self.arrival_alpha) * self.lambda_hat;
                }
            }
        }
        self.last_arrival = Some(now);
    }

    /// Record GPU completion time for a batch to update s_hat.
    pub(crate) fn record_gpu_time(&mut self, batch_size: usize, gpu_time_ms: f64) {
        if batch_size == 0 || batch_size > self.b_max {
            return;
        }
        let per_req = gpu_time_ms / batch_size as f64;
        if !self.s_hat_observed[batch_size] {
            self.s_hat[batch_size] = per_req;
            self.s_hat_observed[batch_size] = true;
        } else {
            self.s_hat[batch_size] =
                self.gpu_alpha * per_req + (1.0 - self.gpu_alpha) * self.s_hat[batch_size];
        }
    }

    /// Current EWMA arrival rate estimate (requests per second).
    pub(crate) fn lambda_hat(&self) -> f64 {
        self.lambda_hat
    }

    /// Whether any GPU time data has been observed.
    fn has_gpu_data(&self) -> bool {
        self.s_hat_observed.iter().any(|&obs| obs)
    }

    /// Look up per-request GPU time for a given batch size.
    /// If not directly observed, interpolate from nearest observed values.
    fn get_s_hat(&self, b: usize) -> f64 {
        if b == 0 || b > self.b_max {
            return 0.0;
        }
        if self.s_hat_observed[b] {
            return self.s_hat[b];
        }

        // Find nearest observed values below and above
        let mut below: Option<(usize, f64)> = None;
        let mut above: Option<(usize, f64)> = None;
        for i in (1..b).rev() {
            if self.s_hat_observed[i] {
                below = Some((i, self.s_hat[i]));
                break;
            }
        }
        for i in (b + 1)..=self.b_max {
            if self.s_hat_observed[i] {
                above = Some((i, self.s_hat[i]));
                break;
            }
        }

        match (below, above) {
            (Some((bi, bv)), Some((ai, av))) => {
                // Fit the GPU cost model  s(b) = C/b + α  to both points:
                //   bv = C/bi + α,  av = C/ai + α
                //   → C = (bv - av) · bi · ai / (ai - bi)
                //   → α = bv - C/bi
                // This captures the sublinear GPU cost structure where fixed
                // kernel launch overhead C is amortized over the batch.
                // Linear interpolation would severely underestimate marginal
                // savings at small batch sizes (e.g. batch 1→2).
                let bi_f = bi as f64;
                let ai_f = ai as f64;
                let c = (bv - av) * bi_f * ai_f / (ai_f - bi_f);
                let alpha = bv - c / bi_f;
                (c / b as f64 + alpha).max(0.0)
            }
            (Some((bi, bv)), None) => {
                // Only have data below — extrapolate using C/b model.
                // Assume total GPU time = C that is amortized, so
                // per_req(b) = (bi * bv) / b.
                let total = bi as f64 * bv;
                total / b as f64
            }
            (None, Some((ai, av))) => {
                // Only have data above — extrapolate using same C/b model.
                let total = ai as f64 * av;
                total / b as f64
            }
            (None, None) => 0.0, // No data at all
        }
    }

    /// Compute the optimal target batch size and wait duration.
    ///
    /// Implements the marginal rule: keep increasing batch size as long as
    /// the per-request GPU time savings from batching one more request exceed
    /// the expected inter-arrival wait time, subject to the wait cap.
    pub(crate) fn compute_optimal_batch_and_wait(&self, queue_len: usize) -> (usize, Duration) {
        let q = queue_len;
        let eps = 1e-6;

        // No data yet or no requests — dispatch immediately
        if self.lambda_hat <= eps {
            return (q.max(1).min(self.b_max), Duration::ZERO);
        }

        // If queue already at or above capacity, dispatch immediately
        if q >= self.b_max {
            return (self.b_max, Duration::ZERO);
        }

        // Cold start: no GPU data yet — dispatch immediately.  At c=1 this
        // avoids any wait penalty.  At high concurrency, the defer logic in
        // submit_ready() merges subsequent arrivals into large batches that
        // seed the EWMA with accurate data.  The C/b+α interpolation model
        // correctly handles batch(1) observations without pollution.
        if !self.has_gpu_data() {
            return (q.max(1).min(self.b_max), Duration::ZERO);
        }

        // Find b* by marginal rule: keep increasing batch size as long as
        // the per-request GPU time savings exceed the expected wait time.
        let mut b = q.max(1);
        while b < self.b_max {
            let s_b = self.get_s_hat(b);
            let s_b1 = self.get_s_hat(b + 1);

            // Per-request savings from adding one more to batch
            let delta_s = s_b - s_b1;
            // Expected time to wait for one more request
            let delta_w = 1000.0 / self.lambda_hat.max(eps);

            // Total wait to accumulate (b+1 - q) requests
            let wait_for_extras = if b + 1 > q {
                (b + 1 - q) as f64 / self.lambda_hat.max(eps) * 1000.0
            } else {
                0.0
            };

            if delta_s > delta_w && wait_for_extras <= self.w_cap {
                b += 1;
            } else {
                break;
            }
        }

        // Compute wait time
        let max_wait_ms = if b > q {
            let computed = (b - q) as f64 / self.lambda_hat.max(eps) * 1000.0;
            self.w_cap.min(computed.max(0.0))
        } else {
            0.0
        };

        (b, Duration::from_secs_f64(max_wait_ms / 1000.0))
    }
}

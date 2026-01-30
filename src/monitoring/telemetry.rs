//! Observability and Prometheus metrics for BitNet-Oxidized.

use std::sync::atomic::{AtomicU64, Ordering};

/// Latency histogram buckets (milliseconds).
const LATENCY_BUCKETS_MS: &[f64] = &[
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
];

/// Telemetry for request latency, throughput, and cache hit rate.
/// Exports Prometheus text exposition format.
#[derive(Default)]
pub struct Telemetry {
    request_counter: AtomicU64,
    token_counter: AtomicU64,
    error_counter: AtomicU64,
    /// Cumulative latency in milliseconds (for mean).
    latency_sum_ms: AtomicU64,
    /// Bucket counts: index i = count of requests with latency <= LATENCY_BUCKETS_MS[i].
    bucket_counts: [AtomicU64; 10],
    /// Cache hit rate 0..1 (stored as u64 = rate * 1e6).
    cache_hit_rate: AtomicU64,
}

impl Telemetry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed request (latency in ms, tokens generated).
    pub fn record_request(&self, duration_ms: f64, tokens: usize) {
        self.request_counter.fetch_add(1, Ordering::Relaxed);
        self.token_counter
            .fetch_add(tokens as u64, Ordering::Relaxed);
        let ms_u = (duration_ms * 1000.0) as u64;
        self.latency_sum_ms.fetch_add(ms_u, Ordering::Relaxed);
        for (i, &bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
            if duration_ms <= bound && i < self.bucket_counts.len() {
                self.bucket_counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn record_error(&self) {
        self.error_counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_cache_hit_rate(&self, rate: f64) {
        let v = (rate.clamp(0.0, 1.0) * 1_000_000.0) as u64;
        self.cache_hit_rate.store(v, Ordering::Relaxed);
    }

    /// Export metrics in Prometheus text exposition format.
    pub fn export_metrics(&self) -> String {
        let requests = self.request_counter.load(Ordering::Relaxed);
        let tokens = self.token_counter.load(Ordering::Relaxed);
        let errors = self.error_counter.load(Ordering::Relaxed);
        let sum_ms = self.latency_sum_ms.load(Ordering::Relaxed);
        let rate_raw = self.cache_hit_rate.load(Ordering::Relaxed);
        let cache_rate = rate_raw as f64 / 1_000_000.0;

        let mut out = String::new();
        out.push_str("# HELP bitnet_requests_total Total number of inference requests.\n");
        out.push_str("# TYPE bitnet_requests_total counter\n");
        out.push_str(&format!("bitnet_requests_total {}", requests));
        out.push('\n');

        out.push_str("# HELP bitnet_tokens_generated_total Total tokens generated.\n");
        out.push_str("# TYPE bitnet_tokens_generated_total counter\n");
        out.push_str(&format!("bitnet_tokens_generated_total {}", tokens));
        out.push('\n');

        out.push_str("# HELP bitnet_errors_total Total errors.\n");
        out.push_str("# TYPE bitnet_errors_total counter\n");
        out.push_str(&format!("bitnet_errors_total {}", errors));
        out.push('\n');

        out.push_str("# HELP bitnet_request_latency_ms_sum Sum of request latencies in ms.\n");
        out.push_str("# TYPE bitnet_request_latency_ms_sum counter\n");
        out.push_str(&format!(
            "bitnet_request_latency_ms_sum {}",
            sum_ms as f64 / 1000.0
        ));
        out.push('\n');

        out.push_str(
            "# HELP bitnet_request_latency_bucket Request latency histogram buckets (ms).\n",
        );
        out.push_str("# TYPE bitnet_request_latency_bucket histogram\n");
        for (i, &bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
            let cum = if i < self.bucket_counts.len() {
                self.bucket_counts[i].load(Ordering::Relaxed)
            } else {
                0
            };
            out.push_str(&format!(
                "bitnet_request_latency_bucket{{le=\"{}\"}} {}\n",
                bound, cum
            ));
        }
        out.push_str("bitnet_request_latency_bucket{le=\"+Inf\"} ");
        out.push_str(&requests.to_string());
        out.push('\n');

        out.push_str("# HELP bitnet_cache_hit_rate KV cache hit rate 0..1.\n");
        out.push_str("# TYPE bitnet_cache_hit_rate gauge\n");
        out.push_str(&format!("bitnet_cache_hit_rate {}", cache_rate));
        out.push('\n');

        out
    }
}

//! Utility metrics and helpers.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Detailed performance metrics.
#[derive(Debug, Clone, Default)]
pub struct DetailedMetrics {
    pub embedding_time_ms: f64,
    pub attention_time_ms: f64,
    pub ffn_time_ms: f64,
    pub total_time_ms: f64,
    pub tokens_per_second: f64,
    pub memory_peak_mb: f64,
    pub cache_hit_rate: f64,
}

/// Simple profiler for timing sections.
#[derive(Debug, Default)]
pub struct Profiler {
    start_times: HashMap<String, Instant>,
    durations: HashMap<String, Duration>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            start_times: HashMap::new(),
            durations: HashMap::new(),
        }
    }

    pub fn start(&mut self, label: &str) {
        self.start_times.insert(label.to_string(), Instant::now());
    }

    pub fn end(&mut self, label: &str) {
        if let Some(start) = self.start_times.remove(label) {
            let duration = start.elapsed();
            *self.durations.entry(label.to_string()).or_default() += duration;
        }
    }

    pub fn report(&self) -> HashMap<String, Duration> {
        self.durations.clone()
    }

    pub fn report_metrics(&self, total_tokens: usize) -> DetailedMetrics {
        let total: Duration = self.durations.values().cloned().sum();
        let total_ms = total.as_secs_f64() * 1000.0;
        let embedding_ms = self
            .durations
            .get("embedding")
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let attention_ms = self
            .durations
            .get("attention")
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let ffn_ms = self
            .durations
            .get("ffn")
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let tps = if total.as_secs_f64() > 0.0 {
            total_tokens as f64 / total.as_secs_f64()
        } else {
            0.0
        };
        DetailedMetrics {
            embedding_time_ms: embedding_ms,
            attention_time_ms: attention_ms,
            ffn_time_ms: ffn_ms,
            total_time_ms: total_ms,
            tokens_per_second: tps,
            memory_peak_mb: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

/// Perplexity (exp of mean negative log-likelihood).
pub fn perplexity(log_probs: &[f32]) -> f32 {
    if log_probs.is_empty() {
        return 1.0;
    }
    let mean_ln = log_probs.iter().sum::<f32>() / log_probs.len() as f32;
    (-mean_ln).exp()
}

/// Argmax over a slice of f32.
pub fn argmax(scores: &[f32]) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

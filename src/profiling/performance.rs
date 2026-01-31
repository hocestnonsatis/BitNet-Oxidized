//! Detailed profiler: time per layer, FLOPS estimation, Chrome trace export.
//!
//! Use with InferenceEngine to get per-layer timings and export to Chrome trace format.

use crate::inference::engine::InferenceEngine;
use crate::model::BitNetModel;
use anyhow::Result;
use serde::Serialize;
use std::time::Instant;

/// Per-layer or per-stage timing (name, duration_ms).
#[derive(Debug, Clone, Serialize)]
pub struct LayerTiming {
    pub name: String,
    pub duration_ms: f64,
    pub flops_estimate: u64,
}

/// Full profiler report: layer timings, total time, FLOPS.
#[derive(Debug, Clone, Serialize)]
pub struct ProfilerReport {
    pub layer_timings: Vec<LayerTiming>,
    pub total_ms: f64,
    pub total_flops_estimate: u64,
    pub tokens_processed: usize,
}

/// Chrome trace event (for chrome://tracing).
#[derive(Debug, Clone, Serialize)]
pub struct ChromeTraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: String,
    pub ts: f64,
    pub pid: u32,
    pub tid: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<f64>,
}

/// Inference profiler: runs forward with per-layer timing and optional Chrome trace.
pub struct InferenceProfiler {
    engine: InferenceEngine,
    model: BitNetModel,
}

impl InferenceProfiler {
    pub fn new(model: BitNetModel) -> Self {
        let engine = InferenceEngine::new(model.clone());
        Self { engine, model }
    }

    /// Reference to the underlying model (e.g. for vocab_size).
    pub fn model(&self) -> &BitNetModel {
        &self.model
    }

    /// Run one forward pass with per-layer timing. Returns (logits, report).
    pub fn forward_profiled(&self, input_ids: &[usize]) -> Result<(Vec<f32>, ProfilerReport)> {
        if input_ids.is_empty() {
            anyhow::bail!("input_ids must not be empty");
        }

        let hidden_size = self.model.hidden_size();
        let vocab_size = self.model.vocab_size();
        let _num_layers = self.model.layers.len();
        let intermediate_size = self.model.config.intermediate_size;

        let mut layer_timings = Vec::new();
        let mut total_flops: u64 = 0;

        // Embed
        let t0 = Instant::now();
        let mut hidden = self.engine.embed_tokens(input_ids);
        let embed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let embed_flops = (input_ids.len() * vocab_size * hidden_size) as u64 * 2;
        total_flops += embed_flops;
        layer_timings.push(LayerTiming {
            name: "embed".to_string(),
            duration_ms: embed_ms,
            flops_estimate: embed_flops,
        });

        let last = hidden.len().saturating_sub(1);

        for (i, layer) in self.model.layers.iter().enumerate() {
            let t0 = Instant::now();
            hidden = self.engine.forward_layer(&hidden, layer);
            let layer_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let seq_len = hidden.len();
            let attn_flops =
                seq_len * (hidden_size * hidden_size * 4 + hidden_size * hidden_size) * 2;
            let ffn_flops = seq_len
                * (hidden_size * intermediate_size * 2 + intermediate_size * hidden_size)
                * 2;
            let layer_flops = attn_flops + ffn_flops;
            total_flops += layer_flops as u64;
            layer_timings.push(LayerTiming {
                name: format!("layer_{}", i),
                duration_ms: layer_ms,
                flops_estimate: layer_flops as u64,
            });
        }

        // Final norm + LM head
        let t0 = Instant::now();
        let mut out_hidden = hidden[last].to_vec();
        self.engine
            .apply_rms_norm(&mut out_hidden, &self.model.norm);
        let mut logits = vec![0.0f32; vocab_size];
        crate::kernels::mat_vec_mul_simd(&self.model.lm_head, &out_hidden, &mut logits);
        let lm_head_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let lm_flops = (vocab_size * hidden_size) as u64 * 2;
        total_flops += lm_flops;
        layer_timings.push(LayerTiming {
            name: "lm_head".to_string(),
            duration_ms: lm_head_ms,
            flops_estimate: lm_flops,
        });

        let total_ms = layer_timings.iter().map(|t| t.duration_ms).sum();

        let report = ProfilerReport {
            layer_timings,
            total_ms,
            total_flops_estimate: total_flops,
            tokens_processed: input_ids.len(),
        };

        Ok((logits, report))
    }

    /// Export report to Chrome trace format (JSON array of events).
    pub fn report_to_chrome_trace(
        report: &ProfilerReport,
        pid: u32,
        tid: u32,
    ) -> Vec<ChromeTraceEvent> {
        let mut events = Vec::new();
        let mut ts_us = 0.0f64;
        for layer in &report.layer_timings {
            let dur_us = layer.duration_ms * 1000.0;
            events.push(ChromeTraceEvent {
                name: layer.name.clone(),
                cat: "inference".to_string(),
                ph: "B".to_string(),
                ts: ts_us,
                pid,
                tid,
                dur: None,
            });
            ts_us += dur_us;
            events.push(ChromeTraceEvent {
                name: layer.name.clone(),
                cat: "inference".to_string(),
                ph: "E".to_string(),
                ts: ts_us,
                pid,
                tid,
                dur: Some(dur_us),
            });
        }
        events
    }
}

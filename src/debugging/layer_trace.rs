//! Layer tracer: record intermediate activations through the forward pass for analysis.
//!
//! Exports trace to JSON for plotting activation distributions per layer.

use crate::kernels::mat_vec_mul_simd;
use crate::model::BitNetModel;
use crate::validation::TensorStats;
use anyhow::Result;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Per-layer activation snapshot: stats and optional sample for small tensors.
#[derive(Debug, Clone, Serialize)]
pub struct LayerActivation {
    pub layer_idx: usize,
    pub stage: String,
    pub stats: TensorStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<Vec<f32>>,
}

/// Full trace of a forward pass: one entry per layer per stage.
#[derive(Debug, Clone, Serialize)]
pub struct TraceRecord {
    pub input_ids: Vec<usize>,
    pub layers: Vec<LayerActivation>,
    pub final_logits_stats: TensorStats,
}

/// Config for tracing: max sample size per tensor, which stages to record.
#[derive(Debug, Clone)]
pub struct TraceConfig {
    pub max_sample_len: usize,
    pub record_post_norm: bool,
    pub record_post_attn: bool,
    pub record_post_ffn: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            max_sample_len: 64,
            record_post_norm: true,
            record_post_attn: true,
            record_post_ffn: true,
        }
    }
}

/// Records intermediate activations layer-by-layer for a single forward pass.
pub struct LayerTracer {
    model: BitNetModel,
    config: TraceConfig,
}

impl LayerTracer {
    pub fn new(model: BitNetModel) -> Self {
        Self {
            model,
            config: TraceConfig::default(),
        }
    }

    pub fn with_config(mut self, config: TraceConfig) -> Self {
        self.config = config;
        self
    }

    /// Run forward and record activations at each layer.
    pub fn trace_forward(&self, input_ids: &[usize]) -> Result<TraceRecord> {
        use crate::InferenceEngine;

        if input_ids.is_empty() {
            anyhow::bail!("input_ids must not be empty");
        }

        let engine = InferenceEngine::new(self.model.clone());
        let mut hidden = engine.embed_tokens(input_ids);
        let last = hidden.len().saturating_sub(1);
        let mut layers = Vec::new();

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            // Post input norm
            let mut normed: Vec<Vec<f32>> = hidden
                .iter()
                .map(|h| {
                    let mut v = h.clone();
                    engine.apply_rms_norm(&mut v, &layer.input_layernorm);
                    v
                })
                .collect();
            if self.config.record_post_norm {
                let flat: Vec<f32> = normed[last].to_vec();
                let stats = TensorStats::from_slice(&flat);
                let sample = if flat.len() <= self.config.max_sample_len {
                    Some(flat)
                } else {
                    Some(flat[..self.config.max_sample_len].to_vec())
                };
                layers.push(LayerActivation {
                    layer_idx,
                    stage: "post_input_norm".to_string(),
                    stats,
                    sample,
                });
            }

            let attn_out = engine.attention(&normed, layer);
            for (i, a) in attn_out.iter().enumerate() {
                for (j, &x) in a.iter().enumerate() {
                    normed[i][j] = hidden[i][j] + x;
                }
            }
            if self.config.record_post_attn {
                let flat: Vec<f32> = normed[last].to_vec();
                let stats = TensorStats::from_slice(&flat);
                let sample = if flat.len() <= self.config.max_sample_len {
                    Some(flat)
                } else {
                    Some(flat[..self.config.max_sample_len].to_vec())
                };
                layers.push(LayerActivation {
                    layer_idx,
                    stage: "post_attention".to_string(),
                    stats,
                    sample,
                });
            }

            let mut normed2: Vec<Vec<f32>> = normed
                .iter()
                .map(|h| {
                    let mut v = h.clone();
                    engine.apply_rms_norm(&mut v, &layer.post_attention_layernorm);
                    v
                })
                .collect();
            let ffn_out = engine.feed_forward(&normed2, layer);
            for (i, f) in ffn_out.iter().enumerate() {
                for (j, &x) in f.iter().enumerate() {
                    normed2[i][j] = normed[i][j] + x;
                }
            }
            hidden = normed2;

            if self.config.record_post_ffn {
                let flat: Vec<f32> = hidden[last].to_vec();
                let stats = TensorStats::from_slice(&flat);
                let sample = if flat.len() <= self.config.max_sample_len {
                    Some(flat)
                } else {
                    Some(flat[..self.config.max_sample_len].to_vec())
                };
                layers.push(LayerActivation {
                    layer_idx,
                    stage: "post_ffn".to_string(),
                    stats,
                    sample,
                });
            }
        }

        let mut out_hidden = hidden[last].to_vec();
        engine.apply_rms_norm(&mut out_hidden, &self.model.norm);
        let mut logits = vec![0.0f32; self.model.vocab_size()];
        mat_vec_mul_simd(&self.model.lm_head, &out_hidden, &mut logits);
        let final_logits_stats = TensorStats::from_slice(&logits);

        Ok(TraceRecord {
            input_ids: input_ids.to_vec(),
            layers,
            final_logits_stats,
        })
    }

    /// Export trace to a JSON file for analysis or plotting.
    pub fn export_to_file(record: &TraceRecord, path: &Path) -> Result<()> {
        let f = File::create(path).map_err(|e| anyhow::anyhow!("create file: {}", e))?;
        let w = BufWriter::new(f);
        serde_json::to_writer_pretty(w, record).map_err(|e| anyhow::anyhow!("serialize: {}", e))?;
        Ok(())
    }
}

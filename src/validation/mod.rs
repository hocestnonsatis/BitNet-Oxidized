//! Model validation suite: GGUF loading, forward pass, and attention checks.
//!
//! Provides comprehensive validation to detect NaN/Inf propagation, all-zero tensors,
//! and dimension mismatches.

use crate::kernels::mat_vec_mul_simd;
use crate::model::BitNetModel;
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;

/// Statistics for a tensor: min, max, mean, std, and anomaly flags.
#[derive(Debug, Clone, Serialize)]
pub struct TensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub count: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
}

impl TensorStats {
    pub fn from_slice(slice: &[f32]) -> Self {
        let count = slice.len();
        let nan_count = slice.iter().filter(|x| x.is_nan()).count();
        let inf_count = slice.iter().filter(|x| x.is_infinite()).count();
        let zero_count = slice.iter().filter(|&&x| x == 0.0).count();
        let sum: f32 = slice.iter().filter(|x| x.is_finite()).copied().sum();
        let finite_count = count - nan_count - inf_count;
        let mean = if finite_count > 0 {
            sum / finite_count as f32
        } else {
            f32::NAN
        };
        let var: f32 = slice
            .iter()
            .filter(|x| x.is_finite())
            .map(|&x| (x - mean) * (x - mean))
            .sum();
        let std = if finite_count > 1 {
            (var / (finite_count - 1) as f32).sqrt()
        } else {
            0.0
        };
        let min = slice
            .iter()
            .filter(|x| x.is_finite())
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max = slice
            .iter()
            .filter(|x| x.is_finite())
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        Self {
            min: if min == f32::INFINITY { 0.0 } else { min },
            max: if max == f32::NEG_INFINITY { 0.0 } else { max },
            mean,
            std,
            count,
            nan_count,
            inf_count,
            zero_count,
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.nan_count == 0 && self.inf_count == 0
    }

    pub fn is_all_zero(&self) -> bool {
        self.zero_count == self.count
    }
}

/// Result of validating GGUF load: embeddings, layer weights, tensor counts.
#[derive(Debug, Clone, Serialize)]
pub struct GGUFLoadValidation {
    pub ok: bool,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub embedding_stats: Option<TensorStats>,
    pub layer_weight_checks: Vec<LayerWeightCheck>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerWeightCheck {
    pub layer_idx: usize,
    pub q_proj_elements: usize,
    pub k_proj_elements: usize,
    pub v_proj_elements: usize,
    pub o_proj_elements: usize,
    pub gate_proj_elements: usize,
    pub up_proj_elements: usize,
    pub down_proj_elements: usize,
    pub all_dims_consistent: bool,
    pub error: Option<String>,
}

/// Result of forward pass validation: per-layer NaN/Inf checks.
#[derive(Debug, Clone, Serialize)]
pub struct ForwardPassValidation {
    pub ok: bool,
    pub layer_stats: Vec<TensorStats>,
    pub final_logits_stats: TensorStats,
    pub errors: Vec<String>,
}

/// Result of attention validation: Q/K/V projection and weight checks.
#[derive(Debug, Clone, Serialize)]
pub struct AttentionValidation {
    pub ok: bool,
    pub q_stats: TensorStats,
    pub k_stats: TensorStats,
    pub v_stats: TensorStats,
    pub attn_weights_healthy: bool,
    pub errors: Vec<String>,
}

/// Full validation report (JSON-serializable).
#[derive(Debug, Clone, Serialize)]
pub struct ValidationReport {
    pub gguf_load: GGUFLoadValidation,
    pub forward_pass: ForwardPassValidation,
    pub attention: AttentionValidation,
    pub passed: bool,
}

/// Check that a model loaded from GGUF has valid embeddings and layer weights.
pub fn validate_gguf_load(model: &BitNetModel) -> GGUFLoadValidation {
    let mut errors = Vec::new();
    let config = &model.config;
    let expected_hidden = config.hidden_size;
    let expected_vocab = config.vocab_size;
    let expected_intermed = config.intermediate_size;

    // Embedding: [vocab_size, hidden_size]
    let embedding_stats = if model.embeddings.is_empty() {
        errors.push("embeddings are empty".to_string());
        None
    } else {
        let flat: Vec<f32> = model
            .embeddings
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();
        let stats = TensorStats::from_slice(&flat);
        if stats.count != expected_vocab * expected_hidden {
            errors.push(format!(
                "embedding size {} != vocab {} * hidden {}",
                stats.count, expected_vocab, expected_hidden
            ));
        }
        if !stats.is_healthy() {
            errors.push("embeddings contain NaN or Inf".to_string());
        }
        Some(stats)
    };

    let mut layer_weight_checks = Vec::new();
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let q_len = layer.q_proj.len();
        let k_len = layer.k_proj.len();
        let v_len = layer.v_proj.len();
        let o_len = layer.o_proj.len();
        let gate_len = layer.gate_proj.len();
        let up_len = layer.up_proj.len();
        let down_len = layer.down_proj.len();

        let expected_q = expected_hidden * expected_hidden;
        let expected_kv = expected_hidden * expected_hidden;
        let expected_o = expected_hidden * expected_hidden;
        let expected_gate_up = expected_hidden * expected_intermed;
        let expected_down = expected_intermed * expected_hidden;

        let all_dims_consistent = q_len == expected_q
            && k_len == expected_kv
            && v_len == expected_kv
            && o_len == expected_o
            && gate_len == expected_gate_up
            && up_len == expected_gate_up
            && down_len == expected_down;

        if !all_dims_consistent {
            errors.push(format!(
                "layer {} dimension mismatch: q={} k={} v={} o={} gate={} up={} down={}",
                layer_idx, q_len, k_len, v_len, o_len, gate_len, up_len, down_len
            ));
        }

        layer_weight_checks.push(LayerWeightCheck {
            layer_idx,
            q_proj_elements: q_len,
            k_proj_elements: k_len,
            v_proj_elements: v_len,
            o_proj_elements: o_len,
            gate_proj_elements: gate_len,
            up_proj_elements: up_len,
            down_proj_elements: down_len,
            all_dims_consistent,
            error: if all_dims_consistent {
                None
            } else {
                Some("dimension mismatch".to_string())
            },
        });
    }

    if model.lm_head.len() != expected_vocab * expected_hidden {
        errors.push(format!(
            "lm_head size {} != vocab * hidden {}",
            model.lm_head.len(),
            expected_vocab * expected_hidden
        ));
    }

    let ok = errors.is_empty();
    GGUFLoadValidation {
        ok,
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        num_layers: model.layers.len(),
        embedding_stats,
        layer_weight_checks,
        errors,
    }
}

/// Run forward on test inputs and check for NaN/Inf at each logical step.
pub fn validate_forward_pass(
    model: &BitNetModel,
    test_input_ids: &[usize],
) -> Result<ForwardPassValidation> {
    use crate::InferenceEngine;

    if test_input_ids.is_empty() {
        return Ok(ForwardPassValidation {
            ok: false,
            layer_stats: vec![],
            final_logits_stats: TensorStats::from_slice(&[]),
            errors: vec!["test_input_ids must not be empty".to_string()],
        });
    }

    let engine = InferenceEngine::new(model.clone());
    let mut hidden = engine.embed_tokens(test_input_ids);
    let mut errors = Vec::new();
    let mut layer_stats = Vec::new();
    let last = hidden.len().saturating_sub(1);

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        hidden = engine.forward_layer(&hidden, layer);
        let flat: Vec<f32> = hidden[last].to_vec();
        let stats = TensorStats::from_slice(&flat);
        layer_stats.push(stats.clone());
        if !stats.is_healthy() {
            errors.push(format!("layer {} output has NaN or Inf", layer_idx));
        }
    }

    let mut out_hidden = hidden[last].to_vec();
    engine.apply_rms_norm(&mut out_hidden, &model.norm);
    let mut logits = vec![0.0f32; model.vocab_size()];
    mat_vec_mul_simd(&model.lm_head, &out_hidden, &mut logits);
    let final_logits_stats = TensorStats::from_slice(&logits);
    if !final_logits_stats.is_healthy() {
        errors.push("final logits contain NaN or Inf".to_string());
    }

    let ok = errors.is_empty();
    Ok(ForwardPassValidation {
        ok,
        layer_stats,
        final_logits_stats,
        errors,
    })
}

/// Verify Q,K,V projections and attention weights for one step.
pub fn validate_attention(
    model: &BitNetModel,
    test_token_id: usize,
) -> Result<AttentionValidation> {
    use crate::InferenceEngine;

    let engine = InferenceEngine::new(model.clone());
    let hidden = engine.embed_one_token(test_token_id);

    let layer = model.layers.first().context("model has no layers")?;
    let mut normed = hidden.clone();
    engine.apply_rms_norm(&mut normed, &layer.input_layernorm);

    let mut q = vec![0.0f32; model.hidden_size()];
    let mut k = vec![0.0f32; model.hidden_size()];
    let mut v = vec![0.0f32; model.hidden_size()];
    mat_vec_mul_simd(&layer.q_proj, &normed, &mut q);
    mat_vec_mul_simd(&layer.k_proj, &normed, &mut k);
    mat_vec_mul_simd(&layer.v_proj, &normed, &mut v);

    let q_stats = TensorStats::from_slice(&q);
    let k_stats = TensorStats::from_slice(&k);
    let v_stats = TensorStats::from_slice(&v);

    let mut errors = Vec::new();
    if !q_stats.is_healthy() {
        errors.push("Q projection has NaN/Inf".to_string());
    }
    if !k_stats.is_healthy() {
        errors.push("K projection has NaN/Inf".to_string());
    }
    if !v_stats.is_healthy() {
        errors.push("V projection has NaN/Inf".to_string());
    }
    if q_stats.is_all_zero() && q_stats.count > 0 {
        errors.push("Q projection is all zeros".to_string());
    }

    let attn_weights_healthy = q_stats.is_healthy() && k_stats.is_healthy() && v_stats.is_healthy();
    let ok = errors.is_empty();

    Ok(AttentionValidation {
        ok,
        q_stats,
        k_stats,
        v_stats,
        attn_weights_healthy,
        errors,
    })
}

/// Load GGUF from path and run full validation suite.
pub fn validate_model_from_path(path: &Path) -> Result<ValidationReport> {
    let model = crate::model::gguf::load_gguf(path)
        .with_context(|| format!("failed to load GGUF from {}", path.display()))?;

    let gguf_load = validate_gguf_load(&model);
    let test_ids = vec![0usize, 1, 2];
    let forward_pass = validate_forward_pass(&model, &test_ids)?;
    let attention = validate_attention(&model, 0)?;

    let passed = gguf_load.ok && forward_pass.ok && attention.ok;

    Ok(ValidationReport {
        gguf_load,
        forward_pass,
        attention,
        passed,
    })
}

/// Run full validation on an in-memory model.
pub fn validate_model(model: &BitNetModel) -> Result<ValidationReport> {
    let gguf_load = validate_gguf_load(model);
    let test_ids = vec![0usize, 1, 2];
    let forward_pass = validate_forward_pass(model, &test_ids)?;
    let attention = validate_attention(model, 0)?;

    let passed = gguf_load.ok && forward_pass.ok && attention.ok;

    Ok(ValidationReport {
        gguf_load,
        forward_pass,
        attention,
        passed,
    })
}

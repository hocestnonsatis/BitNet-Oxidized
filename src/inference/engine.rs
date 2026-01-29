//! Inference engine: forward pass through the BitNet model.

use crate::kernels::mat_vec_mul_lut;
use crate::model::{BitNetLayer, BitNetModel};
use anyhow::Result;

/// Engine that runs forward pass using the LUT kernel.
pub struct InferenceEngine {
    model: BitNetModel,
    hidden_size: usize,
}

impl InferenceEngine {
    pub fn new(model: BitNetModel) -> Self {
        let hidden_size = model.hidden_size();
        Self { model, hidden_size }
    }

    /// Run forward pass; returns logits for the last token [vocab_size].
    pub fn forward(&self, input_ids: &[usize]) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            anyhow::bail!("input_ids must not be empty");
        }
        let mut hidden = self.embed_tokens(input_ids);
        let last = hidden.len().saturating_sub(1);

        for layer in &self.model.layers {
            hidden = self.forward_layer(&hidden, layer);
        }

        // Final RMS norm (on last position only for output)
        let mut out_hidden = hidden[last].to_vec();
        self.apply_rms_norm(&mut out_hidden, &self.model.norm);

        // LM head: [vocab_size, hidden_size] @ [hidden_size]
        let vocab_size = self.model.vocab_size();
        let mut logits = vec![0.0f32; vocab_size];
        mat_vec_mul_lut(&self.model.lm_head, &out_hidden, &mut logits);
        Ok(logits)
    }

    /// Batch forward: run forward on each sequence (padded to same length). Returns logits for last token per item.
    pub fn forward_batch(
        &self,
        input_ids_batch: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        if input_ids_batch.is_empty() {
            return Ok(vec![]);
        }
        let max_len = input_ids_batch.iter().map(|v| v.len()).max().unwrap_or(0);
        if max_len == 0 {
            return Ok(vec![
                vec![0.0; self.model.vocab_size()];
                input_ids_batch.len()
            ]);
        }
        let pad_id = 0usize;
        let mut results = Vec::with_capacity(input_ids_batch.len());
        for seq in input_ids_batch {
            let padded: Vec<usize> = seq
                .iter()
                .cloned()
                .chain(std::iter::repeat_n(pad_id, max_len - seq.len()))
                .collect();
            let logits = self.forward(&padded)?;
            results.push(logits);
        }
        Ok(results)
    }

    /// Embed token IDs to hidden states [seq_len, hidden_size].
    pub fn embed_tokens(&self, input_ids: &[usize]) -> Vec<Vec<f32>> {
        let vocab_size = self.model.vocab_size();
        input_ids
            .iter()
            .map(|&id| {
                let id = id.min(vocab_size.saturating_sub(1));
                self.model.embeddings[id].clone()
            })
            .collect()
    }

    /// One transformer layer: pre-norm attention + residual, pre-norm FFN + residual.
    pub fn forward_layer(&self, hidden: &[Vec<f32>], layer: &BitNetLayer) -> Vec<Vec<f32>> {
        let mut normed: Vec<Vec<f32>> = hidden
            .iter()
            .map(|h| {
                let mut v = h.clone();
                self.apply_rms_norm(&mut v, &layer.input_layernorm);
                v
            })
            .collect();
        let attn_out = self.attention(&normed, layer);
        for (i, a) in attn_out.iter().enumerate() {
            for (j, &x) in a.iter().enumerate() {
                normed[i][j] = hidden[i][j] + x;
            }
        }
        let mut normed2: Vec<Vec<f32>> = normed
            .iter()
            .map(|h| {
                let mut v = h.clone();
                self.apply_rms_norm(&mut v, &layer.post_attention_layernorm);
                v
            })
            .collect();
        let ffn_out = self.feed_forward(&normed2, layer);
        for (i, f) in ffn_out.iter().enumerate() {
            for (j, &x) in f.iter().enumerate() {
                normed2[i][j] = normed[i][j] + x;
            }
        }
        normed2
    }

    /// In-place RMS normalization: x = (x / rms) * weight.
    pub fn apply_rms_norm(&self, hidden: &mut [f32], weight: &[f32]) {
        let eps = self.model.config.rms_norm_eps;
        let n = hidden.len() as f32;
        let square_sum: f32 = hidden.iter().map(|x| x * x).sum();
        let rms = (square_sum / n + eps).sqrt();
        for (i, h) in hidden.iter_mut().enumerate() {
            if i < weight.len() {
                *h = (*h / rms) * weight[i];
            } else {
                *h /= rms;
            }
        }
    }

    /// Causal self-attention: Q,K,V projections, scaled dot-product, O projection.
    pub fn attention(&self, hidden: &[Vec<f32>], layer: &BitNetLayer) -> Vec<Vec<f32>> {
        let seq_len = hidden.len();
        let head_dim = self.model.config.head_dim();
        let num_heads = self.model.config.num_attention_heads;

        let mut q = vec![vec![0.0f32; self.hidden_size]; seq_len];
        let mut k = vec![vec![0.0f32; self.hidden_size]; seq_len];
        let mut v = vec![vec![0.0f32; self.hidden_size]; seq_len];

        for t in 0..seq_len {
            mat_vec_mul_lut(&layer.q_proj, &hidden[t], &mut q[t]);
            mat_vec_mul_lut(&layer.k_proj, &hidden[t], &mut k[t]);
            mat_vec_mul_lut(&layer.v_proj, &hidden[t], &mut v[t]);
        }

        let scale = (head_dim as f32).sqrt().recip();
        let mut out = vec![vec![0.0f32; self.hidden_size]; seq_len];

        for t in 0..seq_len {
            let mut attn_weights = vec![0.0f32; t + 1];
            for s in 0..=t {
                let mut score = 0.0f32;
                for h in 0..num_heads {
                    let q_start = h * head_dim;
                    let k_start = h * head_dim;
                    for d in 0..head_dim {
                        score += q[t][q_start + d] * k[s][k_start + d];
                    }
                }
                attn_weights[s] = score * scale;
            }
            // Softmax over attn_weights
            let max_w = attn_weights
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = attn_weights.iter().map(|w| (w - max_w).exp()).sum();
            for w in &mut attn_weights {
                *w = (*w - max_w).exp() / exp_sum;
            }
            for h in 0..num_heads {
                let v_start = h * head_dim;
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for s in 0..=t {
                        sum += attn_weights[s] * v[s][v_start + d];
                    }
                    out[t][v_start + d] = sum;
                }
            }
        }

        let mut o_proj_out = vec![vec![0.0f32; self.hidden_size]; seq_len];
        for t in 0..seq_len {
            mat_vec_mul_lut(&layer.o_proj, &out[t], &mut o_proj_out[t]);
        }
        o_proj_out
    }

    /// FFN: gate_proj, up_proj, SiLU(gate) * up, down_proj.
    pub fn feed_forward(&self, hidden: &[Vec<f32>], layer: &BitNetLayer) -> Vec<Vec<f32>> {
        let seq_len = hidden.len();
        let intermed = self.model.config.intermediate_size;

        let mut gate = vec![vec![0.0f32; intermed]; seq_len];
        let mut up = vec![vec![0.0f32; intermed]; seq_len];

        for t in 0..seq_len {
            mat_vec_mul_lut(&layer.gate_proj, &hidden[t], &mut gate[t]);
            mat_vec_mul_lut(&layer.up_proj, &hidden[t], &mut up[t]);
        }

        for t in 0..seq_len {
            for i in 0..intermed {
                gate[t][i] = silu(gate[t][i]) * up[t][i];
            }
        }

        let mut out = vec![vec![0.0f32; self.hidden_size]; seq_len];
        for t in 0..seq_len {
            mat_vec_mul_lut(&layer.down_proj, &gate[t], &mut out[t]);
        }
        out
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

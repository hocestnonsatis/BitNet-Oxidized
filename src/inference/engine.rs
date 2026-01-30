//! Inference engine: forward pass through the BitNet model.

use crate::inference::cache::KVCache;
use crate::kernels::{flash_attention_forward, mat_vec_mul_simd};
use crate::model::{BitNetLayer, BitNetModel};
use anyhow::Result;

/// Sequence length above which to use flash (block-tiled) attention to save memory.
const FLASH_ATTENTION_THRESHOLD: usize = 512;
const FLASH_ATTENTION_BLOCK_SIZE: usize = 64;

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

    /// Create an empty KV cache for this model (use with `forward_step` for generation).
    pub fn create_cache(&self) -> KVCache {
        KVCache::new(&self.model.config)
    }

    /// Run one token through the model using the KV cache (attend over cache + new K/V, then append).
    /// Use for autoregressive decoding: call once per prompt token to prefill, then once per generated token.
    /// Returns logits for the current token.
    pub fn forward_step(&self, token_id: usize, cache: &mut KVCache) -> Result<Vec<f32>> {
        let mut hidden = self.embed_one_token(token_id);
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            hidden = self.forward_layer_step(&hidden, layer, layer_idx, cache)?;
        }
        cache.advance();
        self.apply_rms_norm(&mut hidden, &self.model.norm);
        let vocab_size = self.model.vocab_size();
        let mut logits = vec![0.0f32; vocab_size];
        mat_vec_mul_simd(&self.model.lm_head, &hidden, &mut logits);
        Ok(logits)
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
        mat_vec_mul_simd(&self.model.lm_head, &out_hidden, &mut logits);
        Ok(logits)
    }

    /// Run forward and return logits after each token (positions 1..=len). Used for speculative verification.
    /// Result length = input_ids.len() (logits after first token, ..., after last token).
    pub fn forward_logits_after_each(&self, input_ids: &[usize]) -> Result<Vec<Vec<f32>>> {
        if input_ids.is_empty() {
            return Ok(vec![]);
        }
        let mut hidden = self.embed_tokens(input_ids);
        let mut all_logits = Vec::with_capacity(input_ids.len());
        let vocab_size = self.model.vocab_size();

        for layer in &self.model.layers {
            hidden = self.forward_layer(&hidden, layer);
        }

        for h in &hidden {
            let mut out_hidden = h.to_vec();
            self.apply_rms_norm(&mut out_hidden, &self.model.norm);
            let mut logits = vec![0.0f32; vocab_size];
            mat_vec_mul_simd(&self.model.lm_head, &out_hidden, &mut logits);
            all_logits.push(logits);
        }
        Ok(all_logits)
    }

    /// Generate one next token (greedy) given context. Uses full forward.
    pub fn generate_one(&self, context: &[usize]) -> Result<usize> {
        let logits = self.forward(context)?;
        Ok(crate::utils::argmax(&logits).unwrap_or(0))
    }

    /// Verify draft tokens with target model: run forward on prefix||draft, return longest prefix of draft that matches target's greedy predictions.
    pub fn verify_batch(&self, prefix: &[usize], draft_tokens: &[usize]) -> Result<Vec<usize>> {
        if draft_tokens.is_empty() {
            return Ok(vec![]);
        }
        let mut seq: Vec<usize> = prefix.to_vec();
        seq.extend_from_slice(draft_tokens);
        let all_logits = self.forward_logits_after_each(&seq)?;
        let start = prefix.len().saturating_sub(1);
        let mut verified = Vec::new();
        for (i, &draft_tok) in draft_tokens.iter().enumerate() {
            let idx = start + i;
            if idx >= all_logits.len() {
                break;
            }
            let predicted = crate::utils::argmax(&all_logits[idx]).unwrap_or(0);
            if predicted == draft_tok {
                verified.push(draft_tok);
            } else {
                break;
            }
        }
        Ok(verified)
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

    /// Embed a single token to hidden state [hidden_size].
    fn embed_one_token(&self, token_id: usize) -> Vec<f32> {
        let vocab_size = self.model.vocab_size();
        let id = token_id.min(vocab_size.saturating_sub(1));
        self.model.embeddings[id].clone()
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

    /// One layer for a single token: pre-norm, Q/K/V, append to cache, attention over cache, residual, post-norm, FFN, residual.
    fn forward_layer_step(
        &self,
        hidden: &[f32],
        layer: &BitNetLayer,
        layer_idx: usize,
        cache: &mut KVCache,
    ) -> Result<Vec<f32>> {
        let mut normed = hidden.to_vec();
        self.apply_rms_norm(&mut normed, &layer.input_layernorm);

        let head_dim = self.model.config.head_dim();
        let num_kv_heads = self.model.config.num_kv_heads();

        let mut q = vec![0.0f32; self.hidden_size];
        let mut k = vec![0.0f32; self.hidden_size];
        let mut v = vec![0.0f32; self.hidden_size];
        mat_vec_mul_simd(&layer.q_proj, &normed, &mut q);
        mat_vec_mul_simd(&layer.k_proj, &normed, &mut k);
        mat_vec_mul_simd(&layer.v_proj, &normed, &mut v);

        // Store only num_kv_heads K/V (GQA: one KV head per group of Q heads)
        let keys_for_cache: Vec<Vec<Vec<f32>>> = (0..num_kv_heads)
            .map(|h| vec![k[h * head_dim..(h + 1) * head_dim].to_vec()])
            .collect();
        let values_for_cache: Vec<Vec<Vec<f32>>> = (0..num_kv_heads)
            .map(|h| vec![v[h * head_dim..(h + 1) * head_dim].to_vec()])
            .collect();

        let attn_out = self.attention_step(layer, &q, cache, layer_idx, &k, &v);
        cache.append(layer_idx, &keys_for_cache, &values_for_cache)?;
        let mut out: Vec<f32> = hidden
            .iter()
            .zip(attn_out.iter())
            .map(|(a, b)| a + b)
            .collect();

        let mut normed2 = out.clone();
        self.apply_rms_norm(&mut normed2, &layer.post_attention_layernorm);
        let ffn_out = self.feed_forward_step(layer, &normed2);
        for (i, x) in ffn_out.iter().enumerate() {
            out[i] += x;
        }
        Ok(out)
    }

    /// Causal attention for one position: Q attends over cached K/V plus new K/V (current token).
    /// GQA: each Q head h uses KV head (h / num_q_per_kv).
    fn attention_step(
        &self,
        layer: &BitNetLayer,
        q: &[f32],
        cache: &KVCache,
        layer_idx: usize,
        new_k: &[f32],
        new_v: &[f32],
    ) -> Vec<f32> {
        let head_dim = self.model.config.head_dim();
        let num_heads = self.model.config.num_attention_heads;
        let num_q_per_kv = self.model.config.num_q_per_kv();
        let cached_len = cache.current_length();
        let seq_len = cached_len + 1;
        let scale = (head_dim as f32).sqrt().recip();

        let mut out = vec![0.0f32; self.hidden_size];
        for h in 0..num_heads {
            let kv_head = h / num_q_per_kv;
            let q_start = h * head_dim;
            let k_cached = cache.keys_layer_head(layer_idx, kv_head);
            let v_cached = cache.values_layer_head(layer_idx, kv_head);

            let mut scores = vec![0.0f32; seq_len];
            for s in 0..cached_len {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_start + d] * k_cached[s][d];
                }
                scores[s] = score * scale;
            }
            {
                let kv_start = kv_head * head_dim;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_start + d] * new_k[kv_start + d];
                }
                scores[cached_len] = score * scale;
            }
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|w| (w - max_s).exp()).sum();
            for w in &mut scores {
                *w = (*w - max_s).exp() / exp_sum;
            }
            let kv_start = kv_head * head_dim;
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for s in 0..cached_len {
                    sum += scores[s] * v_cached[s][d];
                }
                sum += scores[cached_len] * new_v[kv_start + d];
                out[q_start + d] = sum;
            }
        }

        let mut o_proj_out = vec![0.0f32; self.hidden_size];
        mat_vec_mul_simd(&layer.o_proj, &out, &mut o_proj_out);
        o_proj_out
    }

    /// FFN for a single hidden vector.
    fn feed_forward_step(&self, layer: &BitNetLayer, hidden: &[f32]) -> Vec<f32> {
        let intermed = self.model.config.intermediate_size;
        let mut gate = vec![0.0f32; intermed];
        let mut up = vec![0.0f32; intermed];
        mat_vec_mul_simd(&layer.gate_proj, hidden, &mut gate);
        mat_vec_mul_simd(&layer.up_proj, hidden, &mut up);
        for i in 0..intermed {
            gate[i] = silu(gate[i]) * up[i];
        }
        let mut out = vec![0.0f32; self.hidden_size];
        mat_vec_mul_simd(&layer.down_proj, &gate, &mut out);
        out
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
    /// Uses flash (block-tiled) attention when seq_len >= FLASH_ATTENTION_THRESHOLD.
    pub fn attention(&self, hidden: &[Vec<f32>], layer: &BitNetLayer) -> Vec<Vec<f32>> {
        let seq_len = hidden.len();
        let head_dim = self.model.config.head_dim();
        let num_heads = self.model.config.num_attention_heads;

        let mut q = vec![vec![0.0f32; self.hidden_size]; seq_len];
        let mut k = vec![vec![0.0f32; self.hidden_size]; seq_len];
        let mut v = vec![vec![0.0f32; self.hidden_size]; seq_len];

        for t in 0..seq_len {
            mat_vec_mul_simd(&layer.q_proj, &hidden[t], &mut q[t]);
            mat_vec_mul_simd(&layer.k_proj, &hidden[t], &mut k[t]);
            mat_vec_mul_simd(&layer.v_proj, &hidden[t], &mut v[t]);
        }

        let out = if seq_len >= FLASH_ATTENTION_THRESHOLD {
            flash_attention_forward(&q, &k, &v, num_heads, head_dim, FLASH_ATTENTION_BLOCK_SIZE)
        } else {
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
            out
        };

        let mut o_proj_out = vec![vec![0.0f32; self.hidden_size]; seq_len];
        for t in 0..seq_len {
            mat_vec_mul_simd(&layer.o_proj, &out[t], &mut o_proj_out[t]);
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
            mat_vec_mul_simd(&layer.gate_proj, &hidden[t], &mut gate[t]);
            mat_vec_mul_simd(&layer.up_proj, &hidden[t], &mut up[t]);
        }

        for t in 0..seq_len {
            for i in 0..intermed {
                gate[t][i] = silu(gate[t][i]) * up[t][i];
            }
        }

        let mut out = vec![vec![0.0f32; self.hidden_size]; seq_len];
        for t in 0..seq_len {
            mat_vec_mul_simd(&layer.down_proj, &gate[t], &mut out[t]);
        }
        out
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

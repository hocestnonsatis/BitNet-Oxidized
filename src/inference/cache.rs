//! KV cache for fast autoregressive generation.

use crate::model::BitNetConfig;
use anyhow::Result;

/// KV cache: [layer][head][seq_len][head_dim]
#[derive(Clone)]
pub struct KVCache {
    keys: Vec<Vec<Vec<Vec<f32>>>>,
    values: Vec<Vec<Vec<Vec<f32>>>>,
    current_length: usize,
    max_length: usize,
}

impl KVCache {
    /// Create a KV cache. Uses `num_key_value_heads` for GQA/MQA (smaller cache when num_kv_heads < num_attention_heads).
    pub fn new(config: &BitNetConfig) -> Self {
        let num_layers = config.num_hidden_layers;
        let num_kv_heads = config.num_kv_heads();
        let max_len = config.max_position_embeddings;
        let head_dim = config.head_dim();

        let keys = vec![vec![vec![vec![0.0f32; head_dim]; max_len]; num_kv_heads]; num_layers];
        let values = vec![vec![vec![vec![0.0f32; head_dim]; max_len]; num_kv_heads]; num_layers];

        Self {
            keys,
            values,
            current_length: 0,
            max_length: max_len,
        }
    }

    /// Append new K/V for one token for the given layer. keys/values shape [num_heads][1][head_dim].
    pub fn append(
        &mut self,
        layer_idx: usize,
        keys: &[Vec<Vec<f32>>],
        values: &[Vec<Vec<f32>>],
    ) -> Result<()> {
        if self.current_length >= self.max_length {
            anyhow::bail!("KV cache full");
        }
        if layer_idx >= self.keys.len() {
            anyhow::bail!("layer_idx out of range");
        }
        for (h, kv) in keys.iter().enumerate() {
            if h >= self.keys[layer_idx].len() {
                break;
            }
            if !kv.is_empty() && !kv[0].is_empty() {
                for (d, &v) in kv[0].iter().enumerate() {
                    if d < self.keys[layer_idx][h][self.current_length].len() {
                        self.keys[layer_idx][h][self.current_length][d] = v;
                    }
                }
            }
        }
        for (h, kv) in values.iter().enumerate() {
            if h >= self.values[layer_idx].len() {
                break;
            }
            if !kv.is_empty() && !kv[0].is_empty() {
                for (d, &v) in kv[0].iter().enumerate() {
                    if d < self.values[layer_idx][h][self.current_length].len() {
                        self.values[layer_idx][h][self.current_length][d] = v;
                    }
                }
            }
        }
        Ok(())
    }

    /// Advance cache length by one (call after append for the last layer).
    pub fn advance(&mut self) {
        self.current_length = (self.current_length + 1).min(self.max_length);
    }

    /// Get (keys, values) for a layer. Each is [num_heads][current_length][head_dim].
    #[allow(clippy::type_complexity)]
    pub fn get(&self, layer_idx: usize) -> (&[Vec<Vec<f32>>], &[Vec<Vec<f32>>]) {
        let k = &self.keys[layer_idx];
        let v = &self.values[layer_idx];
        (k, v)
    }

    /// Current sequence length in cache.
    pub fn current_length(&self) -> usize {
        self.current_length
    }

    /// Clear cache (e.g. for new sequence).
    pub fn clear(&mut self) {
        self.current_length = 0;
    }

    /// Slice of keys for layer, head, up to current_length.
    pub fn keys_layer_head(&self, layer_idx: usize, head_idx: usize) -> &[Vec<f32>] {
        &self.keys[layer_idx][head_idx][..self.current_length]
    }

    /// Slice of values for layer, head, up to current_length.
    pub fn values_layer_head(&self, layer_idx: usize, head_idx: usize) -> &[Vec<f32>] {
        &self.values[layer_idx][head_idx][..self.current_length]
    }
}

//! BitNet model configuration.

/// Configuration for a BitNet transformer model.
#[derive(Debug, Clone)]
pub struct BitNetConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden size (embedding and hidden dimension).
    pub hidden_size: usize,
    /// Number of attention heads (query heads).
    pub num_attention_heads: usize,
    /// Number of key/value heads for GQA/MQA. Defaults to num_attention_heads (MHA).
    pub num_key_value_heads: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// FFN intermediate size (often 4 * hidden_size).
    pub intermediate_size: usize,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// Epsilon for RMS LayerNorm.
    pub rms_norm_eps: f32,
}

impl Default for BitNetConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            hidden_size: 768,
            num_attention_heads: 12,
            num_key_value_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
        }
    }
}

impl BitNetConfig {
    /// Head dimension (same for Q, K, V).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of KV heads (for GQA/MQA; defaults to num_attention_heads for MHA).
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }

    /// Repeat factor: each KV head is shared by this many Q heads.
    pub fn num_q_per_kv(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads.max(1)
    }
}

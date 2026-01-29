//! BitNet transformer architecture with ternary weights.

use crate::kernels::TernaryTensor;

use super::config::BitNetConfig;

/// Single transformer layer: attention + FFN with ternary projections.
#[derive(Clone)]
pub struct BitNetLayer {
    // Attention projections (ternary)
    pub q_proj: TernaryTensor,
    pub k_proj: TernaryTensor,
    pub v_proj: TernaryTensor,
    pub o_proj: TernaryTensor,

    // FFN (ternary)
    pub gate_proj: TernaryTensor,
    pub up_proj: TernaryTensor,
    pub down_proj: TernaryTensor,

    // LayerNorm (full precision)
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
}

/// Full BitNet model.
#[derive(Clone)]
pub struct BitNetModel {
    pub config: BitNetConfig,
    /// Token embeddings [vocab_size, hidden_size]
    pub embeddings: Vec<Vec<f32>>,
    pub layers: Vec<BitNetLayer>,
    /// Final LayerNorm weight
    pub norm: Vec<f32>,
    /// LM head (ternary): [vocab_size, hidden_size]
    pub lm_head: TernaryTensor,
}

impl BitNetModel {
    /// Hidden size from config.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Vocabulary size from config.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

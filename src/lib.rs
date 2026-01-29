//! # bitnet-oxidized
//!
//! High-performance 1-bit LLM inference framework in pure Rust, inspired by Microsoft's BitNet.
//!
//! ## Architecture
//!
//! - **Ternary weights**: 2-bit packing (4 weights per byte) with values {-1, 0, +1}
//! - **CPU kernels**: Basic, blocked, and LUT-based mat-vec (LUT is fastest)
//! - **BitNet model**: Transformer with ternary projections and full-precision LayerNorm
//! - **Inference**: Forward pass and text generation (greedy, top-k, top-p)

pub mod errors;
pub mod inference;
pub mod kernels;
pub mod model;
pub mod quantization;
pub mod server;
pub mod tokenizer;
pub mod utils;

pub use errors::BitNetError;
pub use inference::{
    DynamicBatcher, GenerationToken, InferenceEngine, KVCache, StreamGenerator, TextGenerator,
    EOS_TOKEN,
};
pub use kernels::{
    build_lut, mat_vec_mul_basic, mat_vec_mul_blocked, mat_vec_mul_lut, ByteLut, TernaryTensor,
    TernaryTensorError, TernaryWeight,
};
pub use model::{create_demo_model, BitNetConfig, BitNetLayer, BitNetModel};
pub use quantization::{absmax_dequantize, absmax_quantize, compute_quantization_error};
pub use tokenizer::BitNetTokenizer;
pub use utils::{argmax, perplexity, DetailedMetrics, Profiler};

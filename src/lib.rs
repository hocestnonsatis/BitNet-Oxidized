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

pub mod debugging;
pub mod errors;
pub mod inference;
pub mod kernels;
pub mod model;
pub mod monitoring;
pub mod optimization;
pub mod profiling;
pub mod quantization;
pub mod server;
pub mod tokenizer;
pub mod utils;
pub mod validation;

pub use debugging::LayerTracer;
pub use errors::BitNetError;
pub use inference::{
    sample_beam_search_step, sample_contrastive, sample_locally_typical, sample_min_p,
    sample_mirostat,
};
pub use inference::{
    AllowedTokensConstraint, BadWordsProcessor, CachedActivations, DynamicBatcher,
    ForcedTokensProcessor, FormatConstraint, FrequencyPenaltyProcessor, GenerationConfig,
    GenerationPipeline, GenerationToken, InferenceEngine, JsonSchema, KVCache, LexicalConstraint,
    LogitProcessor, MinPProcessor, OutputFormat, PrefixCache, ProcessorChain,
    RepetitionPenaltyProcessor, SamplingStrategy, SpeculativeDecoder, StreamGenerator,
    StructuredGenerator, TemperatureProcessor, TextGenerator, TopKProcessor, TopPProcessor,
    EOS_TOKEN,
};
pub use kernels::{
    build_lut, mat_vec_mul_basic, mat_vec_mul_blocked, mat_vec_mul_lut, ByteLut, TernaryTensor,
    TernaryTensorError, TernaryWeight,
};
pub use model::{
    convert_pytorch_to_gguf, convert_safetensors_to_gguf, create_demo_model,
    create_demo_model_seeded, from_pretrained, inspect_gguf, repair_gguf, upgrade_gguf_version,
    BitNetConfig, BitNetExpert, BitNetLayer, BitNetModel, InspectResult, MoELayer, ModelEntry,
    ModelRegistry, TensorInfoInspect,
};
pub use monitoring::Telemetry;
pub use optimization::{best_kernel_for_size, select_kernel, KernelChoice};
pub use profiling::{ChromeTraceEvent, InferenceProfiler, LayerTiming, ProfilerReport};
pub use quantization::{absmax_dequantize, absmax_quantize, compute_quantization_error};
pub use tokenizer::BitNetTokenizer;
pub use utils::{acquire, argmax, perplexity, release, DetailedMetrics, PooledBuffer, Profiler};
pub use validation::{
    validate_attention, validate_forward_pass, validate_gguf_load, validate_model,
    validate_model_from_path, AttentionValidation, ForwardPassValidation, GGUFLoadValidation,
    TensorStats, ValidationReport,
};

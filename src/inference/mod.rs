//! Inference and text generation.

pub mod cache;
pub mod constraints;
pub mod dynamic_batch;
pub mod engine;
pub mod generator;
pub mod logit_processors;
pub mod pipeline;
pub mod prefix_cache;
pub mod sampling;
pub mod speculative;
pub mod streaming;
pub mod structured;

pub use cache::KVCache;
pub use constraints::{AllowedTokensConstraint, FormatConstraint, LexicalConstraint, OutputFormat};
pub use dynamic_batch::DynamicBatcher;
pub use engine::InferenceEngine;
pub use generator::{GenerationConfig, TextGenerator};
pub use logit_processors::{
    BadWordsProcessor, ForcedTokensProcessor, FrequencyPenaltyProcessor, LogitProcessor,
    MinPProcessor, ProcessorChain, RepetitionPenaltyProcessor, TemperatureProcessor, TopKProcessor,
    TopPProcessor,
};
pub use pipeline::{GenerationPipeline, SamplingStrategy};
pub use prefix_cache::{CachedActivations, PrefixCache};
pub use sampling::{
    sample_beam_search_step, sample_contrastive, sample_locally_typical, sample_min_p,
    sample_mirostat,
};
pub use speculative::SpeculativeDecoder;
pub use streaming::{GenerationToken, StreamGenerator, EOS_TOKEN};
pub use structured::{JsonSchema, StructuredGenerator};

//! Inference and text generation.

pub mod cache;
pub mod dynamic_batch;
pub mod engine;
pub mod generator;
pub mod prefix_cache;
pub mod speculative;
pub mod streaming;
pub mod structured;

pub use cache::KVCache;
pub use dynamic_batch::DynamicBatcher;
pub use engine::InferenceEngine;
pub use generator::TextGenerator;
pub use prefix_cache::{CachedActivations, PrefixCache};
pub use speculative::SpeculativeDecoder;
pub use streaming::{GenerationToken, StreamGenerator, EOS_TOKEN};
pub use structured::{JsonSchema, StructuredGenerator};

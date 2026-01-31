//! Performance profiling: per-layer timing, FLOPS, Chrome trace export.

pub mod performance;

pub use performance::{ChromeTraceEvent, InferenceProfiler, LayerTiming, ProfilerReport};

//! Utility functions and metrics.

pub mod memory_pool;
pub mod metrics;

pub use memory_pool::{acquire, release, PooledBuffer};
pub use metrics::{argmax, perplexity, DetailedMetrics, Profiler};

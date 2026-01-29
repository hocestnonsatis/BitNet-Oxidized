//! Quantization utilities for ternary weights.

pub mod absmax;

pub use absmax::{absmax_dequantize, absmax_quantize, compute_quantization_error};

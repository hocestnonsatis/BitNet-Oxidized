//! AbsMax quantization: convert FP32 weights to ternary.

use crate::kernels::{TernaryTensor, TernaryWeight};

/// Quantize FP32 weights to ternary using sign (absmax-style: scale stored separately).
///
/// Returns (ternary_tensor, scale). Scale can be computed as max abs value per block;
/// here we use a single scale for the whole tensor: scale = max(|weights|) or 1.0 if empty.
pub fn absmax_quantize(weights: &[f32], shape: Vec<usize>) -> (TernaryTensor, f32) {
    let len: usize = shape.iter().product();
    let len = len.min(weights.len());
    let scale = if len == 0 {
        1.0f32
    } else {
        weights[..len]
            .iter()
            .map(|&w| w.abs())
            .fold(0.0f32, |a, b| a.max(b))
            .max(1e-12)
    };

    let mut tensor = TernaryTensor::zeros(len);
    for (i, &w) in weights[..len].iter().enumerate() {
        tensor.set_ternary(i, TernaryWeight::from_f32(w));
    }
    (tensor, scale)
}

/// Dequantize ternary tensor back to f32 with a single scale.
pub fn absmax_dequantize(tensor: &TernaryTensor, scale: f32) -> Vec<f32> {
    (0..tensor.len()).map(|i| tensor.get(i) * scale).collect()
}

/// Compute RMSE between original and quantized-dequantized values.
pub fn compute_quantization_error(original: &[f32], quantized: &[f32]) -> f32 {
    let n = original.len().min(quantized.len());
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f32 = original[..n]
        .iter()
        .zip(quantized[..n].iter())
        .map(|(a, q)| (a - q).powi(2))
        .sum();
    (sum_sq / n as f32).sqrt()
}

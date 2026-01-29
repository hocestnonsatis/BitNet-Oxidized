//! Quantize FP32 weights to ternary and measure error.

use bitnet_oxidized::quantization::{
    absmax_dequantize, absmax_quantize, compute_quantization_error,
};
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    let n = 10000;
    let weights: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let shape = vec![100, 100];

    let (ternary, scale) = absmax_quantize(&weights, shape);
    println!(
        "Quantized to ternary: {} elements, scale = {}",
        ternary.len(),
        scale
    );
    println!(
        "Memory: {} bytes (vs {} for FP32)",
        ternary.memory_usage(),
        n * 4
    );

    let dequant = absmax_dequantize(&ternary, scale);
    let rmse = compute_quantization_error(&weights, &dequant);
    println!("RMSE (original vs dequant): {}", rmse);

    let compression = (n * 4) as f64 / ternary.memory_usage() as f64;
    println!("Compression vs FP32: {:.1}x", compression);
}

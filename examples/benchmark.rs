//! Benchmark example: compare basic, blocked, and LUT matmul kernels.

use bitnet_oxidized::kernels::{
    mat_vec_mul_basic, mat_vec_mul_blocked, mat_vec_mul_lut, TernaryTensor,
};
use rand::Rng;
use std::time::Instant;

fn main() {
    let out_features = 256;
    let in_features = 512;
    let len = out_features * in_features;

    let mut rng = rand::thread_rng();
    let mut t = TernaryTensor::zeros(len);
    for i in 0..len {
        let v: i32 = rng.gen_range(-1..=1);
        t.set(i, v as f32);
    }
    let input: Vec<f32> = (0..in_features)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    let mut out_basic = vec![0.0f32; out_features];
    let mut out_blocked = vec![0.0f32; out_features];
    let mut out_lut = vec![0.0f32; out_features];

    let iters = 100;
    let warmup = 10;

    for _ in 0..warmup {
        mat_vec_mul_basic(&t, &input, &mut out_basic);
        mat_vec_mul_blocked(&t, &input, &mut out_blocked);
        mat_vec_mul_lut(&t, &input, &mut out_lut);
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        mat_vec_mul_basic(&t, &input, &mut out_basic);
    }
    let basic_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        mat_vec_mul_blocked(&t, &input, &mut out_blocked);
    }
    let blocked_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        mat_vec_mul_lut(&t, &input, &mut out_lut);
    }
    let lut_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    println!(
        "Matrix-vector {}x{} ({} iters):",
        out_features, in_features, iters
    );
    println!("  basic:   {:.3} ms", basic_ms);
    println!("  blocked: {:.3} ms", blocked_ms);
    println!("  LUT:     {:.3} ms", lut_ms);
    println!("  LUT speedup vs basic:   {:.2}x", basic_ms / lut_ms);
    println!("  LUT speedup vs blocked: {:.2}x", blocked_ms / lut_ms);

    for i in 0..out_features.min(5) {
        assert!(
            (out_basic[i] - out_lut[i]).abs() < 1e-3,
            "basic vs LUT mismatch at {}: {} vs {}",
            i,
            out_basic[i],
            out_lut[i]
        );
        assert!(
            (out_blocked[i] - out_lut[i]).abs() < 1e-3,
            "blocked vs LUT mismatch at {}",
            i
        );
    }
    println!("Consistency check: basic, blocked, LUT match.");
}

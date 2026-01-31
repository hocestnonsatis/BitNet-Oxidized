//! Kernel autotuning: select best mat-vec kernel for given size.
//!
//! On first run, benchmarks basic, blocked, LUT, and SIMD; caches result.
//! Override with env `BITNET_KERNEL_OVERRIDE=basic|blocked|lut|simd`.

use crate::kernels::{
    mat_vec_mul_basic, mat_vec_mul_blocked, mat_vec_mul_lut, mat_vec_mul_simd, TernaryTensor,
};
use rand::Rng;
use std::env;
use std::sync::OnceLock;
use std::time::Instant;

/// Kernel name for selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelChoice {
    Basic,
    Blocked,
    Lut,
    Simd,
}

impl KernelChoice {
    pub fn name(&self) -> &'static str {
        match self {
            KernelChoice::Basic => "basic",
            KernelChoice::Blocked => "blocked",
            KernelChoice::Lut => "lut",
            KernelChoice::Simd => "simd",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "basic" => Some(KernelChoice::Basic),
            "blocked" => Some(KernelChoice::Blocked),
            "lut" => Some(KernelChoice::Lut),
            "simd" => Some(KernelChoice::Simd),
            _ => None,
        }
    }
}

/// Run a single kernel and return median time over a few iterations.
fn bench_kernel(
    choice: KernelChoice,
    weight: &TernaryTensor,
    input: &[f32],
    output: &mut [f32],
    iters: usize,
) -> f64 {
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        match choice {
            KernelChoice::Basic => mat_vec_mul_basic(weight, input, output),
            KernelChoice::Blocked => mat_vec_mul_blocked(weight, input, output),
            KernelChoice::Lut => mat_vec_mul_lut(weight, input, output),
            KernelChoice::Simd => mat_vec_mul_simd(weight, input, output),
        }
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times[iters / 2]
}

/// Select best kernel for (out_features, in_features) by quick benchmark.
/// Cached per process. Override with BITNET_KERNEL_OVERRIDE.
pub fn best_kernel_for_size(out: usize, inp: usize) -> KernelChoice {
    if let Ok(override_name) = env::var("BITNET_KERNEL_OVERRIDE") {
        if let Some(choice) = KernelChoice::from_name(&override_name) {
            return choice;
        }
    }

    static CACHED: OnceLock<KernelChoice> = OnceLock::new();
    if let Some(&cached) = CACHED.get() {
        return cached;
    }

    let mut rng = rand::thread_rng();
    let mut t = TernaryTensor::zeros(out * inp);
    for i in 0..(out * inp) {
        t.set(i, rng.gen_range(-1..=1) as f32);
    }
    let input: Vec<f32> = (0..inp).map(|_| rng.gen()).collect();
    let mut output = vec![0.0f32; out];

    const ITERS: usize = 5;
    let basic_ms = bench_kernel(KernelChoice::Basic, &t, &input, &mut output, ITERS);
    let blocked_ms = bench_kernel(KernelChoice::Blocked, &t, &input, &mut output, ITERS);
    let lut_ms = bench_kernel(KernelChoice::Lut, &t, &input, &mut output, ITERS);
    let simd_ms = bench_kernel(KernelChoice::Simd, &t, &input, &mut output, ITERS);

    let best = [
        (KernelChoice::Basic, basic_ms),
        (KernelChoice::Blocked, blocked_ms),
        (KernelChoice::Lut, lut_ms),
        (KernelChoice::Simd, simd_ms),
    ]
    .into_iter()
    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    .map(|(k, _)| k)
    .unwrap_or(KernelChoice::Simd);

    let _ = CACHED.set(best);
    best
}

/// Run mat-vec with the selected kernel for (out, inp) size.
pub fn select_kernel(
    choice: KernelChoice,
    weight: &TernaryTensor,
    input: &[f32],
    output: &mut [f32],
) {
    match choice {
        KernelChoice::Basic => mat_vec_mul_basic(weight, input, output),
        KernelChoice::Blocked => mat_vec_mul_blocked(weight, input, output),
        KernelChoice::Lut => mat_vec_mul_lut(weight, input, output),
        KernelChoice::Simd => mat_vec_mul_simd(weight, input, output),
    }
}

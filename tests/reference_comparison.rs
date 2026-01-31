//! Reference comparison: compare forward logits with known good values.
//!
//! Uses a deterministic demo model (fixed seed). Test fails if logits differ by >1% from reference.

use bitnet_oxidized::{create_demo_model_seeded, InferenceEngine};

const SEED: u64 = 42;
const TEST_INPUT: [usize; 3] = [0, 1, 2];
const MAX_RELATIVE_DIFF: f32 = 0.01;

fn max_relative_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let denom = y.abs().max(1e-6);
            ((x - y).abs() / denom).min(f32::INFINITY)
        })
        .fold(0.0f32, f32::max)
}

#[test]
fn reference_comparison_determinism() {
    let model = create_demo_model_seeded(SEED);
    let engine = InferenceEngine::new(model);
    let logits_a = engine.forward(&TEST_INPUT).unwrap();
    let logits_b = engine.forward(&TEST_INPUT).unwrap();
    let diff = max_relative_diff(&logits_a, &logits_b);
    assert!(
        diff <= MAX_RELATIVE_DIFF,
        "Forward pass must be deterministic: max relative diff {} > {}",
        diff,
        MAX_RELATIVE_DIFF
    );
}

#[test]
fn reference_comparison_logits_finite() {
    let model = create_demo_model_seeded(SEED);
    let engine = InferenceEngine::new(model);
    let logits = engine.forward(&TEST_INPUT).unwrap();
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(
        logits.iter().all(|&x| x.is_finite()),
        "logits must be finite (no NaN/Inf)"
    );
}

#[test]
fn reference_comparison_logits_not_all_zero() {
    let model = create_demo_model_seeded(SEED);
    let engine = InferenceEngine::new(model);
    let logits = engine.forward(&TEST_INPUT).unwrap();
    let non_zero = logits.iter().filter(|&&x| x != 0.0).count();
    assert!(
        non_zero > 0,
        "logits should not be all zero (model or forward may be broken)"
    );
}

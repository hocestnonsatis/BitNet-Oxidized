//! Integration tests: ternary tensor, kernels consistency, full forward pass.

use bitnet_oxidized::kernels::{
    mat_vec_mul_basic, mat_vec_mul_blocked, mat_vec_mul_lut, TernaryTensor,
};
use bitnet_oxidized::{create_demo_model, InferenceEngine, TextGenerator};
use rand::Rng;

#[test]
fn ternary_tensor_pack_unpack() {
    let mut t = TernaryTensor::zeros(20);
    t.set(0, 1.0);
    t.set(1, -1.0);
    t.set(2, 0.0);
    assert_eq!(t.get(0), 1.0);
    assert_eq!(t.get(1), -1.0);
    assert_eq!(t.get(2), 0.0);
    assert_eq!(t.memory_usage(), (20 + 3) / 4);
}

#[test]
fn matmul_kernels_agree() {
    let mut rng = rand::thread_rng();
    let out = 32;
    let inp = 64;
    let mut w = TernaryTensor::zeros(out * inp);
    for i in 0..(out * inp) {
        w.set(i, rng.gen_range(-1..=1) as f32);
    }
    let input: Vec<f32> = (0..inp).map(|_| rng.gen()).collect();
    let mut out_basic = vec![0.0f32; out];
    let mut out_blocked = vec![0.0f32; out];
    let mut out_lut = vec![0.0f32; out];

    mat_vec_mul_basic(&w, &input, &mut out_basic);
    mat_vec_mul_blocked(&w, &input, &mut out_blocked);
    mat_vec_mul_lut(&w, &input, &mut out_lut);

    for i in 0..out {
        assert!(
            (out_basic[i] - out_lut[i]).abs() < 1e-4,
            "basic vs LUT at {}: {} vs {}",
            i,
            out_basic[i],
            out_lut[i]
        );
        assert!(
            (out_blocked[i] - out_lut[i]).abs() < 1e-4,
            "blocked vs LUT at {}",
            i
        );
    }
}

#[test]
fn forward_pass_runs() {
    let model = create_demo_model();
    let engine = InferenceEngine::new(model);
    let input_ids = vec![0usize, 1, 2];
    let logits = engine.forward(&input_ids).unwrap();
    assert_eq!(logits.len(), 256);
}

#[test]
fn generate_greedy_runs() {
    let model = create_demo_model();
    let gen = TextGenerator::new(model);
    let prompt = vec![0usize, 1];
    let out = gen.generate_greedy(&prompt, 10).unwrap();
    assert!(out.len() >= 2 && out.len() <= 10);
}

#[test]
fn generate_top_k_runs() {
    let model = create_demo_model();
    let gen = TextGenerator::new(model);
    let prompt = vec![0usize];
    let out = gen.generate_top_k(&prompt, 5, 10, 0.8).unwrap();
    assert!(out.len() >= 1 && out.len() <= 5);
}

#[test]
fn generate_top_p_runs() {
    let model = create_demo_model();
    let gen = TextGenerator::new(model);
    let prompt = vec![0usize];
    let out = gen.generate_top_p(&prompt, 5, 0.9, 0.8).unwrap();
    assert!(out.len() >= 1 && out.len() <= 5);
}

#[test]
fn gguf_roundtrip() {
    let model = create_demo_model();
    let path = std::env::temp_dir().join("bitnet_gguf_roundtrip.gguf");
    bitnet_oxidized::model::gguf::save_gguf(&model, &path).unwrap();
    let loaded = bitnet_oxidized::model::gguf::load_gguf(&path).unwrap();
    assert_eq!(loaded.vocab_size(), model.vocab_size());
    assert_eq!(loaded.hidden_size(), model.hidden_size());
    assert_eq!(loaded.num_layers(), model.num_layers());
    let _ = std::fs::remove_file(&path);
}

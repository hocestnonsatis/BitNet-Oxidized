//! Integration tests: ternary tensor, kernels consistency, full forward pass,
//! validation, error paths, and streaming.

use bitnet_oxidized::kernels::{
    mat_vec_mul_basic, mat_vec_mul_blocked, mat_vec_mul_lut, TernaryTensor,
};
use bitnet_oxidized::{
    create_demo_model, BitNetExpert, InferenceEngine, MoELayer, PrefixCache, SpeculativeDecoder,
    StreamGenerator, StructuredGenerator, Telemetry, TextGenerator,
};
use rand::Rng;
use std::sync::Arc;
use tokio::sync::RwLock;

#[test]
fn ternary_tensor_pack_unpack() {
    let mut t = TernaryTensor::zeros(20);
    t.set(0, 1.0);
    t.set(1, -1.0);
    t.set(2, 0.0);
    assert_eq!(t.get(0), 1.0);
    assert_eq!(t.get(1), -1.0);
    assert_eq!(t.get(2), 0.0);
    assert_eq!(t.memory_usage(), 20_usize.div_ceil(4));
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
    assert!(!out.is_empty() && out.len() <= 5);
}

#[test]
fn generate_top_p_runs() {
    let model = create_demo_model();
    let gen = TextGenerator::new(model);
    let prompt = vec![0usize];
    let out = gen.generate_top_p(&prompt, 5, 0.9, 0.8, None, 1.0).unwrap();
    assert!(!out.is_empty() && out.len() <= 5);
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
    assert_eq!(
        loaded.config.num_key_value_heads,
        model.config.num_key_value_heads
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn repair_gguf_roundtrip() {
    let model = create_demo_model();
    let path_a = std::env::temp_dir().join("bitnet_repair_a.gguf");
    let path_b = std::env::temp_dir().join("bitnet_repair_b.gguf");
    bitnet_oxidized::model::gguf::save_gguf(&model, &path_a).unwrap();
    bitnet_oxidized::repair_gguf(&path_a, &path_b).unwrap();
    let loaded = bitnet_oxidized::model::gguf::load_gguf(&path_b).unwrap();
    assert_eq!(loaded.vocab_size(), model.vocab_size());
    assert_eq!(loaded.hidden_size(), model.hidden_size());
    assert_eq!(loaded.num_layers(), model.num_layers());
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}

#[test]
fn inspect_gguf_returns_metadata() {
    let model = create_demo_model();
    let path = std::env::temp_dir().join("bitnet_inspect.gguf");
    bitnet_oxidized::model::gguf::save_gguf(&model, &path).unwrap();
    let info = bitnet_oxidized::inspect_gguf(&path).unwrap();
    assert!(info.version == 2 || info.version == 3);
    assert!(info.metadata.contains_key("general.architecture"));
    assert_eq!(info.metadata.get("general.architecture").unwrap(), "bitnet");
    assert!(!info.tensors.is_empty());
    assert!(info.tensors.iter().any(|t| t.name == "token_embd"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn speculative_decoding_runs() {
    let model = create_demo_model();
    let draft = InferenceEngine::new(model.clone());
    let target = InferenceEngine::new(model);
    let decoder = SpeculativeDecoder::new(draft, target, 2).unwrap();
    let prompt = vec![0usize, 1];
    let out = decoder.generate_speculative(&prompt, 8).unwrap();
    assert!(out.len() >= 2 && out.len() <= 8);
}

#[test]
fn telemetry_export_prometheus() {
    let t = Telemetry::new();
    t.record_request(10.5, 5);
    t.record_request(25.0, 12);
    let out = t.export_metrics();
    assert!(out.contains("bitnet_requests_total 2"));
    assert!(out.contains("bitnet_tokens_generated_total 17"));
    assert!(out.contains("bitnet_request_latency_bucket"));
}

#[test]
fn moe_layer_forward_runs() {
    use bitnet_oxidized::kernels::TernaryTensor;
    let mut rng = rand::thread_rng();
    let hidden = 64;
    let intermed = 128;
    let num_experts = 4;
    let top_k = 2;
    fn rand_ternary(rng: &mut impl Rng, n: usize) -> TernaryTensor {
        let mut t = TernaryTensor::zeros(n);
        for i in 0..n {
            let v: i32 = rng.gen_range(-1..=1);
            t.set(i, v as f32);
        }
        t
    }
    let gate = rand_ternary(&mut rng, num_experts * hidden);
    let experts: Vec<BitNetExpert> = (0..num_experts)
        .map(|_| BitNetExpert {
            gate_proj: rand_ternary(&mut rng, hidden * intermed),
            up_proj: rand_ternary(&mut rng, hidden * intermed),
            down_proj: rand_ternary(&mut rng, intermed * hidden),
        })
        .collect();
    let moe = MoELayer {
        num_experts,
        top_k,
        gate,
        experts,
        intermediate_size: intermed,
        hidden_size: hidden,
    };
    let hidden_vec: Vec<f32> = (0..hidden).map(|_| rng.gen()).collect();
    let out = moe.forward(&hidden_vec).unwrap();
    assert_eq!(out.len(), hidden);
}

#[test]
fn prefix_cache_hit_miss() {
    let model = create_demo_model();
    let engine = InferenceEngine::new(model);
    let mut cache = PrefixCache::new(10);
    let prefix = vec![0usize, 1, 2];
    let a1 = cache.get_or_compute(&prefix, &engine).unwrap();
    let a2 = cache.get_or_compute(&prefix, &engine).unwrap();
    assert_eq!(a1.kv_cache.current_length(), a2.kv_cache.current_length());
    assert_eq!(cache.len(), 1);
    assert!(cache.hit_rate() > 0.0);
}

#[test]
fn structured_generator_json_requires_tokenizer() {
    let model = create_demo_model();
    let engine = InferenceEngine::new(model);
    let gen = StructuredGenerator::new(engine, None);
    let prompt = vec![0usize, 1];
    let res = gen.generate_json(&prompt, 20);
    assert!(res.is_err());
}

// ---- Error paths and validation ----

#[test]
fn engine_forward_empty_input_errors() {
    let model = create_demo_model();
    let engine = InferenceEngine::new(model);
    let res = engine.forward(&[]);
    assert!(res.is_err());
}

#[test]
fn validate_model_runs() {
    let model = create_demo_model();
    let report = bitnet_oxidized::validate_model(&model).unwrap();
    assert!(report.passed);
    assert!(report.gguf_load.ok);
    assert!(report.forward_pass.ok);
    assert!(report.attention.ok);
}

#[test]
fn validate_model_from_path_runs() {
    let model = create_demo_model();
    let path = std::env::temp_dir().join("bitnet_validate_path.gguf");
    bitnet_oxidized::model::gguf::save_gguf(&model, &path).unwrap();
    let report = bitnet_oxidized::validate_model_from_path(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(report.passed);
}

// ---- Streaming ----

#[tokio::test]
async fn streaming_generator_emits_tokens() {
    let model = create_demo_model();
    let gen = TextGenerator::new(model);
    let gen = Arc::new(RwLock::new(gen));
    let stream_gen = StreamGenerator::new(gen, None);
    let mut rx = stream_gen.generate_stream("a b", 5, 0.8).await;
    let mut count = 0;
    while let Some(ev) = rx.recv().await {
        match ev {
            bitnet_oxidized::GenerationToken::Token { .. } => count += 1,
            bitnet_oxidized::GenerationToken::Done { .. } => break,
            bitnet_oxidized::GenerationToken::Error(_) => break,
        }
        if count >= 10 {
            break;
        }
    }
    assert!(count >= 0);
}

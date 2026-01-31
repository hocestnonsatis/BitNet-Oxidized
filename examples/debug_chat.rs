//! Debug example: load model + tokenizer, run forward pass, print logits stats and top predictions, then try generation.
//!
//! Run: cargo run --release --example debug_chat
//!
//! Requires models/bitnet_b1_58-large.gguf and models/bitnet_b1_58-large/tokenizer.json.

use anyhow::Result;
use bitnet_oxidized::{model::load_gguf, BitNetTokenizer, InferenceEngine, TextGenerator};

fn main() -> Result<()> {
    let model_path = "models/bitnet_b1_58-large.gguf";
    let tokenizer_path = "models/bitnet_b1_58-large/tokenizer.json";

    // 1. Load model
    let model = load_gguf(model_path)?;
    println!(
        "✓ Model loaded: vocab={}, hidden={}, layers={}",
        model.vocab_size(),
        model.hidden_size(),
        model.num_layers()
    );

    // 2. Load tokenizer
    let tok = BitNetTokenizer::from_file(tokenizer_path)?;
    println!("✓ Tokenizer loaded");

    // 3. Test encode/decode
    let test_text = "Hello";
    let test_ids = tok.encode(test_text)?;
    println!("✓ Encoded '{}' -> {:?}", test_text, test_ids);

    // 4. Create engine and run forward
    let engine = InferenceEngine::new(model.clone());
    println!("\n--- Forward pass on test_ids {:?} ---", test_ids);
    let logits = engine.forward(&test_ids)?;

    println!("Logits: len={}", logits.len());
    let min_l = logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean_l = logits.iter().sum::<f32>() / logits.len() as f32;
    println!("  min={:.4}, max={:.4}, mean={:.4}", min_l, max_l, mean_l);

    let has_nan = logits.iter().any(|&x| x.is_nan());
    let has_inf = logits.iter().any(|&x| x.is_infinite());
    if has_nan {
        println!("❌ ERROR: Logits contain NaN!");
    }
    if has_inf {
        println!("❌ ERROR: Logits contain Infinity!");
    }

    // 5. Top 10 predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nTop 10 predictions:");
    for (rank, (idx, logit)) in indexed.iter().take(10).enumerate() {
        let token = tok
            .decode(&[*idx])
            .unwrap_or_else(|_| format!("<id {}>", idx));
        println!(
            "  {}: id={} logit={:.2} text={:?}",
            rank + 1,
            idx,
            logit,
            token
        );
    }

    // 6. Try generation
    println!("\n--- Generation (top_p, temp=0.5, max 20 tokens) ---");
    let gen = TextGenerator::new(model);
    let generated = gen.generate_top_p(&test_ids, test_ids.len() + 20, 0.9, 0.5, Some(2), 1.2)?;
    let new_tokens: Vec<usize> = generated
        .iter()
        .skip(test_ids.len())
        .take_while(|&&id| id != 2)
        .copied()
        .collect();
    let output_text = tok.decode(&new_tokens).unwrap_or_default();
    println!(
        "Generated ({} new tokens): {}",
        new_tokens.len(),
        output_text.trim()
    );

    Ok(())
}

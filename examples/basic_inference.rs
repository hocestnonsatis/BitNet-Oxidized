//! Basic inference example: create a demo model and run forward pass + greedy generation.

use bitnet_oxidized::{create_demo_model, InferenceEngine, TextGenerator};

fn main() -> anyhow::Result<()> {
    println!("Creating demo BitNet model...");
    let model = create_demo_model();
    let engine = InferenceEngine::new(model.clone());

    let input_ids = [0usize, 1, 2];
    println!("Forward pass on token ids {:?}", input_ids);
    let logits = engine.forward(&input_ids)?;
    println!("Logits length: {}", logits.len());
    let best_id = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    println!("Argmax token id: {}", best_id);

    println!("Generating 5 tokens (greedy)...");
    let gen = TextGenerator::new(model);
    let generated = gen.generate_greedy(&input_ids, input_ids.len() + 5)?;
    println!("Generated ids: {:?}", generated);
    Ok(())
}

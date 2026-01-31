//! Build a tiny demo BitNet model for testing (random ternary weights).

use super::{BitNetConfig, BitNetLayer, BitNetModel};
use crate::kernels::{TernaryTensor, TernaryWeight};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn rand_ternary(rng: &mut impl Rng, n: usize) -> TernaryTensor {
    let mut t = TernaryTensor::zeros(n);
    for i in 0..n {
        let v: i32 = rng.gen_range(-1..=1);
        t.set_ternary(
            i,
            match v {
                -1 => TernaryWeight::MinusOne,
                0 => TernaryWeight::Zero,
                _ => TernaryWeight::PlusOne,
            },
        );
    }
    t
}

fn rand_f32_vec(rng: &mut impl Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(0.9f32..=1.1f32)).collect()
}

/// Create a small demo model with random ternary weights for testing/demo.
pub fn create_demo_model() -> BitNetModel {
    create_demo_model_seeded(rand::random::<u64>())
}

/// Create a deterministic demo model from a seed (for tests and golden outputs).
pub fn create_demo_model_seeded(seed: u64) -> BitNetModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let config = BitNetConfig {
        vocab_size: 256,
        hidden_size: 64,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        num_hidden_layers: 2,
        intermediate_size: 128,
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
    };
    let hidden = config.hidden_size;
    let intermed = config.intermediate_size;
    let vocab = config.vocab_size;

    let layers = (0..config.num_hidden_layers)
        .map(|_| BitNetLayer {
            q_proj: rand_ternary(&mut rng, hidden * hidden),
            k_proj: rand_ternary(&mut rng, hidden * hidden),
            v_proj: rand_ternary(&mut rng, hidden * hidden),
            o_proj: rand_ternary(&mut rng, hidden * hidden),
            gate_proj: rand_ternary(&mut rng, hidden * intermed),
            up_proj: rand_ternary(&mut rng, hidden * intermed),
            down_proj: rand_ternary(&mut rng, intermed * hidden),
            input_layernorm: rand_f32_vec(&mut rng, hidden),
            post_attention_layernorm: rand_f32_vec(&mut rng, hidden),
        })
        .collect();

    let embeddings: Vec<Vec<f32>> = (0..vocab)
        .map(|_| {
            (0..hidden)
                .map(|_| rng.gen_range(-0.1f32..0.1f32))
                .collect()
        })
        .collect();

    let lm_head = rand_ternary(&mut rng, vocab * hidden);
    let norm = rand_f32_vec(&mut rng, hidden);

    BitNetModel {
        config,
        embeddings,
        layers,
        norm,
        lm_head,
    }
}

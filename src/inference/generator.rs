//! Text generation with greedy, top-k, and top-p sampling.

use crate::inference::engine::InferenceEngine;
use crate::model::BitNetModel;
use crate::utils::argmax;
use anyhow::Result;
use rand::Rng;

/// Default RNG for sampling (thread-local).
fn rng() -> rand::rngs::ThreadRng {
    rand::thread_rng()
}

/// Text generator wrapping the inference engine.
pub struct TextGenerator {
    engine: InferenceEngine,
}

impl TextGenerator {
    pub fn new(model: BitNetModel) -> Self {
        Self {
            engine: InferenceEngine::new(model),
        }
    }

    /// Generate token IDs greedily (argmax at each step). Uses KV cache: one forward step per token.
    pub fn generate_greedy(&self, prompt_ids: &[usize], max_length: usize) -> Result<Vec<usize>> {
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt_ids must not be empty");
        }
        let mut cache = self.engine.create_cache();
        let mut ids = Vec::with_capacity(max_length);

        let mut logits = Vec::new();
        for &token_id in prompt_ids {
            logits = self.engine.forward_step(token_id, &mut cache)?;
            ids.push(token_id);
        }
        for _ in ids.len()..max_length {
            let next_id = argmax(&logits).unwrap_or(0);
            ids.push(next_id);
            logits = self.engine.forward_step(next_id, &mut cache)?;
        }
        Ok(ids)
    }

    /// Generate with top-k sampling and temperature. Uses KV cache.
    pub fn generate_top_k(
        &self,
        prompt_ids: &[usize],
        max_length: usize,
        k: usize,
        temperature: f32,
    ) -> Result<Vec<usize>> {
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt_ids must not be empty");
        }
        let mut cache = self.engine.create_cache();
        let mut ids = Vec::with_capacity(max_length);
        let t = temperature.max(1e-6);

        let mut logits = Vec::new();
        for &token_id in prompt_ids {
            logits = self.engine.forward_step(token_id, &mut cache)?;
            ids.push(token_id);
        }
        for _ in ids.len()..max_length {
            let next_id = sample_top_k(&logits, k, t)?;
            ids.push(next_id);
            logits = self.engine.forward_step(next_id, &mut cache)?;
        }
        Ok(ids)
    }

    /// Generate with top-p (nucleus) sampling and temperature. Uses KV cache.
    pub fn generate_top_p(
        &self,
        prompt_ids: &[usize],
        max_length: usize,
        p: f32,
        temperature: f32,
    ) -> Result<Vec<usize>> {
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt_ids must not be empty");
        }
        let mut cache = self.engine.create_cache();
        let mut ids = Vec::with_capacity(max_length);
        let t = temperature.max(1e-6);

        let mut logits = Vec::new();
        for &token_id in prompt_ids {
            logits = self.engine.forward_step(token_id, &mut cache)?;
            ids.push(token_id);
        }
        for _ in ids.len()..max_length {
            let next_id = sample_top_p(&logits, p, t)?;
            ids.push(next_id);
            logits = self.engine.forward_step(next_id, &mut cache)?;
        }
        Ok(ids)
    }
}

/// Sample one token using top-k: keep top k logits, softmax, sample.
fn sample_top_k(logits: &[f32], k: usize, temperature: f32) -> Result<usize> {
    let k = k.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temperature))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    let max_ln = indexed
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = indexed.iter().map(|(_, v)| (v - max_ln).exp()).sum();
    let r: f32 = rng().gen();
    let mut cum = 0.0f32;
    for (idx, &v) in indexed.iter().map(|(i, v)| (*i, v)) {
        cum += (v - max_ln).exp() / sum;
        if r <= cum {
            return Ok(idx);
        }
    }
    Ok(indexed.last().map(|(i, _)| *i).unwrap_or(0))
}

/// Sample one token using top-p: smallest set whose cumulative probability >= p.
fn sample_top_p(logits: &[f32], p: f32, temperature: f32) -> Result<usize> {
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temperature))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_ln = indexed
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_ln).exp()).collect();
    let sum_all: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|e| e / sum_all).collect();

    let mut cum = 0.0f32;
    let mut n = 0;
    for (i, &pr) in probs.iter().enumerate() {
        cum += pr;
        n = i + 1;
        if cum >= p {
            break;
        }
    }
    let top_n = n.max(1);
    let sum_top: f32 = probs[..top_n].iter().sum();
    let r: f32 = rng().gen();
    let mut cum = 0.0f32;
    for (i, &pr) in probs[..top_n].iter().enumerate() {
        cum += pr / sum_top;
        if r <= cum {
            return Ok(indexed[i].0);
        }
    }
    Ok(indexed[top_n - 1].0)
}

//! Text generation with greedy, top-k, and top-p sampling.

use crate::inference::engine::InferenceEngine;
use crate::model::BitNetModel;
use crate::utils::argmax;
use anyhow::Result;
use rand::Rng;
use std::collections::HashMap;

/// Default RNG for sampling (thread-local).
fn rng() -> rand::rngs::ThreadRng {
    rand::thread_rng()
}

/// Generation parameters for quality control (repetition/frequency/presence penalties, top-k/top-p).
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub eos_token_id: Option<usize>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(50),
            repetition_penalty: 1.15,
            frequency_penalty: 0.5,
            presence_penalty: 0.3,
            eos_token_id: Some(2),
        }
    }
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
    /// Stops at `eos_token_id` if given. Applies `repetition_penalty` to logits of already-generated tokens.
    pub fn generate_top_p(
        &self,
        prompt_ids: &[usize],
        max_length: usize,
        p: f32,
        temperature: f32,
        eos_token_id: Option<usize>,
        repetition_penalty: f32,
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
            if repetition_penalty != 1.0 && repetition_penalty > 0.0 {
                for &tid in ids.iter() {
                    if tid < logits.len() {
                        let v = logits[tid];
                        logits[tid] = if v > 0.0 {
                            v / repetition_penalty
                        } else {
                            v * repetition_penalty
                        };
                    }
                }
            }
            let next_id = sample_top_p(&logits, p, t)?;
            ids.push(next_id);
            if eos_token_id == Some(next_id) {
                break;
            }
            logits = self.engine.forward_step(next_id, &mut cache)?;
        }
        Ok(ids)
    }

    /// Apply repetition, frequency, and presence penalties to logits in place.
    fn apply_penalties(
        logits: &mut [f32],
        generated: &[usize],
        token_counts: &HashMap<usize, usize>,
        config: &GenerationConfig,
    ) {
        // 1. Repetition penalty (transformers-style)
        if config.repetition_penalty != 1.0 && config.repetition_penalty > 0.0 {
            for &tid in generated {
                if tid < logits.len() {
                    let v = logits[tid];
                    logits[tid] = if v > 0.0 {
                        v / config.repetition_penalty
                    } else {
                        v * config.repetition_penalty
                    };
                }
            }
        }
        // 2. Frequency penalty
        if config.frequency_penalty != 0.0 {
            for (&tid, &count) in token_counts {
                if tid < logits.len() {
                    logits[tid] -= config.frequency_penalty * count as f32;
                }
            }
        }
        // 3. Presence penalty
        if config.presence_penalty != 0.0 {
            for &tid in generated {
                if tid < logits.len() {
                    logits[tid] -= config.presence_penalty;
                }
            }
        }
    }

    /// Generate with full config: KV cache, all penalties, top-k + top-p (nucleus) sampling.
    /// When `debug` is true, prints first sampling step: logits before/after penalties and sampled token.
    pub fn generate_with_config(
        &self,
        prompt_ids: &[usize],
        config: &GenerationConfig,
        debug: bool,
    ) -> Result<Vec<usize>> {
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt_ids must not be empty");
        }
        let mut cache = self.engine.create_cache();
        let mut ids = Vec::with_capacity(prompt_ids.len() + config.max_tokens);
        let mut token_counts: HashMap<usize, usize> = HashMap::new();
        let t = config.temperature.max(1e-6);

        let mut logits = Vec::new();
        for &token_id in prompt_ids {
            logits = self.engine.forward_step(token_id, &mut cache)?;
            ids.push(token_id);
            *token_counts.entry(token_id).or_insert(0) += 1;
        }

        let max_length = (prompt_ids.len() + config.max_tokens).min(ids.capacity());
        let mut first_step = debug;
        for _ in ids.len()..max_length {
            if first_step {
                eprintln!("\nFirst generation step debug:");
                eprintln!("  Input tokens: {:?}", ids);
                let mut sorted: Vec<(usize, f32)> =
                    logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!("  Logits before penalties (top 10):");
                for (i, (tid, l)) in sorted.iter().take(10).enumerate() {
                    eprintln!("    {}: token_id={} logit={:.2}", i, tid, l);
                }
            }
            Self::apply_penalties(&mut logits, &ids, &token_counts, config);
            if first_step {
                let mut sorted: Vec<(usize, f32)> =
                    logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!("  Logits after penalties (top 10):");
                for (i, (tid, l)) in sorted.iter().take(10).enumerate() {
                    eprintln!("    {}: token_id={} logit={:.2}", i, tid, l);
                }
            }
            let next_id = sample_nucleus(&logits, config.top_p, config.top_k, t)?;
            if first_step {
                eprintln!("  Sampled token: {}", next_id);
                eprintln!();
                first_step = false;
            }
            ids.push(next_id);
            *token_counts.entry(next_id).or_insert(0) += 1;
            if config.eos_token_id == Some(next_id) {
                break;
            }
            logits = self.engine.forward_step(next_id, &mut cache)?;
        }
        Ok(ids)
    }
}

/// Sample one token using top-k + top-p (nucleus): apply top-k first, then nucleus filtering.
fn sample_nucleus(
    logits: &[f32],
    top_p: f32,
    top_k: Option<usize>,
    temperature: f32,
) -> Result<usize> {
    let t = temperature.max(1e-6);
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / t))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(k) = top_k {
        let k = k.min(indexed.len());
        indexed.truncate(k);
    }
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
        if cum >= top_p {
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

/// Sample one token using top-k: keep top k logits, softmax, sample. keep top k logits, softmax, sample.
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

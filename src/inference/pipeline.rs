//! Composable generation pipeline: chain logit processors and sample.
//!
//! Example: `pipeline.add(RepetitionPenalty(1.2)).add(TopK(50)).generate()`

use crate::inference::engine::InferenceEngine;
use crate::inference::logit_processors::{LogitProcessor, ProcessorChain};
use crate::model::BitNetModel;
use crate::utils::argmax;
use anyhow::Result;
use rand::Rng;

/// Sampling strategy for the pipeline.
#[derive(Clone, Debug)]
pub enum SamplingStrategy {
    /// Greedy: argmax.
    Greedy,
    /// Top-k + temperature.
    TopK { k: usize, temperature: f32 },
    /// Top-p (nucleus) + temperature.
    TopP { p: f32, temperature: f32 },
    /// Mirostat (adaptive temperature).
    Mirostat { tau: f32, eta: f32 },
    /// Locally typical.
    LocallyTypical { p: f32, temperature: f32 },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::TopP {
            p: 0.9,
            temperature: 0.7,
        }
    }
}

/// Composable generation pipeline: processors + sampling strategy.
pub struct GenerationPipeline {
    engine: InferenceEngine,
    processors: ProcessorChain,
    strategy: SamplingStrategy,
    max_tokens: usize,
    eos_token_id: Option<usize>,
}

impl GenerationPipeline {
    pub fn new(model: BitNetModel) -> Self {
        Self {
            engine: InferenceEngine::new(model),
            processors: ProcessorChain::new(),
            strategy: SamplingStrategy::default(),
            max_tokens: 100,
            eos_token_id: Some(2),
        }
    }

    pub fn add_processor<P: LogitProcessor + 'static>(mut self, p: P) -> Self {
        self.processors.add_boxed(Box::new(p));
        self
    }

    pub fn chain(mut self, chain: ProcessorChain) -> Self {
        self.processors = chain;
        self
    }

    pub fn with_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn eos_token_id(mut self, id: Option<usize>) -> Self {
        self.eos_token_id = id;
        self
    }

    /// Generate token IDs from prompt using the pipeline.
    pub fn generate(&self, prompt_ids: &[usize]) -> Result<Vec<usize>> {
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt_ids must not be empty");
        }
        let mut cache = self.engine.create_cache();
        let mut ids = Vec::with_capacity(prompt_ids.len() + self.max_tokens);
        let mut logits = Vec::new();

        for &token_id in prompt_ids {
            logits = self.engine.forward_step(token_id, &mut cache)?;
            ids.push(token_id);
        }

        let max_length = prompt_ids.len() + self.max_tokens;
        let mut mirostat_temp = 1.0f32;

        while ids.len() < max_length {
            self.processors.process(&mut logits, &ids)?;

            let next_id = match &self.strategy {
                SamplingStrategy::Greedy => argmax(&logits).unwrap_or(0),
                SamplingStrategy::TopK { k, temperature } => {
                    sample_top_k(&logits, *k, *temperature)?
                }
                SamplingStrategy::TopP { p, temperature } => {
                    sample_top_p(&logits, *p, *temperature)?
                }
                SamplingStrategy::Mirostat { tau, eta } => {
                    let (token, new_temp) = crate::inference::sampling::sample_mirostat(
                        &logits,
                        *tau,
                        *eta,
                        mirostat_temp,
                    )?;
                    mirostat_temp = new_temp;
                    token
                }
                SamplingStrategy::LocallyTypical { p, temperature } => {
                    crate::inference::sampling::sample_locally_typical(&logits, *p, *temperature)?
                }
            };

            ids.push(next_id);
            if self.eos_token_id == Some(next_id) {
                break;
            }
            logits = self.engine.forward_step(next_id, &mut cache)?;
        }
        Ok(ids)
    }
}

fn sample_top_k(logits: &[f32], k: usize, temperature: f32) -> Result<usize> {
    let k = k.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temperature.max(1e-6)))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    let max_ln = indexed
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = indexed.iter().map(|(_, v)| (v - max_ln).exp()).sum();
    let r: f32 = rand::thread_rng().gen();
    let mut cum = 0.0f32;
    for (idx, v) in &indexed {
        cum += (v - max_ln).exp() / sum;
        if r <= cum {
            return Ok(*idx);
        }
    }
    Ok(indexed.last().map(|(i, _)| *i).unwrap_or(0))
}

fn sample_top_p(logits: &[f32], p: f32, temperature: f32) -> Result<usize> {
    let t = temperature.max(1e-6);
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / t))
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
    let r: f32 = rand::thread_rng().gen();
    let mut cum = 0.0f32;
    for (i, &pr) in probs[..top_n].iter().enumerate() {
        cum += pr / sum_top;
        if r <= cum {
            return Ok(indexed[i].0);
        }
    }
    Ok(indexed[top_n - 1].0)
}

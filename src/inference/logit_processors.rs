//! Logit processors: composable transformations before sampling.
//!
//! Processors can be chained: logits → proc1 → proc2 → proc3 → sample.

use anyhow::Result;
use std::collections::HashMap;

/// Process logits in place before sampling (e.g. temperature, penalties, masking).
pub trait LogitProcessor: Send + Sync {
    /// Apply the processor. May modify `logits` and use `generated` for context.
    fn process(&self, logits: &mut [f32], generated: &[usize]) -> Result<()>;
}

/// Temperature scaling: logits /= temperature.
#[derive(Clone, Debug)]
pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl TemperatureProcessor {
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature: temperature.max(1e-6),
        }
    }
}

impl LogitProcessor for TemperatureProcessor {
    fn process(&self, logits: &mut [f32], _generated: &[usize]) -> Result<()> {
        let t = self.temperature.max(1e-6);
        for x in logits.iter_mut() {
            *x /= t;
        }
        Ok(())
    }
}

/// Repetition penalty (transformers-style): scale down logits of already-generated tokens.
#[derive(Clone, Debug)]
pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
}

impl RepetitionPenaltyProcessor {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitProcessor for RepetitionPenaltyProcessor {
    fn process(&self, logits: &mut [f32], generated: &[usize]) -> Result<()> {
        if self.penalty <= 0.0 || (self.penalty - 1.0).abs() < 1e-6 {
            return Ok(());
        }
        for &tid in generated {
            if tid < logits.len() {
                let v = logits[tid];
                logits[tid] = if v > 0.0 {
                    v / self.penalty
                } else {
                    v * self.penalty
                };
            }
        }
        Ok(())
    }
}

/// Top-k: zero out logits outside the top k.
#[derive(Clone, Debug)]
pub struct TopKProcessor {
    pub k: usize,
}

impl TopKProcessor {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl LogitProcessor for TopKProcessor {
    fn process(&self, logits: &mut [f32], _generated: &[usize]) -> Result<()> {
        let k = self.k.min(logits.len());
        if k >= logits.len() {
            return Ok(());
        }
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, _) in indexed.iter().skip(k) {
            logits[*i] = f32::NEG_INFINITY;
        }
        Ok(())
    }
}

/// Top-p (nucleus): zero out logits outside the smallest set with cumulative prob >= p.
#[derive(Clone, Debug)]
pub struct TopPProcessor {
    pub p: f32,
}

impl TopPProcessor {
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl LogitProcessor for TopPProcessor {
    fn process(&self, logits: &mut [f32], _generated: &[usize]) -> Result<()> {
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max_ln = indexed
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_ln).exp()).collect();
        let sum_all: f32 = exp.iter().sum();
        if sum_all <= 0.0 {
            return Ok(());
        }
        let probs: Vec<f32> = exp.iter().map(|e| e / sum_all).collect();
        let mut cum = 0.0f32;
        let mut n = 0;
        for (i, &pr) in probs.iter().enumerate() {
            cum += pr;
            n = i + 1;
            if cum >= self.p {
                break;
            }
        }
        let keep = n.max(1);
        for (i, _) in indexed.iter().skip(keep) {
            logits[*i] = f32::NEG_INFINITY;
        }
        Ok(())
    }
}

/// Min-p: zero out logits with probability below min_p * max_prob.
#[derive(Clone, Debug)]
pub struct MinPProcessor {
    pub min_p: f32,
}

impl MinPProcessor {
    pub fn new(min_p: f32) -> Self {
        Self { min_p }
    }
}

impl LogitProcessor for MinPProcessor {
    fn process(&self, logits: &mut [f32], _generated: &[usize]) -> Result<()> {
        let max_ln = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&v| (v - max_ln).exp()).sum();
        if exp_sum <= 0.0 {
            return Ok(());
        }
        let max_prob = 1.0 / exp_sum;
        let threshold = self.min_p * max_prob;
        let indexed: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, (v - max_ln).exp() / exp_sum))
            .collect();
        for (i, prob) in indexed {
            if prob < threshold {
                logits[i] = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

/// Bad words: set logits of given token ids to negative infinity.
#[derive(Clone, Debug)]
pub struct BadWordsProcessor {
    pub token_ids: Vec<usize>,
}

impl BadWordsProcessor {
    pub fn new(token_ids: Vec<usize>) -> Self {
        Self { token_ids }
    }
}

impl LogitProcessor for BadWordsProcessor {
    fn process(&self, logits: &mut [f32], _generated: &[usize]) -> Result<()> {
        for &tid in &self.token_ids {
            if tid < logits.len() {
                logits[tid] = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

/// Forced tokens: zero out all logits except the given token id (or list for position).
#[derive(Clone, Debug)]
pub struct ForcedTokensProcessor {
    /// If Some(id), force this token at the next position; None = no force.
    pub next_forced: Option<usize>,
}

impl ForcedTokensProcessor {
    pub fn new(next_forced: Option<usize>) -> Self {
        Self { next_forced }
    }
}

impl LogitProcessor for ForcedTokensProcessor {
    fn process(&self, logits: &mut [f32], _generated: &[usize]) -> Result<()> {
        if let Some(id) = self.next_forced {
            if id < logits.len() {
                let max_ln = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for (i, x) in logits.iter_mut().enumerate() {
                    *x = if i == id { max_ln } else { f32::NEG_INFINITY };
                }
            }
        }
        Ok(())
    }
}

/// Frequency penalty: subtract count * penalty from each token's logit.
#[derive(Clone, Debug)]
pub struct FrequencyPenaltyProcessor {
    pub penalty: f32,
}

impl FrequencyPenaltyProcessor {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitProcessor for FrequencyPenaltyProcessor {
    fn process(&self, logits: &mut [f32], generated: &[usize]) -> Result<()> {
        if self.penalty == 0.0 {
            return Ok(());
        }
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &tid in generated {
            *counts.entry(tid).or_insert(0) += 1;
        }
        for (tid, count) in counts {
            if tid < logits.len() {
                logits[tid] -= self.penalty * count as f32;
            }
        }
        Ok(())
    }
}

/// Chain of processors: applied in order.
pub struct ProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}

impl ProcessorChain {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    pub fn add_processor<P: LogitProcessor + 'static>(mut self, p: P) -> Self {
        self.processors.push(Box::new(p));
        self
    }

    /// Add an already-boxed processor (for building chains from pipeline).
    pub fn add_boxed(&mut self, p: Box<dyn LogitProcessor>) {
        self.processors.push(p);
    }

    pub fn process(&self, logits: &mut [f32], generated: &[usize]) -> Result<()> {
        for p in &self.processors {
            p.process(logits, generated)?;
        }
        Ok(())
    }
}

impl Default for ProcessorChain {
    fn default() -> Self {
        Self::new()
    }
}

//! Speculative decoding: 2–3x faster generation using a small draft model and a target model.

use crate::inference::engine::InferenceEngine;
use crate::utils::argmax;
use anyhow::Result;

/// Speculative decoder: draft model proposes K tokens, target model verifies in one batch.
pub struct SpeculativeDecoder {
    draft_engine: InferenceEngine,
    target_engine: InferenceEngine,
    num_speculative_tokens: usize,
}

impl SpeculativeDecoder {
    pub fn new(
        draft_engine: InferenceEngine,
        target_engine: InferenceEngine,
        num_speculative_tokens: usize,
    ) -> Result<Self> {
        if num_speculative_tokens == 0 {
            anyhow::bail!("num_speculative_tokens must be > 0");
        }
        Ok(Self {
            draft_engine,
            target_engine,
            num_speculative_tokens,
        })
    }

    /// Generate tokens using speculative decoding: draft proposes K tokens, target verifies.
    pub fn generate_speculative(
        &self,
        prompt_ids: &[usize],
        max_tokens: usize,
    ) -> Result<Vec<usize>> {
        if prompt_ids.is_empty() {
            anyhow::bail!("prompt_ids must not be empty");
        }
        let mut generated = prompt_ids.to_vec();
        let k = self.num_speculative_tokens;

        while generated.len() < max_tokens {
            // 1. Draft model: generate K candidate tokens (fast, with KV cache)
            let draft_tokens = self.draft_generate_fast(&generated, k)?;
            if draft_tokens.is_empty() {
                let next = self.target_engine.generate_one(&generated)?;
                generated.push(next);
                continue;
            }

            // 2. Target model: verify draft tokens in one batch
            let verified = self.target_engine.verify_batch(&generated, &draft_tokens)?;

            generated.extend_from_slice(&verified);

            if verified.len() < draft_tokens.len() {
                // Rejection: get one token from target and continue
                let token = self.target_engine.generate_one(&generated)?;
                generated.push(token);
            }
        }

        Ok(generated)
    }

    /// Draft model generates up to K tokens greedily using KV cache (fast path).
    fn draft_generate_fast(&self, context: &[usize], k: usize) -> Result<Vec<usize>> {
        let mut cache = self.draft_engine.create_cache();
        let mut logits = Vec::new();
        for &token_id in context {
            logits = self.draft_engine.forward_step(token_id, &mut cache)?;
        }
        let mut out = Vec::with_capacity(k);
        for _ in 0..k {
            let next_id = argmax(&logits).unwrap_or(0);
            out.push(next_id);
            logits = self.draft_engine.forward_step(next_id, &mut cache)?;
        }
        Ok(out)
    }
}

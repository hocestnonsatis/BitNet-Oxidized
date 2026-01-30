//! Structured output: JSON mode with schema-guided generation.

use crate::inference::engine::InferenceEngine;
use crate::tokenizer::BitNetTokenizer;
use crate::utils::argmax;
use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;

/// JSON schema for constrained generation (type hints; full validation is best-effort).
#[derive(Clone, Debug)]
pub enum JsonSchema {
    Object,
    Array,
    String,
    Number,
    Boolean,
}

/// Generator that produces parseable JSON by stopping when valid JSON is detected.
pub struct StructuredGenerator {
    engine: InferenceEngine,
    tokenizer: Option<Arc<BitNetTokenizer>>,
}

impl StructuredGenerator {
    pub fn new(engine: InferenceEngine, tokenizer: Option<Arc<BitNetTokenizer>>) -> Self {
        Self { engine, tokenizer }
    }

    /// Generate token IDs until decoded text parses as valid JSON, then return the value.
    /// Requires tokenizer. Uses greedy decoding. Stops at first valid JSON or max_tokens.
    pub fn generate_json(&self, prompt_ids: &[usize], max_tokens: usize) -> Result<Value> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("generate_json requires a tokenizer"))?;

        if prompt_ids.is_empty() {
            anyhow::bail!("prompt_ids must not be empty");
        }

        let mut cache = self.engine.create_cache();
        let mut ids = prompt_ids.to_vec();
        let prompt_len = prompt_ids.len();

        let mut logits = Vec::new();
        for &token_id in prompt_ids {
            logits = self.engine.forward_step(token_id, &mut cache)?;
        }

        for _ in 0..max_tokens {
            let next_id = argmax(&logits).unwrap_or(0);
            ids.push(next_id);
            logits = self.engine.forward_step(next_id, &mut cache)?;

            let suffix = &ids[prompt_len..];
            let text = tokenizer.decode(suffix).unwrap_or_default();
            if let Ok(v) = serde_json::from_str::<Value>(text.trim()) {
                return Ok(v);
            }
        }

        let text = tokenizer.decode(&ids[prompt_len..]).unwrap_or_default();
        serde_json::from_str(text.trim())
            .map_err(|e| anyhow::anyhow!("no valid JSON within max_tokens: {}", e))
    }

    /// Generate raw token IDs with JSON-mode stop (stop when valid JSON detected).
    pub fn generate_json_ids(&self, prompt_ids: &[usize], max_tokens: usize) -> Result<Vec<usize>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("generate_json_ids requires a tokenizer"))?;

        if prompt_ids.is_empty() {
            anyhow::bail!("prompt_ids must not be empty");
        }

        let mut cache = self.engine.create_cache();
        let mut ids = prompt_ids.to_vec();
        let prompt_len = prompt_ids.len();

        let mut logits = Vec::new();
        for &token_id in prompt_ids {
            logits = self.engine.forward_step(token_id, &mut cache)?;
        }

        for _ in 0..max_tokens {
            let next_id = argmax(&logits).unwrap_or(0);
            ids.push(next_id);
            logits = self.engine.forward_step(next_id, &mut cache)?;

            let suffix = &ids[prompt_len..];
            let text = tokenizer.decode(suffix).unwrap_or_default();
            if serde_json::from_str::<Value>(text.trim()).is_ok() {
                return Ok(ids);
            }
        }
        Ok(ids)
    }
}

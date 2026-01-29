//! Tokenizer integration for encode/decode.

use crate::errors::BitNetError;
use std::path::Path;

/// Wrapper around HuggingFace tokenizers for BitNet text I/O.
pub struct BitNetTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl BitNetTokenizer {
    /// Load tokenizer from a JSON file (e.g. tokenizer.json).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, BitNetError> {
        let tokenizer = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| BitNetError::Tokenizer(e.to_string()))?;
        Ok(Self { tokenizer })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, BitNetError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| BitNetError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[usize]) -> Result<String, BitNetError> {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        let s = self
            .tokenizer
            .decode(&ids_u32, true)
            .map_err(|e| BitNetError::Tokenizer(e.to_string()))?;
        Ok(s)
    }
}

//! Central error types for bitnet-oxidized.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum BitNetError {
    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("GGUF version {0} not supported")]
    UnsupportedGGUFVersion(u32),

    #[error("Out of memory: requested {requested} MB, available {available} MB")]
    OutOfMemory { requested: usize, available: usize },

    #[error("Token {0} out of vocabulary range")]
    InvalidToken(usize),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

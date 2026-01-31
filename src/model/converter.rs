//! Model format conversion: safetensors/PyTorch to GGUF, GGUF version upgrade.
//!
//! BitNet models are typically converted from Hugging Face via Python scripts;
//! this module provides Rust-side helpers and version upgrade.

use crate::model::gguf;
use crate::BitNetError;
use std::path::Path;

/// Convert Hugging Face safetensors to GGUF (BitNet I2_S or F32).
///
/// For full Hugging Face → GGUF conversion use the Python script:
/// `scripts/convert_huggingface_to_gguf.py`.
/// This stub returns an error with instructions.
pub fn convert_safetensors_to_gguf(
    _input: impl AsRef<Path>,
    _output: impl AsRef<Path>,
    _format_i2s: bool,
) -> Result<(), BitNetError> {
    Err(BitNetError::InvalidFormat(
        "safetensors→GGUF: use scripts/convert_huggingface_to_gguf.py for Hugging Face models"
            .into(),
    ))
}

/// Convert PyTorch .bin / .pt to GGUF.
///
/// PyTorch checkpoint format is typically handled in Python.
/// This stub returns an error with instructions.
pub fn convert_pytorch_to_gguf(
    _input: impl AsRef<Path>,
    _output: impl AsRef<Path>,
) -> Result<(), BitNetError> {
    Err(BitNetError::InvalidFormat(
        "PyTorch→GGUF: use scripts/convert_huggingface_to_gguf.py or export to safetensors first"
            .into(),
    ))
}

/// Upgrade GGUF file from v2 to v3: load and re-save with correct alignment.
/// Preserves all metadata and tensor data; normalizes to 32-byte alignment.
pub fn upgrade_gguf_version(
    path_in: impl AsRef<Path>,
    path_out: impl AsRef<Path>,
) -> Result<(), BitNetError> {
    gguf::repair_gguf(path_in, path_out)
}

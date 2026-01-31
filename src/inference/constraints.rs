//! Generation constraints: lexical, format, and allowed-token constraints.
//!
//! Use with logit processors (e.g. force token) or with the pipeline.

use std::collections::HashSet;

/// Lexical constraint: generation must include these token ids (in order) at some point.
/// Implemented by masking logits until the required token is chosen (simplified: force next if in sequence).
#[derive(Clone, Debug)]
pub struct LexicalConstraint {
    /// Token ids that must appear in order.
    pub required_ids: Vec<usize>,
}

impl LexicalConstraint {
    pub fn new(required_ids: Vec<usize>) -> Self {
        Self { required_ids }
    }

    /// If we are currently expecting the next required token, return it; else None.
    pub fn next_forced(&self, position_in_sequence: usize) -> Option<usize> {
        self.required_ids.get(position_in_sequence).copied()
    }
}

/// Allowed-tokens constraint: only these token ids are valid at the next step.
/// Use with a logit processor that masks others (e.g. set logits to -inf for disallowed).
#[derive(Clone, Debug)]
pub struct AllowedTokensConstraint {
    pub allowed: HashSet<usize>,
}

impl AllowedTokensConstraint {
    pub fn new(allowed: impl IntoIterator<Item = usize>) -> Self {
        Self {
            allowed: allowed.into_iter().collect(),
        }
    }

    /// Mask logits in place: set logits[i] = -inf for i not in allowed.
    pub fn apply(&self, logits: &mut [f32]) {
        for (i, x) in logits.iter_mut().enumerate() {
            if !self.allowed.contains(&i) {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

/// Format constraint: require output to be valid JSON (high-level; actual validation in StructuredGenerator).
#[derive(Clone, Debug)]
pub struct FormatConstraint {
    pub format: OutputFormat,
}

#[derive(Clone, Debug)]
pub enum OutputFormat {
    Json,
    Xml,
}

impl FormatConstraint {
    pub fn json() -> Self {
        Self {
            format: OutputFormat::Json,
        }
    }

    pub fn xml() -> Self {
        Self {
            format: OutputFormat::Xml,
        }
    }
}

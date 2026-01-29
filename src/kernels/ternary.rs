//! Ternary weight system for 1.58-bit quantization.
//!
//! Packs 4 ternary weights per byte using 2 bits each:
//! - 00 = Zero
//! - 01 = PlusOne
//! - 11 = MinusOne
//! - 10 = Unused (mapped to Zero)

use std::fmt;

/// Ternary weight value: -1, 0, or +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryWeight {
    /// Weight value -1
    MinusOne,
    /// Weight value 0
    Zero,
    /// Weight value +1
    PlusOne,
}

impl TernaryWeight {
    /// Convert to f32 for computation.
    #[inline]
    pub fn to_f32(self) -> f32 {
        match self {
            TernaryWeight::MinusOne => -1.0,
            TernaryWeight::Zero => 0.0,
            TernaryWeight::PlusOne => 1.0,
        }
    }

    /// Quantize an f32 value to ternary using threshold at 0.
    /// Values > 0 → PlusOne, < 0 → MinusOne, == 0 → Zero.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        if value > 0.0 {
            TernaryWeight::PlusOne
        } else if value < 0.0 {
            TernaryWeight::MinusOne
        } else {
            TernaryWeight::Zero
        }
    }

    /// Encode to 2-bit pattern for packing.
    #[inline]
    pub fn to_bits(self) -> u8 {
        match self {
            TernaryWeight::Zero => 0b00,
            TernaryWeight::PlusOne => 0b01,
            TernaryWeight::MinusOne => 0b11,
        }
    }

    /// Decode from 2-bit pattern. 0b10 is unused and maps to Zero.
    #[inline]
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => TernaryWeight::Zero,
            0b01 => TernaryWeight::PlusOne,
            0b11 => TernaryWeight::MinusOne,
            _ => TernaryWeight::Zero, // 0b10 unused
        }
    }
}

impl fmt::Display for TernaryWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

/// Dense tensor of ternary weights packed 4 per byte.
///
/// Layout: byte = [W3 W2 W1 W0], each W is 2 bits (LSB first: W0 in bits 0-1).
#[derive(Clone)]
pub struct TernaryTensor {
    /// Raw packed data: ceil(num_elements / 4) bytes.
    data: Vec<u8>,
    /// Total number of elements (not bytes).
    len: usize,
}

impl TernaryTensor {
    /// Create a new tensor of the given length, all zeros.
    pub fn zeros(len: usize) -> Self {
        let num_bytes = len.div_ceil(4);
        Self {
            data: vec![0u8; num_bytes],
            len,
        }
    }

    /// Create from raw packed bytes. `len` is the number of elements (not bytes).
    pub fn from_raw(data: Vec<u8>, len: usize) -> Result<Self, TernaryTensorError> {
        let expected_bytes = len.div_ceil(4);
        if data.len() < expected_bytes {
            return Err(TernaryTensorError::InvalidLength {
                expected: expected_bytes,
                got: data.len(),
            });
        }
        Ok(Self { data, len })
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw packed data (for LUT kernel). Length is ceil(len/4).
    #[inline]
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }

    /// Get weight at index as f32.
    #[inline]
    pub fn get(&self, index: usize) -> f32 {
        if index >= self.len {
            return 0.0;
        }
        let byte_idx = index / 4;
        let pos = index % 4;
        let bits = (self.data[byte_idx] >> (pos * 2)) & 0b11;
        TernaryWeight::from_bits(bits).to_f32()
    }

    /// Get weight at index as TernaryWeight.
    #[inline]
    pub fn get_ternary(&self, index: usize) -> TernaryWeight {
        if index >= self.len {
            return TernaryWeight::Zero;
        }
        let byte_idx = index / 4;
        let pos = index % 4;
        let bits = (self.data[byte_idx] >> (pos * 2)) & 0b11;
        TernaryWeight::from_bits(bits)
    }

    /// Set weight at index from f32 (quantizes to ternary).
    #[inline]
    pub fn set(&mut self, index: usize, value: f32) {
        if index >= self.len {
            return;
        }
        let w = TernaryWeight::from_f32(value);
        let byte_idx = index / 4;
        let pos = index % 4;
        let shift = pos * 2;
        self.data[byte_idx] = (self.data[byte_idx] & !(0b11 << shift)) | (w.to_bits() << shift);
    }

    /// Set weight at index from TernaryWeight.
    #[inline]
    pub fn set_ternary(&mut self, index: usize, weight: TernaryWeight) {
        if index >= self.len {
            return;
        }
        let byte_idx = index / 4;
        let pos = index % 4;
        let shift = pos * 2;
        self.data[byte_idx] =
            (self.data[byte_idx] & !(0b11 << shift)) | (weight.to_bits() << shift);
    }

    /// Memory usage in bytes (packed storage only).
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.data.len()
    }

    /// Dequantize to full f32 vector (for comparison / debugging).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        (0..self.len).map(|i| self.get(i)).collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TernaryTensorError {
    #[error("Invalid length: expected at least {expected} bytes, got {got}")]
    InvalidLength { expected: usize, got: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_weight_roundtrip() {
        for &v in &[-1.0f32, 0.0, 1.0, 0.5, -0.3] {
            let t = TernaryWeight::from_f32(v);
            let back = t.to_f32();
            assert!(back == -1.0 || back == 0.0 || back == 1.0);
        }
    }

    #[test]
    fn tensor_get_set() {
        let mut t = TernaryTensor::zeros(10);
        t.set(0, 1.0);
        t.set(1, -1.0);
        t.set(2, 0.0);
        assert_eq!(t.get(0), 1.0);
        assert_eq!(t.get(1), -1.0);
        assert_eq!(t.get(2), 0.0);
    }

    #[test]
    fn tensor_memory_usage() {
        let t = TernaryTensor::zeros(8);
        assert_eq!(t.memory_usage(), 2); // 8/4 = 2 bytes
        let t = TernaryTensor::zeros(9);
        assert_eq!(t.memory_usage(), 3);
    }
}

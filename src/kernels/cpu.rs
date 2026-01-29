//! Optimized CPU matrix-vector multiplication kernels for ternary weights.

#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;

use super::ternary::TernaryTensor;

/// Basic mat-vec: one dot product per output row, parallel over rows.
///
/// `weight` shape [out_features, in_features], row-major packed.
/// `input` [in_features], `output` [out_features].
pub fn mat_vec_mul_basic(weight: &TernaryTensor, input: &[f32], output: &mut [f32]) {
    let in_features = input.len();
    let out_features = output.len();
    debug_assert_eq!(weight.len(), out_features * in_features);

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(row, out_val)| {
            let mut sum = 0.0f32;
            let row_start = row * in_features;
            for col in 0..in_features {
                sum += weight.get(row_start + col) * input[col];
            }
            *out_val = sum;
        });
}

/// Blocked mat-vec: process 4 weights at a time, manual unrolling, better cache use.
pub fn mat_vec_mul_blocked(weight: &TernaryTensor, input: &[f32], output: &mut [f32]) {
    let in_features = input.len();
    let out_features = output.len();
    debug_assert_eq!(weight.len(), out_features * in_features);

    let cols4 = in_features - (in_features % 4);

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(row, out_val)| {
            let mut sum = 0.0f32;
            let row_start = row * in_features;

            // Process 4 at a time
            let mut j = 0usize;
            while j < cols4 {
                let b = row_start + j;
                sum += weight.get(b) * input[j]
                    + weight.get(b + 1) * input[j + 1]
                    + weight.get(b + 2) * input[j + 2]
                    + weight.get(b + 3) * input[j + 3];
                j += 4;
            }
            for col in cols4..in_features {
                sum += weight.get(row_start + col) * input[col];
            }
            *out_val = sum;
        });
}

/// LUT: 256 entries, each entry is [f32; 4] for the 4 decoded weights from one byte.
pub type ByteLut = [[f32; 4]; 256];

/// Build the lookup table: for each byte 0..256, decode 4 ternary weights.
pub fn build_lut() -> ByteLut {
    let mut lut: ByteLut = [[0.0; 4]; 256];
    for pattern in 0..256u32 {
        for pos in 0..4 {
            let bits = ((pattern >> (pos * 2)) & 0b11) as u8;
            lut[pattern as usize][pos] = match bits {
                0b00 => 0.0,
                0b01 => 1.0,
                0b11 => -1.0,
                _ => 0.0,
            };
        }
    }
    lut
}

/// LUT-based mat-vec: one table lookup per 4 input elements (fastest).
pub fn mat_vec_mul_lut(weight: &TernaryTensor, input: &[f32], output: &mut [f32]) {
    mat_vec_mul_lut_with(lut(), weight, input, output);
}

/// LUT-based mat-vec with a pre-built LUT (for benchmarks that want to avoid static).
pub fn mat_vec_mul_lut_with(
    lut: &ByteLut,
    weight: &TernaryTensor,
    input: &[f32],
    output: &mut [f32],
) {
    let in_features = input.len();
    let out_features = output.len();
    debug_assert_eq!(weight.len(), out_features * in_features);

    let raw = weight.raw_data();
    let cols4 = in_features / 4;

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(row, out_val)| {
            let mut sum = 0.0f32;
            let row_byte_start = row * in_features / 4;

            for chunk in 0..cols4 {
                let byte_idx = row_byte_start + chunk;
                let byte = raw[byte_idx] as usize;
                let w = &lut[byte];
                let j = chunk * 4;
                sum += w[0] * input[j]
                    + w[1] * input[j + 1]
                    + w[2] * input[j + 2]
                    + w[3] * input[j + 3];
            }

            // Remainder
            for col in (cols4 * 4)..in_features {
                sum += weight.get(row * in_features + col) * input[col];
            }
            *out_val = sum;
        });
}

use std::sync::OnceLock;

static LUT_GLOBAL: OnceLock<ByteLut> = OnceLock::new();

fn lut() -> &'static ByteLut {
    LUT_GLOBAL.get_or_init(build_lut)
}

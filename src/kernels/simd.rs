//! SIMD-accelerated mat-vec: AVX2 (x86_64) and NEON (aarch64) with LUT fallback.

#![allow(clippy::needless_range_loop)]

#[cfg(not(target_arch = "aarch64"))]
use super::cpu::mat_vec_mul_lut;
use super::ternary::TernaryTensor;

/// SIMD mat-vec: use AVX2 on x86_64, NEON on aarch64, else LUT.
pub fn mat_vec_mul_simd(weight: &TernaryTensor, input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { mat_vec_mul_avx2(weight, input, output) };
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { mat_vec_mul_neon(weight, input, output) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    mat_vec_mul_lut(weight, input, output);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mat_vec_mul_avx2(weight: &TernaryTensor, input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let in_features = input.len();
    let out_features = output.len();
    let raw = weight.raw_data();
    let row_bytes = in_features / 4;
    let cols8 = row_bytes / 2;

    for row in 0..out_features {
        let mut sum = _mm256_setzero_ps();
        let row_byte_start = row * row_bytes;

        for chunk in 0..cols8 {
            let b0 = raw[row_byte_start + chunk * 2];
            let b1 = raw[row_byte_start + chunk * 2 + 1];
            let w0 = decode_byte_lut(b0);
            let w1 = decode_byte_lut(b1);
            let j = chunk * 8;
            let inp = _mm256_loadu_ps(input[j..].as_ptr());
            let wv = _mm256_set_ps(w1[3], w1[2], w1[1], w1[0], w0[3], w0[2], w0[1], w0[0]);
            sum = _mm256_add_ps(_mm256_mul_ps(wv, inp), sum);
        }

        let mut sum_arr: [f32; 8] = std::mem::zeroed();
        _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum);
        let mut total = sum_arr.iter().sum::<f32>();

        for chunk in (cols8 * 2)..row_bytes {
            let byte_idx = row_byte_start + chunk;
            let byte = raw[byte_idx];
            let w = decode_byte_lut(byte);
            let j = chunk * 4;
            for (i, &wi) in w.iter().enumerate() {
                if j + i < in_features {
                    total += wi * input[j + i];
                }
            }
        }

        for col in (row_bytes * 4)..in_features {
            total += weight.get(row * in_features + col) * input[col];
        }
        output[row] = total;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::needless_range_loop, clippy::unnecessary_cast)]
fn decode_byte_lut(byte: u8) -> [f32; 4] {
    let mut w = [0.0f32; 4];
    for pos in 0..4 {
        let bits = (byte >> (pos * 2)) & 0b11;
        w[pos] = match bits {
            0b00 => 0.0,
            0b01 => 1.0,
            0b11 => -1.0,
            _ => 0.0,
        };
    }
    w
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mat_vec_mul_neon(weight: &TernaryTensor, input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    let in_features = input.len();
    let out_features = output.len();
    let raw = weight.raw_data();
    let cols4 = in_features / 4;

    for row in 0..out_features {
        let mut sum = vdupq_n_f32(0.0);
        let row_byte_start = row * in_features / 4;

        for chunk in 0..cols4 {
            let byte_idx = row_byte_start + chunk;
            let byte = raw[byte_idx];
            let w = decode_byte_lut_neon(byte);
            let j = chunk * 4;
            let inp = vld1q_f32(input[j..].as_ptr());
            sum = vmlaq_f32(sum, w, inp);
        }

        let mut sum_arr: [f32; 4] = std::mem::zeroed();
        vst1q_f32(sum_arr.as_mut_ptr(), sum);
        let mut total = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

        for col in (cols4 * 4)..in_features {
            total += weight.get(row * in_features + col) * input[col];
        }
        output[row] = total;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn decode_byte_lut_neon(byte: u8) -> std::arch::aarch64::float32x4_t {
    let mut w = [0.0f32; 4];
    for pos in 0..4 {
        let bits = (byte >> (pos * 2)) & 0b11;
        w[pos] = match bits {
            0b00 => 0.0,
            0b01 => 1.0,
            0b11 => -1.0,
            _ => 0.0,
        };
    }
    unsafe { std::arch::aarch64::vld1q_f32(w.as_ptr()) }
}

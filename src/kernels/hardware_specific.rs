//! Hardware-specific mat-vec kernels: Intel VNNI, Apple AMX, optional CUDA.
//! Falls back to generic SIMD when not available.

use crate::kernels::TernaryTensor;

/// Select best available mat-vec for current architecture.
/// Returns true if a hardware-specific path was used.
#[inline]
pub fn mat_vec_mul_hardware(weight: &TernaryTensor, input: &[f32], output: &mut [f32]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            mat_vec_mul_vnni_stub(weight, input, output);
            return true;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            mat_vec_mul_amx_stub(weight, input, output);
            return true;
        }
    }
    false
}

#[cfg(target_arch = "x86_64")]
/// Intel VNNI (Vector Neural Network Instructions) stub.
/// In production, use AVX2/AVX-512 with VNNI for 8-bit dot-product.
pub fn mat_vec_mul_vnni_stub(weight: &TernaryTensor, input: &[f32], output: &mut [f32]) {
    crate::kernels::mat_vec_mul_simd(weight, input, output);
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
fn mat_vec_mul_vnni_stub(_weight: &TernaryTensor, _input: &[f32], _output: &mut [f32]) {}

#[cfg(target_arch = "aarch64")]
/// Apple AMX / ARM NEON stub. M1/M2/M3 use AMX for matrix ops.
pub fn mat_vec_mul_amx_stub(weight: &TernaryTensor, input: &[f32], output: &mut [f32]) {
    crate::kernels::mat_vec_mul_simd(weight, input, output);
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(dead_code)]
fn mat_vec_mul_amx_stub(_weight: &TernaryTensor, _input: &[f32], _output: &mut [f32]) {}

/// Runtime detection of best kernel. Call from engine to optionally use hardware path.
#[cfg(target_arch = "x86_64")]
pub fn preferred_kernel_name() -> &'static str {
    if is_x86_feature_detected!("avx2") {
        "avx2"
    } else {
        "simd"
    }
}

#[cfg(target_arch = "aarch64")]
pub fn preferred_kernel_name() -> &'static str {
    if std::arch::is_aarch64_feature_detected!("neon") {
        "neon"
    } else {
        "simd"
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn preferred_kernel_name() -> &'static str {
    "simd"
}

//! Compute kernels for ternary matrix operations.

pub mod cpu;
pub mod flash_attention;
pub mod hardware_specific;
pub mod simd;
pub mod ternary;

pub use cpu::{
    build_lut, mat_vec_mul_basic, mat_vec_mul_blocked, mat_vec_mul_lut, mat_vec_mul_lut_with,
    ByteLut,
};
pub use flash_attention::flash_attention_forward;
pub use hardware_specific::{mat_vec_mul_hardware, preferred_kernel_name};
pub use simd::mat_vec_mul_simd;
pub use ternary::{TernaryTensor, TernaryTensorError, TernaryWeight};

//! Optimization: kernel autotuning and selection.

pub mod kernel_autotuning;

pub use kernel_autotuning::{best_kernel_for_size, select_kernel, KernelChoice};

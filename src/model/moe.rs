//! Mixture of Experts (MoE) layer: sparse activation over multiple BitNet-style experts.

use crate::kernels::{mat_vec_mul_simd, TernaryTensor};
use anyhow::Result;

/// Single expert: same structure as BitNet FFN (gate + up, then down).
#[derive(Clone)]
pub struct BitNetExpert {
    pub gate_proj: TernaryTensor,
    pub up_proj: TernaryTensor,
    pub down_proj: TernaryTensor,
}

impl BitNetExpert {
    pub fn forward(
        &self,
        hidden: &[f32],
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let mut gate = vec![0.0f32; intermediate_size];
        let mut up = vec![0.0f32; intermediate_size];
        mat_vec_mul_simd(&self.gate_proj, hidden, &mut gate);
        mat_vec_mul_simd(&self.up_proj, hidden, &mut up);
        for i in 0..intermediate_size {
            gate[i] = silu(gate[i]) * up[i];
        }
        let mut out = vec![0.0f32; hidden_size];
        mat_vec_mul_simd(&self.down_proj, &gate, &mut out);
        out
    }
}

/// MoE layer: router (gate) selects top-k experts, their outputs are combined by router weights.
#[derive(Clone)]
pub struct MoELayer {
    pub num_experts: usize,
    pub top_k: usize,
    pub gate: TernaryTensor,
    pub experts: Vec<BitNetExpert>,
    pub intermediate_size: usize,
    pub hidden_size: usize,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softmax_slice(in_out: &mut [f32]) {
    let max = in_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = in_out.iter().map(|&x| (x - max).exp()).sum();
    for x in in_out.iter_mut() {
        *x = (*x - max).exp() / sum;
    }
}

impl MoELayer {
    /// Forward: router selects top-k experts, run them, combine with softmax weights.
    pub fn forward(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let hidden_size = self.hidden_size;
        let mut router_logits = vec![0.0f32; self.num_experts];
        mat_vec_mul_simd(&self.gate, hidden, &mut router_logits);

        let (expert_indices, expert_weights) = self.top_k_routing(&router_logits)?;

        let mut output = vec![0.0f32; hidden_size];
        for (&idx, &weight) in expert_indices.iter().zip(expert_weights.iter()) {
            let expert_out = self.experts[idx].forward(hidden, self.intermediate_size, hidden_size);
            for (i, &val) in expert_out.iter().enumerate() {
                output[i] += weight * val;
            }
        }
        Ok(output)
    }

    fn top_k_routing(&self, logits: &[f32]) -> Result<(Vec<usize>, Vec<f32>)> {
        let k = self.top_k.min(self.num_experts).min(logits.len());
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        let indices: Vec<usize> = indexed.iter().map(|&(i, _)| i).collect();
        let mut weights: Vec<f32> = indexed.iter().map(|&(_, l)| l).collect();
        softmax_slice(&mut weights);
        Ok((indices, weights))
    }
}

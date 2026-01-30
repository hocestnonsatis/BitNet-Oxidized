//! Flash Attention: block-tiled causal attention. O(block_size) working memory per head.

/// Causal scaled dot-product attention using block tiling.
/// q, k, v: [seq_len, num_heads * head_dim]. Output: [seq_len, num_heads * head_dim].
/// block_size: tile size (e.g. 64 or 128). Smaller = less peak memory.
pub fn flash_attention_forward(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    let hidden_size = num_heads * head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    let block_size = block_size.min(seq_len).max(1);

    let mut out = vec![vec![0.0f32; hidden_size]; seq_len];
    let num_q_blocks = (seq_len + block_size - 1) / block_size;
    let num_kv_blocks = (seq_len + block_size - 1) / block_size;

    for h in 0..num_heads {
        let q_off = h * head_dim;
        let mut m_row: Vec<f32> = vec![f32::NEG_INFINITY; seq_len];
        let mut l_row: Vec<f32> = vec![0.0; seq_len];

        for qb in 0..num_q_blocks {
            let q_start = qb * block_size;
            let q_end = (q_start + block_size).min(seq_len);

            for kvb in 0..num_kv_blocks {
                let kv_start = kvb * block_size;
                let kv_end = (kv_start + block_size).min(seq_len);

                for qi in 0..(q_end - q_start) {
                    let t = q_start + qi;
                    let row_end = (t + 1).min(kv_end);
                    if kv_start >= row_end {
                        continue;
                    }
                    let row_start = kv_start.max(0);

                    let mut m_block = f32::NEG_INFINITY;
                    let mut p_vals: Vec<f32> = Vec::with_capacity(row_end - row_start);
                    let mut o_block = vec![0.0f32; head_dim];

                    for s in row_start..row_end {
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q[t][q_off + d] * k[s][q_off + d];
                        }
                        let score = score * scale;
                        m_block = m_block.max(score);
                        p_vals.push(score);
                    }
                    for (si, &p) in p_vals.iter().enumerate() {
                        let s = row_start + si;
                        let p = (p - m_block).exp();
                        for d in 0..head_dim {
                            o_block[d] += p * v[s][q_off + d];
                        }
                    }
                    let p_sum: f32 = p_vals.iter().map(|&p| (p - m_block).exp()).sum();
                    if p_sum <= 0.0 {
                        continue;
                    }

                    let m_old = m_row[t];
                    let m_new = m_old.max(m_block);
                    let l_old = l_row[t];
                    let correction_old = (m_old - m_new).exp();
                    let correction_new = (m_block - m_new).exp();
                    l_row[t] = l_old * correction_old + p_sum * correction_new;
                    for d in 0..head_dim {
                        out[t][q_off + d] =
                            out[t][q_off + d] * correction_old + (o_block[d] * correction_new);
                    }
                    m_row[t] = m_new;
                }
            }
        }

        for t in 0..seq_len {
            let l = l_row[t];
            if l > 0.0 {
                for d in 0..head_dim {
                    out[t][q_off + d] /= l;
                }
            }
        }
    }

    out
}

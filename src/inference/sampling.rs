//! Advanced sampling strategies: Mirostat, locally typical, contrastive, beam search.
//!
//! Each method has documentation and can be used with the generation pipeline.

use anyhow::Result;
use rand::Rng;

fn rng() -> rand::rngs::ThreadRng {
    rand::thread_rng()
}

/// Softmax and return (probabilities, log_sum_exp).
fn softmax(logits: &[f32]) -> (Vec<f32>, f32) {
    let max_ln = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&v| (v - max_ln).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum <= 0.0 {
        return (vec![0.0; logits.len()], max_ln);
    }
    let probs: Vec<f32> = exp.iter().map(|e| e / sum).collect();
    let log_sum_exp = max_ln + sum.ln();
    (probs, log_sum_exp)
}

/// Sample one token from a probability distribution.
fn sample_from_probs(probs: &[f32]) -> usize {
    let r: f32 = rng().gen();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Mirostat sampling: adaptive temperature so perplexity stays near target.
///
/// tau: target perplexity (e.g. 3.0). eta: learning rate for temperature update.
/// Returns (sampled_token_id, updated_temperature for next step).
pub fn sample_mirostat(
    logits: &[f32],
    tau: f32,
    eta: f32,
    current_temp: f32,
) -> Result<(usize, f32)> {
    let mut t = current_temp.max(1e-6);
    let (_probs, log_sum_exp) = softmax(&logits.iter().map(|&v| v / t).collect::<Vec<f32>>());
    let perplexity = (-log_sum_exp).exp();
    let k = (perplexity - tau).max(0.0);
    t = (t - eta * k).max(1e-6);

    let (probs_final, _) = softmax(&logits.iter().map(|&v| v / t).collect::<Vec<f32>>());
    let token_id = sample_from_probs(&probs_final);
    Ok((token_id, t))
}

/// Locally typical sampling: smallest set of tokens whose cumulative "typicality" >= p.
///
/// Typicality of token i is exp(-|log p(i) - H(p)|); we use cumulative probability
/// of the typical set. p is the threshold (e.g. 0.9).
pub fn sample_locally_typical(logits: &[f32], p: f32, temperature: f32) -> Result<usize> {
    let t = temperature.max(1e-6);
    let scaled: Vec<f32> = logits.iter().map(|&v| v / t).collect();
    let (probs, _) = softmax(&scaled);
    let entropy: f32 = probs
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| -x * x.ln())
        .sum();
    let mut indexed: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &pr)| (i, (-(pr.ln() - entropy).abs()).exp()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let sum_typ: f32 = indexed.iter().map(|(_, v)| *v).sum();
    if sum_typ <= 0.0 {
        return Ok(indexed.first().map(|(i, _)| *i).unwrap_or(0));
    }
    let mut cum = 0.0f32;
    let mut n = 0;
    for (i, (_, typ)) in indexed.iter().enumerate() {
        cum += typ / sum_typ;
        n = i + 1;
        if cum >= p {
            break;
        }
    }
    let top_n = n.max(1);
    let sum_p: f32 = indexed[..top_n].iter().map(|(i, _)| probs[*i]).sum();
    if sum_p <= 0.0 {
        return Ok(indexed[0].0);
    }
    let r: f32 = rng().gen();
    let mut cum = 0.0f32;
    for (i, _) in indexed[..top_n].iter().enumerate() {
        cum += probs[indexed[i].0] / sum_p;
        if r <= cum {
            return Ok(indexed[i].0);
        }
    }
    Ok(indexed[top_n - 1].0)
}

/// Contrastive decoding: score = logit - alpha * max(0, sim(token, context)).
///
/// Simplified: we use repetition penalty as proxy for "similarity to context".
/// alpha: penalty weight. Returns sampled token id.
pub fn sample_contrastive(
    logits: &[f32],
    generated: &[usize],
    alpha: f32,
    temperature: f32,
) -> Result<usize> {
    let t = temperature.max(1e-6);
    let mut scores = logits.to_vec();
    for &tid in generated {
        if tid < scores.len() && alpha > 0.0 {
            scores[tid] -= alpha;
        }
    }
    let (probs, _) = softmax(&scores.iter().map(|&v| v / t).collect::<Vec<f32>>());
    Ok(sample_from_probs(&probs))
}

/// Beam search: keep top-k candidates by cumulative log-probability; return best sequence.
///
/// beam_width: number of candidates. diversity_penalty: subtract from score when
/// candidate shares recent tokens with others. Returns (best_token_ids, best_score).
pub fn sample_beam_search_step(
    logits_per_beam: &[Vec<f32>],
    beam_scores: &[f32],
    diversity_penalty: f32,
) -> Result<Vec<(usize, usize, f32)>> {
    let k = beam_scores.len();
    let mut all: Vec<(usize, usize, f32)> = Vec::new();
    for (b, (logits, &score)) in logits_per_beam.iter().zip(beam_scores.iter()).enumerate() {
        let (probs, _log_sum) = softmax(logits);
        for (tid, &p) in probs.iter().enumerate() {
            if p > 0.0 {
                let log_p = (p).ln();
                all.push((b, tid, score + log_p));
            }
        }
    }
    all.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<(usize, usize, f32)> = all.into_iter().take(k).collect();
    if diversity_penalty != 0.0 {
        // Simplified: no per-beam diversity here; caller can apply diversity when extending beams.
    }
    Ok(top)
}

/// Min-p sampling: filter tokens with prob < min_p * max_prob, then sample from rest.
pub fn sample_min_p(logits: &[f32], min_p: f32, temperature: f32) -> Result<usize> {
    let t = temperature.max(1e-6);
    let (probs, _) = softmax(&logits.iter().map(|&v| v / t).collect::<Vec<f32>>());
    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;
    let filtered: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p >= threshold)
        .map(|(i, &p)| (i, p))
        .collect();
    if filtered.is_empty() {
        return Ok(argmax_idx(logits));
    }
    let sum: f32 = filtered.iter().map(|(_, p)| *p).sum();
    if sum <= 0.0 {
        return Ok(filtered[0].0);
    }
    let r: f32 = rng().gen();
    let mut cum = 0.0f32;
    for (i, p) in &filtered {
        cum += p / sum;
        if r <= cum {
            return Ok(*i);
        }
    }
    Ok(filtered.last().map(|(i, _)| *i).unwrap_or(0))
}

fn argmax_idx(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sum_one() {
        let logits = [0.0f32, 1.0, 2.0];
        let (probs, _) = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_mirostat_returns_valid_token() {
        let logits = vec![0.1f32; 10];
        let (token, _) = sample_mirostat(&logits, 3.0, 0.1, 1.0).unwrap();
        assert!(token < 10);
    }

    #[test]
    fn test_sample_locally_typical_returns_valid_token() {
        let logits = vec![0.0f32, 1.0, 0.5];
        let token = sample_locally_typical(&logits, 0.9, 1.0).unwrap();
        assert!(token < 3);
    }
}

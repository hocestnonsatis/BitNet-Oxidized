//! Streaming text generation via async channels.

use crate::{BitNetTokenizer, TextGenerator};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// EOS token id (configurable; many models use 2).
pub const EOS_TOKEN: usize = 2;

/// Event emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum GenerationToken {
    Token { id: usize, text: String },
    Done { total_tokens: usize, time_ms: f64 },
    Error(String),
}

/// Async stream generator: runs generation and sends tokens over a channel.
pub struct StreamGenerator {
    generator: Arc<tokio::sync::RwLock<TextGenerator>>,
    tokenizer: Option<Arc<BitNetTokenizer>>,
}

impl StreamGenerator {
    pub fn new(
        generator: Arc<tokio::sync::RwLock<TextGenerator>>,
        tokenizer: Option<Arc<BitNetTokenizer>>,
    ) -> Self {
        Self {
            generator,
            tokenizer,
        }
    }

    /// Start streaming generation; returns a receiver for GenerationToken events.
    /// Runs generation on the current task (blocks until done); use tokio::task::spawn if you need to avoid blocking.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> mpsc::Receiver<GenerationToken> {
        let (tx, rx) = mpsc::channel(128);
        let prompt_ids = if let Some(ref t) = self.tokenizer {
            t.encode(prompt)
                .unwrap_or_else(|_| simple_tokenizer(prompt))
        } else {
            simple_tokenizer(prompt)
        };

        if prompt_ids.is_empty() {
            let _ = tx
                .send(GenerationToken::Done {
                    total_tokens: 0,
                    time_ms: 0.0,
                })
                .await;
            return rx;
        }

        let max_tokens = max_tokens.min(256);
        let target_len = prompt_ids.len() + max_tokens;
        let gen = Arc::clone(&self.generator);
        let tok = self.tokenizer.clone();

        tokio::task::spawn(async move {
            let start = Instant::now();
            let mut ids = prompt_ids;

            for _ in 0..(target_len - ids.len()) {
                let out = {
                    let g = gen.read().await;
                    g.generate_top_p(&ids, ids.len() + 1, 0.9, temperature.max(1e-6))
                };

                match out {
                    Ok(out) => {
                        let next_id = *out.last().unwrap_or(&0);
                        ids = out;
                        let text = if let Some(ref t) = tok {
                            t.decode(&[next_id]).unwrap_or_else(|_| next_id.to_string())
                        } else {
                            next_id.to_string()
                        };
                        if tx
                            .send(GenerationToken::Token { id: next_id, text })
                            .await
                            .is_err()
                        {
                            break;
                        }
                        if next_id == EOS_TOKEN {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(GenerationToken::Error(e.to_string())).await;
                        break;
                    }
                }
            }

            let time_ms = start.elapsed().as_secs_f64() * 1000.0;
            let _ = tx
                .send(GenerationToken::Done {
                    total_tokens: ids.len(),
                    time_ms,
                })
                .await;
        });

        rx
    }
}

fn simple_tokenizer(prompt: &str) -> Vec<usize> {
    const VOCAB: usize = 256;
    prompt
        .split_whitespace()
        .map(|s| {
            let h = s.bytes().fold(0u64, |a, b| a.wrapping_add(b as u64));
            (h as usize) % VOCAB
        })
        .collect()
}

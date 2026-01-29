//! Dynamic batching: collect requests and process in batches with timeout.

use crate::InferenceEngine;
use anyhow::Result;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::interval;

/// A single request to the batcher.
pub struct BatchRequest {
    pub id: String,
    pub prompt_ids: Vec<usize>,
    pub max_tokens: usize,
    pub response_tx: oneshot::Sender<Result<Vec<Vec<f32>>>>,
}

/// Dynamic batcher: collects requests up to max_batch_size or timeout_ms, then runs forward_batch.
pub struct DynamicBatcher {
    engine: InferenceEngine,
    max_batch_size: usize,
    timeout_ms: u64,
    request_rx: mpsc::Receiver<BatchRequest>,
}

impl DynamicBatcher {
    pub fn new(
        engine: InferenceEngine,
        max_batch_size: usize,
        timeout_ms: u64,
        request_rx: mpsc::Receiver<BatchRequest>,
    ) -> Self {
        Self {
            engine,
            max_batch_size,
            timeout_ms,
            request_rx,
        }
    }

    /// Run the batcher loop: wait for requests or timeout, then process batch.
    pub async fn run(&mut self) {
        let mut pending: Vec<BatchRequest> = Vec::new();
        let mut timer = interval(Duration::from_millis(self.timeout_ms));
        timer.tick().await;

        loop {
            tokio::select! {
                Some(req) = self.request_rx.recv() => {
                    pending.push(req);
                    if pending.len() >= self.max_batch_size {
                        self.process_batch(&mut pending).await;
                    }
                }
                _ = timer.tick() => {
                    if !pending.is_empty() {
                        self.process_batch(&mut pending).await;
                    }
                }
            }
        }
    }

    async fn process_batch(&self, requests: &mut Vec<BatchRequest>) {
        if requests.is_empty() {
            return;
        }

        let batch: Vec<Vec<usize>> = requests.iter().map(|r| r.prompt_ids.clone()).collect();
        let results = match self.engine.forward_batch(&batch) {
            Ok(logits) => logits,
            Err(e) => {
                for req in requests.drain(..) {
                    let _ = req.response_tx.send(Err(anyhow::anyhow!("{}", e)));
                }
                return;
            }
        };

        for (req, logits) in requests.drain(..).zip(results.into_iter()) {
            let _ = req.response_tx.send(Ok(vec![logits]));
        }
    }
}

/// Pad a sequence to target length with pad_id.
pub fn pad_sequence(seq: &[usize], target_len: usize, pad_id: usize) -> Vec<usize> {
    if seq.len() >= target_len {
        seq[..target_len].to_vec()
    } else {
        seq.iter()
            .cloned()
            .chain(std::iter::repeat_n(pad_id, target_len - seq.len()))
            .collect()
    }
}

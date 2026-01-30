//! Prefix (context) caching: reuse KV cache for repeated prompt prefixes.

use crate::inference::cache::KVCache;
use crate::inference::engine::InferenceEngine;
use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;

/// Cached activations for a prefix: filled KV cache ready to continue generation.
#[derive(Clone)]
pub struct CachedActivations {
    pub kv_cache: KVCache,
    pub timestamp: Instant,
}

/// Prefix cache: maps prefix token IDs to cached KV state. LRU eviction when full.
pub struct PrefixCache {
    cache: HashMap<Vec<usize>, CachedActivations>,
    max_entries: usize,
    hit_count: usize,
    miss_count: usize,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries: max_entries.max(1),
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Get cached activations for prefix, or compute and cache them.
    pub fn get_or_compute(
        &mut self,
        prefix: &[usize],
        engine: &InferenceEngine,
    ) -> Result<CachedActivations> {
        if prefix.is_empty() {
            let kv_cache = engine.create_cache();
            return Ok(CachedActivations {
                kv_cache,
                timestamp: Instant::now(),
            });
        }

        if let Some(cached) = self.cache.get(prefix) {
            self.hit_count = self.hit_count.saturating_add(1);
            return Ok(cached.clone());
        }

        self.miss_count = self.miss_count.saturating_add(1);
        let mut kv_cache = engine.create_cache();
        for &token_id in prefix {
            engine.forward_step(token_id, &mut kv_cache)?;
        }
        let activations = CachedActivations {
            kv_cache,
            timestamp: Instant::now(),
        };
        self.cache.insert(prefix.to_vec(), activations.clone());
        if self.cache.len() > self.max_entries {
            self.evict_oldest();
        }
        Ok(activations)
    }

    fn evict_oldest(&mut self) {
        let oldest = self
            .cache
            .iter()
            .min_by_key(|(_, v)| v.timestamp)
            .map(|(k, _)| k.clone());
        if let Some(k) = oldest {
            self.cache.remove(&k);
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total > 0 {
            self.hit_count as f64 / total as f64
        } else {
            0.0
        }
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

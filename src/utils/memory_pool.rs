//! Memory pool for reusable f32 buffers to reduce allocation in inference loops.
//!
//! Buffers are returned to the pool when dropped (via a guard) or explicitly.

use std::cell::RefCell;
use std::collections::VecDeque;

thread_local! {
    static POOL: RefCell<VecDeque<Vec<f32>>> = const { RefCell::new(VecDeque::new()) };
}

/// Minimum capacity to keep in pool; smaller buffers are discarded when returned.
const MIN_POOL_CAPACITY: usize = 256;

/// Maximum number of buffers to keep per thread.
const MAX_POOL_SIZE: usize = 32;

/// Acquire a buffer with at least `capacity` elements. May reuse a pooled buffer or allocate new.
pub fn acquire(capacity: usize) -> Vec<f32> {
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        while let Some(mut buf) = pool.pop_front() {
            if buf.capacity() >= capacity {
                buf.resize(capacity, 0.0);
                return buf;
            }
        }
        vec![0.0; capacity]
    })
}

/// Return a buffer to the pool for reuse. Only keeps buffers >= MIN_POOL_CAPACITY and limits pool size.
pub fn release(mut buf: Vec<f32>) {
    if buf.capacity() < MIN_POOL_CAPACITY {
        return;
    }
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() >= MAX_POOL_SIZE {
            return;
        }
        buf.clear();
        pool.push_back(buf);
    });
}

/// Guard that returns the buffer to the pool when dropped.
pub struct PooledBuffer {
    buf: Option<Vec<f32>>,
}

impl PooledBuffer {
    /// Take ownership of a buffer; when dropped it will be released to the pool.
    pub fn new(buf: Vec<f32>) -> Self {
        Self { buf: Some(buf) }
    }

    /// Get mutable slice. Panics if already taken.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.buf.as_deref_mut().expect("buffer already taken")
    }

    /// Get slice.
    pub fn as_slice(&self) -> &[f32] {
        self.buf.as_deref().expect("buffer already taken")
    }

    /// Take the buffer out without returning to pool.
    pub fn take(&mut self) -> Option<Vec<f32>> {
        self.buf.take()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            release(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_release_roundtrip() {
        let b = acquire(1024);
        assert!(b.len() >= 1024);
        release(b);
        let b2 = acquire(1024);
        assert!(b2.len() >= 1024);
        release(b2);
    }

    #[test]
    fn pooled_guard_returns_on_drop() {
        let b = acquire(512);
        let mut guard = PooledBuffer::new(b);
        guard.as_mut_slice()[0] = 1.0;
        drop(guard);
        let b2 = acquire(512);
        assert!(b2.capacity() >= 512);
        release(b2);
    }
}

//! Pre-allocated KV cache buffer (Vec-style: size/capacity + 2x grow).
//!
//! Used by model attention layers for CPU decode. Each attention layer owns
//! a pair of `KvBuf` (one for K, one for V) that accumulates KV state across
//! decode steps.
//!
//! To add KV cache to a new model architecture:
//! 1. Add `k_cache: KvBuf` and `v_cache: KvBuf` to your attention struct
//! 2. In `forward_with_cache`: call `k_cache.append(&k)`, then `k_cache.view()`
//! 3. In `reset_kv_cache`: call `k_cache.reset()` (keeps buffer allocated)
//! 4. Implement `ModelForward::forward_with_cache` and `supports_kv_cache`

use crate::tensor::{Result, Tensor};

/// Pre-allocated KV cache buffer for a single K or V tensor.
///
/// Layout: `[capacity, ...]` with `len` valid rows along dim 0.
/// - Append: O(1) amortized (scatter write via `slice_set`; grow only when full)
/// - View: zero-copy `narrow(0, 0, len)`
/// - Reset: sets len=0 without freeing memory (buffer reused across requests)
#[derive(Debug, Clone)]
pub struct KvBuf {
    buf: Option<Tensor>,
    len: usize,
}

impl KvBuf {
    pub fn new() -> Self {
        Self { buf: None, len: 0 }
    }

    /// Append `new_tokens` to the buffer along dim 0. Grows capacity if needed.
    pub fn append(&mut self, new_tokens: &Tensor) -> Result<()> {
        let n = new_tokens.dim(0)?;
        match &self.buf {
            None => {
                let cap = n.next_power_of_two().max(64);
                let shape = new_tokens.dims();
                let buf = Tensor::zeros(
                    (cap, shape[1], shape[2]),
                    new_tokens.dtype(),
                    new_tokens.device(),
                )?;
                buf.slice_set(new_tokens, 0, 0)?;
                self.buf = Some(buf);
                self.len = n;
            }
            Some(buf) => {
                let cap = buf.dim(0)?;
                if self.len + n > cap {
                    let new_cap = (cap * 2).max(self.len + n).next_power_of_two();
                    let shape = buf.dims();
                    let new_buf =
                        Tensor::zeros((new_cap, shape[1], shape[2]), buf.dtype(), buf.device())?;
                    let existing = buf.narrow(0, 0, self.len)?;
                    new_buf.slice_set(&existing, 0, 0)?;
                    new_buf.slice_set(new_tokens, 0, self.len)?;
                    self.buf = Some(new_buf);
                } else {
                    buf.slice_set(new_tokens, 0, self.len)?;
                }
                self.len += n;
            }
        }
        Ok(())
    }

    /// View of valid data: `[len, ...]`. Zero-copy narrow.
    pub fn view(&self) -> Result<Tensor> {
        match &self.buf {
            Some(buf) => buf.narrow(0, 0, self.len),
            None => crate::tensor::bail!("KvBuf not initialized"),
        }
    }

    /// Reset length to 0. Buffer memory is retained for reuse.
    pub fn reset(&mut self) {
        self.len = 0;
    }
}

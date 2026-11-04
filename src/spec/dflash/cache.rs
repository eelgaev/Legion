//! The drafter's per-sequence memory: the K and V of the target's fused hidden states, per
//! drafter layer, for the sequence's positions.
//!
//! Layout: F16 `[head_dim × n_head_kv, scratch + window]` per layer and kind. The first
//! `scratch` cells hold the K/V of the noise block of the pass being run (overwritten by every
//! pass); the others are a ring of `window` positions, position `p` in ring cell `p % window`.
//! The noise block's own K/V thus sit in the same tensor as the context, and one attention
//! reads a prefix of it: `scratch` plus the highest ring cell in use, as long as the ring has
//! not wrapped. Which position a ring cell holds is bookkeeping here (`-1`: none); the graph
//! gets it as a mask. A sequence longer than the window loses its oldest positions, and
//! rolling back after wrapping loses as many of the oldest as the rolled-back rows overwrote.

use crate::ggml::{BufferType, Context, GgmlError, Tensor, types};

use super::config::Config;

pub struct SeqCache {
    ctx: Context,
    scratch: usize,
    window: usize,
    /// The position each ring cell holds.
    pos: Vec<i64>,
}

impl SeqCache {
    pub fn new(
        cfg: &Config,
        buft: BufferType,
        scratch: usize,
        window: usize,
    ) -> Result<Self, GgmlError> {
        let ctx = Context::new(2 * cfg.n_layer, 0);
        let row = (cfg.head_dim * cfg.n_head_kv) as i64;
        for il in 0..cfg.n_layer {
            for kind in ["k", "v"] {
                ctx.new_tensor(types::F16, &[row, (scratch + window) as i64])
                    .named(&format!("cache_{kind}-{il}"));
            }
        }
        ctx.alloc_tensors(buft)?;
        // Masked cells still go through the matmuls: they must be finite, never NaN.
        ctx.clear_buffers(0);
        Ok(Self {
            ctx,
            scratch,
            window,
            pos: vec![-1; window],
        })
    }

    pub fn tensor(&self, kind: &str, il: usize) -> Tensor<'_> {
        self.ctx
            .tensor(&format!("cache_{kind}-{il}"))
            .unwrap_or_else(|| panic!("layer {il} has no cache"))
    }

    pub fn scratch(&self) -> usize {
        self.scratch
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// The cell position `p` is stored in.
    pub fn cell(&self, p: usize) -> usize {
        self.scratch + p % self.window
    }

    /// The position the ring cell `i` (0-based within the ring) holds, if any.
    pub fn held(&self, i: usize) -> Option<usize> {
        usize::try_from(self.pos[i]).ok()
    }

    /// Notes that position `p` is now stored (after the graph wrote it).
    pub fn store(&mut self, p: usize) {
        self.pos[p % self.window] = p as i64;
    }

    /// Forgets every position from `start` on (rows the target rejected).
    pub fn rollback(&mut self, start: usize) {
        for p in self.pos.iter_mut() {
            if *p >= start as i64 {
                *p = -1;
            }
        }
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        self.pos.fill(-1);
    }

    /// Cells an attention has to read: the scratch cells and the ring up to its last used cell.
    pub fn used_cells(&self) -> usize {
        let ring = self
            .pos
            .iter()
            .rposition(|&p| p >= 0)
            .map_or(0, |last| last + 1);
        self.scratch + ring
    }
}

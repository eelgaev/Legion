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
//!
//! With `--spec-draft-batched-attn` the caches are slots of a [`CachePool`]: one tensor per
//! layer and kind holds every sequence's cells, so one attention reads them all.

use crate::ggml::{BufferType, Context, GgmlError, Tensor, ffi, types};

use super::config::Config;

pub struct SeqCache {
    ctx: Context,
    scratch: usize,
    window: usize,
    /// The position each ring cell holds.
    pos: Vec<i64>,
    /// Its slot, when its cells are a [`CachePool`]'s.
    slot: Option<usize>,
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
            slot: None,
        })
    }

    /// The cache of slot `slot` of `pool`: its tensors are views of the pool's (the same layout
    /// as [`Self::new`]'s, `scratch + window` cells of a slot's `pool.cells()`). The pool must
    /// outlive it.
    pub fn in_pool(
        cfg: &Config,
        pool: &CachePool,
        slot: usize,
        scratch: usize,
        window: usize,
    ) -> Result<Self, GgmlError> {
        assert!(slot < pool.slots && scratch + window <= pool.cells);
        let ctx = Context::new(2 * cfg.n_layer, 0);
        let row = (cfg.head_dim * cfg.n_head_kv) as i64;
        for il in 0..cfg.n_layer {
            for kind in ["k", "v"] {
                let all = pool.tensor(kind, il);
                let nb1 = all.nb()[1];
                let view = ctx
                    .view_2d(
                        all,
                        [row, (scratch + window) as i64],
                        nb1,
                        slot * pool.cells * nb1,
                    )
                    .named(&format!("cache_{kind}-{il}"));
                // SAFETY: a view of an allocated tensor of `pool`, which outlives this cache: it
                // takes the pool's buffer and the data at its offset.
                let st = unsafe { ffi::ggml_backend_view_init(view.raw()) };
                if st != ffi::ggml_status_GGML_STATUS_SUCCESS {
                    return Err(GgmlError::Alloc {
                        bytes: 0,
                        buffer: "drafter cache pool view".into(),
                    });
                }
            }
        }
        Ok(Self {
            ctx,
            scratch,
            window,
            pos: vec![-1; window],
            slot: Some(slot),
        })
    }

    /// Its slot in a [`CachePool`], if it is one.
    pub fn slot(&self) -> Option<usize> {
        self.slot
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

/// The caches of up to `slots` sequences (`--spec-draft-batched-attn`): F16 `[head_dim ×
/// n_head_kv, cells × slots]` per layer and kind, slot `s` in cells `s · cells..(s + 1) · cells`
/// ([`SeqCache::in_pool`]). `cells` is a multiple of 256, so a flash attention over a prefix of
/// every slot reads whole tiles (ggml-cuda's GQA and mask-scan paths need it).
pub struct CachePool {
    ctx: Context,
    cells: usize,
    slots: usize,
    /// Slots no sequence holds.
    free: Vec<usize>,
}

/// [`CachePool`] cells per slot: a multiple of this.
pub const POOL_CELLS: usize = 256;

impl CachePool {
    pub fn new(
        cfg: &Config,
        buft: BufferType,
        scratch: usize,
        window: usize,
        slots: usize,
    ) -> Result<Self, GgmlError> {
        let cells = (scratch + window).next_multiple_of(POOL_CELLS);
        let ctx = Context::new(2 * cfg.n_layer, 0);
        let row = (cfg.head_dim * cfg.n_head_kv) as i64;
        for il in 0..cfg.n_layer {
            for kind in ["k", "v"] {
                ctx.new_tensor(types::F16, &[row, (cells * slots) as i64])
                    .named(&format!("pool_{kind}-{il}"));
            }
        }
        ctx.alloc_tensors(buft)?;
        // As SeqCache::new: masked cells are read too, so they must be finite.
        ctx.clear_buffers(0);
        Ok(Self {
            ctx,
            cells,
            slots,
            free: (0..slots).rev().collect(),
        })
    }

    pub fn tensor(&self, kind: &str, il: usize) -> Tensor<'_> {
        self.ctx
            .tensor(&format!("pool_{kind}-{il}"))
            .unwrap_or_else(|| panic!("layer {il} has no pool"))
    }

    /// Cells of one slot.
    pub fn cells(&self) -> usize {
        self.cells
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// A free slot (the lowest), if any.
    pub fn take(&mut self) -> Option<usize> {
        self.free.pop()
    }

    /// Gives `slot` back.
    pub fn give(&mut self, slot: usize) {
        debug_assert!(!self.free.contains(&slot));
        self.free.push(slot);
        // The lowest first again, so a full batch keeps slots in order.
        self.free.sort_unstable_by(|a, b| b.cmp(a));
    }
}

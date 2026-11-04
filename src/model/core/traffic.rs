//! The KV bytes a step's attention must read (#391's model bytes, see
//! [`crate::ggml::model_bytes`]): counted from the sequences' lengths, per pipeline stage, with
//! the share in the KV host tier apart. Required traffic, not measured: every position a layer
//! reads counts once per step (a prompt chunk's query tiles re-reading K/V are L2 or
//! re-reads), and the step's own K/V writes are left out.

use std::ops::Range;

use super::kv_type::KvTypes;
use super::{Group, Hparams, KV_BLOCK};

/// One stage's attention reads per position, by kind of layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvRates {
    /// K and V bytes per position over the layers that read every position (full attention).
    pub full: u64,
    /// K and V bytes per position over the sliding-window layers: they read the last
    /// [`Self::window`] positions (and the step's own).
    pub sliding: u64,
    pub window: usize,
    /// K and V bytes per position over the sparse-attention layers ([`Self::sparse_keys`]).
    pub sparse: u64,
    /// Indexer pool-key bytes per pool over the sparse layers (every complete pool is scored).
    pub pool_key: u64,
    /// `(keys a token selects, pool size)` of sparse attention, if the model has it.
    pub sparse_keys: Option<(usize, usize)>,
}

impl KvRates {
    /// The rates of the attention layers among `layers` (shared-KV layers read their source's
    /// cache, so they count too), K and V in `types`.
    pub fn new(hp: &Hparams, layers: Range<usize>, types: KvTypes) -> Self {
        let mut r = KvRates {
            window: hp.arch.swa_window(),
            sparse_keys: hp.arch.sparse_keys().filter(|&(_, kpool)| kpool > 0),
            ..Default::default()
        };
        // The pool keys are F16.
        let key = hp.arch.index_dim().max(0) as u64 * 2;
        for il in layers.filter(|&il| il < hp.n_layer && hp.is_attention(il)) {
            let b = hp.kv_cell_bytes(hp.kv_source(il), types) as u64;
            if hp.arch.indexed(il) && r.sparse_keys.is_some() {
                r.sparse += b;
                r.pool_key += key;
            } else if hp.sliding_layer(il) && r.window > 0 {
                r.sliding += b;
            } else {
                r.full += b;
            }
        }
        r
    }

    /// `(device, host)` bytes one sequence's attention reads in a step of `n_tok` tokens after
    /// `n_past`, its blocks `blocks`. `paged`: the step reads sparse layers' selections only
    /// (the gather path reads their whole span, as dense layers). `hbm_blocks`: with a KV host
    /// tier, blocks from it on are host memory (full and sparse layers; rings stay in VRAM).
    pub fn seq(
        &self,
        n_past: usize,
        n_tok: usize,
        blocks: &[u32],
        paged: bool,
        hbm_blocks: Option<usize>,
    ) -> (u64, u64) {
        let ctx = n_past + n_tok;
        if n_tok == 0 {
            return (0, 0);
        }
        let full = self.full * ctx as u64;
        let sliding = self.sliding * ctx.min(self.window + n_tok) as u64;
        let (selected, pools) = match self.sparse_keys {
            // Every token's selection at most, all of them different.
            Some((top_k, kpool)) if paged => {
                (ctx.min(n_tok.saturating_mul(top_k + kpool)), ctx / kpool)
            }
            Some((_, kpool)) => (ctx, ctx / kpool),
            None => (ctx, 0),
        };
        let sparse = self.sparse * selected as u64;
        let keys = self.pool_key * pools as u64;
        let host = match hbm_blocks {
            Some(hbm) => {
                let used = ctx.div_ceil(KV_BLOCK).min(blocks.len());
                let host_blocks = blocks[..used]
                    .iter()
                    .filter(|&&b| b as usize >= hbm)
                    .count();
                let host_pos = (host_blocks * KV_BLOCK).min(ctx) as u128;
                // The tier's share of the positions read.
                ((full + sparse) as u128 * host_pos / ctx as u128) as u64
            }
            None => 0,
        };
        (full + sliding + sparse + keys - host, host)
    }

    /// `(device, host)` bytes a step of `groups` reads ([`Self::seq`] summed).
    pub fn step(&self, groups: &[Group<'_>], paged: bool, hbm_blocks: Option<usize>) -> (u64, u64) {
        let mut out = (0, 0);
        for s in groups.iter().flat_map(|g| &g.seqs) {
            let (d, h) = self.seq(s.n_past, s.tokens.len(), s.blocks, paged, hbm_blocks);
            out.0 += d;
            out.1 += h;
        }
        out
    }
}

#[cfg(test)]
mod tests;

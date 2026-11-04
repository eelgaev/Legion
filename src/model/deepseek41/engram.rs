//! DeepSeek-V4.1-Flash's Engram tables (`blk.{l}.engram_embd.weight`, ~110 GB at IQ4_NL in
//! the companion `engram.file`) read through a host lookup table
//! ([`crate::residency::lookup_table`]), as qwen4exp's per-layer n-gram table is, instead of
//! being mapped like weights: the weight loader would read the whole file ahead
//! (`POSIX_FADV_WILLNEED`) and fill the page cache with it.
//!
//! The engine leaves the tables out of the weights ([`lookup_table`] names them) and opens one
//! table over the per-layer tables' row ranges, numbered on from one layer to the next, with
//! `--ngram-io` and its options: on a GPU sharing the host's memory (GB10) `auto` reads them
//! from the SSD with a bounded RAM row cache, elsewhere it takes them into RAM when they fit
//! next to what the load took. A step hashes its tokens, issues each layer's row reads as soon
//! as that layer's rows are known, and collects them layer by layer before the step's inputs
//! go to its graph (so the first layer's rows dequantize while the next one's are read); a
//! prompt's next chunk is prefetched while the current one computes
//! ([`Model::prefetch_engram`]). The rows are the file's bytes, dequantized as the weights' path
//! does ([`super::engram_rows`]): the same bits in every io mode.

#[cfg(test)]
mod tests;

use crate::ggml::row_size;
use crate::gguf::GgufFile;
use crate::model::core::state::NO_TOKEN;
use crate::model::core::{Group, Hparams, Model};
use crate::residency::lookup_table::{LookupTable, TableFile};

use super::{EngramParams, HashInputs, params, row_f32};

/// Rows longer than this are not read through a table (its direct tier reads at most two
/// pages per row).
const MAX_ROW_BYTES: usize = 4096;

/// The tensor name of Engram layer `l`'s table.
pub fn table_name(l: usize) -> String {
    format!("blk.{l}.engram_embd.weight")
}

/// The Engram tables of the `deepseek41` model `g` (hyperparameters `hp`) as the parts of one
/// lookup table, in the order of the Engram layers, and the tensors the weights leave out for
/// it. `None` for other models, and where one table cannot take them: tables in several files
/// (a split model holding them inline), of different row sizes, rows longer than a page or
/// more than a `u32` numbers; those load as weights, as before.
pub fn lookup_table(g: &GgufFile, hp: &Hparams) -> Option<(Vec<TableFile>, Vec<String>)> {
    let e = params(hp)?.engram.as_ref()?;
    let paths: Vec<_> = g.shard_paths().collect();
    let mut parts = Vec::with_capacity(e.layers.len());
    let mut names = Vec::with_capacity(e.layers.len());
    for &l in &e.layers {
        let name = table_name(l);
        let t = g.tensor(&name)?;
        // FP8 rows (a safetensors checkpoint's) need their scale rows too: the weights' path.
        if g.recipe(&name).is_some()
            || t.shape.len() != 2
            || t.shape[0] != e.dim as u64
            || t.ggml_type == crate::ggml::types::I8
        {
            return None;
        }
        parts.push((
            t.ggml_type,
            TableFile {
                path: paths.get(t.file_index)?.to_path_buf(),
                offset: t.offset,
                n_rows: t.shape[1],
                row_bytes: row_size(t.ggml_type, e.dim as usize),
            },
        ));
        names.push(name);
    }
    let (ty, first) = parts.first()?;
    let rows: u64 = parts.iter().map(|(_, p)| p.n_rows).sum();
    if parts
        .iter()
        .any(|(t, p)| t != ty || p.path != first.path || p.row_bytes != first.row_bytes)
        || first.row_bytes > MAX_ROW_BYTES
        || rows > u32::MAX as u64 + 1
    {
        return None;
    }
    Some((parts.into_iter().map(|(_, p)| p).collect(), names))
}

/// Where each Engram layer's rows start in the table of [`lookup_table`].
pub(super) fn bases(e: &EngramParams) -> Vec<u64> {
    e.tables
        .iter()
        .scan(0u64, |at, &(_, n)| {
            let base = *at;
            *at += n;
            Some(base)
        })
        .collect()
}

/// [`super::engram_rows`] with the rows read through `table` (opened on [`lookup_table`]'s
/// parts): each layer's reads issued as soon as its rows are hashed, then collected and
/// dequantized layer by layer. The same F32 values, in the same order (per token, per layer,
/// per column).
pub(super) fn rows_from_table(
    m: &Model,
    e: &EngramParams,
    table: &LookupTable,
    groups: &[Group<'_>],
    mut history: impl FnMut(&Group<'_>, usize, &[u32]) -> Vec<u32>,
) -> Vec<f32> {
    let inputs = HashInputs::read(m);
    let hash = inputs.hash(e);
    // Every sequence's tokens after their predecessors, in the order of the rows.
    let mut seqs: Vec<(Vec<u32>, &[u32])> = Vec::new();
    for g in groups {
        for (i, s) in g.seqs.iter().enumerate() {
            seqs.push((history(g, g.slot + i, s.tokens), s.tokens));
        }
    }
    let n_tok: usize = seqs.iter().map(|(_, t)| t.len()).sum();
    let (dim, cols, nl) = (e.dim as usize, e.cols(), e.layers.len());
    let bases = bases(e);
    let mut ids = Vec::new();
    let tickets: Vec<_> = (0..nl)
        .map(|li| {
            let (base, n_rows) = (bases[li], e.tables[li].1 as i64);
            let mut rows = Vec::with_capacity(n_tok * cols);
            for (prev, tokens) in &seqs {
                let mut prev = prev.clone();
                for &tok in *tokens {
                    hash.ids(li, tok, &prev, &mut ids);
                    for &id in &ids {
                        assert!(
                            (0..n_rows).contains(&id),
                            "Engram row {id} outside the {n_rows}-row table"
                        );
                        rows.push((base + id as u64) as u32);
                    }
                    prev.remove(0);
                    prev.push(tok);
                }
            }
            table.issue(&rows)
        })
        .collect();
    let ty = e.tables[0].0;
    let rb = table.row_bytes();
    let mut out = vec![0f32; n_tok * nl * cols * dim];
    for (li, ticket) in tickets.into_iter().enumerate() {
        let bytes = table
            .collect(ticket)
            .unwrap_or_else(|err| panic!("reading the Engram tables: {err}"));
        for (j, src) in bytes.chunks_exact(rb).enumerate() {
            let (t, c) = (j / cols, j % cols);
            let at = ((t * nl + li) * cols + c) * dim;
            out[at..at + dim].copy_from_slice(&row_f32(ty, src, dim));
        }
    }
    out
}

impl Model {
    /// Starts reading the Engram rows of `tokens`, which follow `before` in a sequence (its
    /// prompt so far; empty at its start), into the Engram table's row cache, so that their
    /// step finds them read: called with a prompt's next chunk while the current one computes.
    /// A hint ([`LookupTable::prefetch`]); nothing without an Engram table.
    pub fn prefetch_engram(&self, before: &[u32], tokens: &[u32]) {
        let e = params(&self.hp).and_then(|p| p.engram.as_ref());
        let (Some(e), Some(table)) = (e, &self.ngram) else {
            return;
        };
        let inputs = HashInputs::read(self);
        let hash = inputs.hash(e);
        let n_prev = e.ngram - 1;
        let mut prev: Vec<u32> = std::iter::repeat_n(NO_TOKEN, n_prev)
            .chain(
                before[before.len().saturating_sub(n_prev)..]
                    .iter()
                    .copied(),
            )
            .collect();
        prev.drain(..prev.len() - n_prev);
        let bases = bases(e);
        let mut ids = Vec::new();
        let mut rows = Vec::with_capacity(tokens.len() * e.layers.len() * e.cols());
        for &tok in tokens {
            for (li, &base) in bases.iter().enumerate() {
                hash.ids(li, tok, &prev, &mut ids);
                rows.extend(
                    ids.iter()
                        .filter(|&&id| (0..e.tables[li].1 as i64).contains(&id))
                        .map(|&id| (base + id as u64) as u32),
                );
            }
            prev.remove(0);
            prev.push(tok);
        }
        table.prefetch(&rows);
    }
}

//! The MTP head's output matrix as a 4-bit copy (`--spec-draft-lm-head q4`): a draft pass reads
//! the model's whole output head (or its first `--spec-draft-vocab` rows) once per draft token,
//! the largest matrix of the pass. Qwen3.8-27B's FP8 head is 1.27 GB over its 248,320 rows;
//! 377 MB as Q4_K over the first 131,072. NInfer drafts with a 4-bit head the same way
//! [Inspired by Ninfer (neroued/ninfer)]. Drafts only: every token is verified by the target with
//! its own head, so the output does not depend on it; only acceptance can change.

use super::{Model, ModelError, linear};
use crate::ggml::{Context, Tensor, types};

/// The copy's name in its context.
const NAME: &str = "draft.output.weight";

/// The name of its rows' token ids, when they are picked ([`Model::set_draft_ids`]).
const IDS: &str = "draft.output.ids";

impl Model {
    /// Makes the 4-bit copy of `output.weight` the MTP head drafts with from now on: its first
    /// [`Self::draft_vocab`] rows (all of them with 0), the weight's scale (per tensor or per
    /// row) folded in, next to it (same buffer type). Returns (rows, MiB of those rows as stored,
    /// MiB of the copy).
    pub fn quantize_draft_head(&mut self) -> Result<(usize, f64, f64), ModelError> {
        let w = self.w("output.weight")?;
        let [k, m, n2, n3] = w.ne();
        if m != self.hp.n_vocab || n2 != 1 || n3 != 1 {
            return Err(ModelError::Tensor(format!(
                "output.weight: shape {:?} is not the whole head ({} rows)",
                w.ne(),
                self.hp.n_vocab
            )));
        }
        let (rows, bytes) = self.draft_rows_bytes(w)?;
        let before = bytes.len();
        let scale = match linear::scale_name("output.weight").and_then(|s| self.w(&s).ok()) {
            None => None,
            Some(s) if s.ty() != types::F32 => {
                return Err(ModelError::Tensor(format!(
                    "output.scale is {}, expected F32",
                    crate::ggml::type_name(s.ty())
                )));
            }
            Some(s) if s.elements() == 1 => Some(s.to_f32()),
            Some(s) if s.elements() == m as usize => {
                let mut v = s.to_f32();
                match self.draft_ids() {
                    Some(ids) => v = ids.iter().map(|&i| v[i as usize]).collect(),
                    None => v.truncate(rows as usize),
                }
                Some(v)
            }
            Some(s) => {
                return Err(ModelError::Tensor(format!(
                    "output.scale has {} values for a head of {m} rows",
                    s.elements()
                )));
            }
        };
        let (ty, q) = crate::spec::q4_head::to_q4(w.ty(), &bytes, scale.as_deref(), k as usize)
            .map_err(|e| ModelError::Tensor(format!("output.weight: {e}")))?;
        let buft = buffer_type(w)?;
        self.store_draft_head(buft, ty, k, rows, &q)?;
        let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
        Ok((rows as usize, mib(before), mib(q.len())))
    }

    /// Makes a copy of the rows of `output.weight` that [`Self::set_draft_ids`] picked, as
    /// stored (same type, same bytes: a kept token's draft logit has the same bits), the one
    /// the MTP head drafts with from now on, next to the weight with the rows' token ids
    /// (#1172). A weight with a scale of its own is not copied. Returns (rows, MiB of the copy).
    pub fn copy_draft_head(&mut self) -> Result<(usize, f64), ModelError> {
        let w = self.w("output.weight")?;
        let [k, m, n2, n3] = w.ne();
        if m != self.hp.n_vocab || n2 != 1 || n3 != 1 || self.draft_ids().is_none() {
            return Err(ModelError::Tensor(format!(
                "output.weight: shape {:?} is not the whole head ({} rows) or no rows picked",
                w.ne(),
                self.hp.n_vocab
            )));
        }
        if linear::scale_name("output.weight").is_some_and(|s| self.w(&s).is_ok()) {
            return Err(ModelError::Unsupported(
                "picked draft rows of a scaled output head".into(),
            ));
        }
        let (rows, bytes) = self.draft_rows_bytes(w)?;
        let (ty, buft) = (w.ty(), buffer_type(w)?);
        self.store_draft_head(buft, ty, k, rows, &bytes)?;
        Ok((rows as usize, bytes.len() as f64 / (1u64 << 20) as f64))
    }

    /// The draft rows of `w` (`output.weight`) as stored: the first [`Self::draft_vocab`] (all
    /// with 0), or the rows [`Self::set_draft_ids`] picked, in that order. Returns (rows, bytes).
    fn draft_rows_bytes(&self, w: Tensor<'_>) -> Result<(i64, Vec<u8>), ModelError> {
        let [k, m, ..] = w.ne();
        let row = crate::ggml::row_size(w.ty(), k as usize);
        let Some(ids) = self.draft_ids() else {
            let rows = match self.draft_vocab() {
                v if v > 0 && (v as i64) < m => v as i64,
                _ => m,
            };
            return Ok((rows, w.bytes_prefix(row * rows as usize)));
        };
        if ids.is_empty() || ids.last().is_some_and(|&i| i as i64 >= m) {
            return Err(ModelError::Tensor(format!(
                "output.weight: {} draft rows for a head of {m}",
                ids.len()
            )));
        }
        // Runs of consecutive ids, one read each.
        let mut bytes = Vec::with_capacity(row * ids.len());
        for run in ids_runs(ids) {
            bytes.extend(w.bytes_at(run.start as usize * row, run.len() * row));
        }
        Ok((ids.len() as i64, bytes))
    }

    /// Stores `q` (`rows` rows of `k` values of type `ty`) as the draft head in `buft`, with
    /// the picked rows' token ids beside it when [`Self::set_draft_ids`] picked them.
    fn store_draft_head(
        &mut self,
        buft: crate::ggml::BufferType,
        ty: crate::ggml::ffi::ggml_type,
        k: i64,
        rows: i64,
        q: &[u8],
    ) -> Result<(), ModelError> {
        let ctx = Context::new(2, 0);
        ctx.new_tensor(ty, &[k, rows]).named(NAME);
        if self.draft_ids.is_some() {
            ctx.new_tensor(types::I32, &[rows]).named(IDS);
        }
        ctx.alloc_tensors(buft)?;
        ctx.tensor(NAME).expect("created above").set_bytes(q);
        if let Some(ids) = self.draft_ids() {
            let b: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
            ctx.tensor(IDS).expect("created above").set_bytes(&b);
        }
        self.draft_head = Some(ctx);
        Ok(())
    }

    /// The copy [`Self::quantize_draft_head`] or [`Self::copy_draft_head`] made, if any: the
    /// MTP head's output matrix.
    pub(crate) fn draft_head(&self) -> Option<Tensor<'_>> {
        self.draft_head.as_ref().and_then(|c| c.tensor(NAME))
    }

    /// The token ids of the draft head's rows (I32), when [`Self::set_draft_ids`] picked them.
    pub(crate) fn draft_head_ids(&self) -> Option<Tensor<'_>> {
        self.draft_head.as_ref().and_then(|c| c.tensor(IDS))
    }
}

/// The buffer type `w` (an allocated weight) lives in.
fn buffer_type(w: Tensor<'_>) -> Result<crate::ggml::BufferType, ModelError> {
    // SAFETY: an allocated weight's buffer; reads its type.
    unsafe {
        Ok(crate::ggml::BufferType(
            std::ptr::NonNull::new(crate::ggml::ffi::ggml_backend_buffer_get_type(
                (*w.raw()).buffer,
            ))
            .ok_or_else(|| ModelError::Tensor("output.weight: no buffer".into()))?,
        ))
    }
}

/// `ids` (ascending) as runs of consecutive ids.
pub(crate) fn ids_runs(ids: &[u32]) -> Vec<std::ops::Range<u32>> {
    let mut out: Vec<std::ops::Range<u32>> = Vec::new();
    for &i in ids {
        match out.last_mut() {
            Some(r) if r.end == i => r.end += 1,
            _ => out.push(i..i + 1),
        }
    }
    out
}

#[cfg(test)]
mod tests;

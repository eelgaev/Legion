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
        let rows = match self.draft_vocab() {
            v if v > 0 && (v as i64) < m => v as i64,
            _ => m,
        };
        let row = crate::ggml::row_size(w.ty(), k as usize);
        let bytes = w.bytes_prefix(row * rows as usize);
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
                v.truncate(rows as usize);
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
        // SAFETY: an allocated weight's buffer; reads its type.
        let buft = unsafe {
            crate::ggml::BufferType(
                std::ptr::NonNull::new(crate::ggml::ffi::ggml_backend_buffer_get_type(
                    (*w.raw()).buffer,
                ))
                .ok_or_else(|| ModelError::Tensor("output.weight: no buffer".into()))?,
            )
        };
        let ctx = Context::new(1, 0);
        ctx.new_tensor(ty, &[k, rows]).named(NAME);
        ctx.alloc_tensors(buft)?;
        ctx.tensor(NAME).expect("created above").set_bytes(&q);
        self.draft_head = Some(ctx);
        let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
        Ok((rows as usize, mib(before), mib(q.len())))
    }

    /// The copy [`Self::quantize_draft_head`] made, if any: the MTP head's output matrix.
    pub(crate) fn draft_head(&self) -> Option<Tensor<'_>> {
        self.draft_head.as_ref().and_then(|c| c.tensor(NAME))
    }
}

#[cfg(test)]
mod tests;

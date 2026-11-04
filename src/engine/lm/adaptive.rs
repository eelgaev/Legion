//! Per-sequence policy integration and the last drafting step's mean requested depth.

use super::{Seq, draft_cap};
use crate::spec::SpecConfig;

#[derive(Default)]
pub(super) struct StepDepth {
    sum: usize,
    sequences: usize,
}

impl StepDepth {
    pub(super) fn mean(&self) -> Option<f64> {
        (self.sequences > 0).then(|| self.sum as f64 / self.sequences as f64)
    }
}

impl Seq<'_> {
    pub(super) fn adaptive_cap(
        &mut self,
        cfg: &SpecConfig,
        batch: usize,
        max_ctx: usize,
        depths: &mut StepDepth,
    ) -> usize {
        let k = if self.active.constrained() {
            0
        } else {
            self.adaptive.select(cfg.k, batch, cfg)
        };
        depths.sum += k;
        depths.sequences += 1;
        draft_cap(
            k,
            self.active.max_tokens,
            self.active.generated,
            max_ctx,
            self.n_past,
        )
    }

    /// The cap for depth `k` chosen for the whole batch ([`crate::spec::depth`]); grammar
    /// constrained sequences do not draft.
    pub(super) fn depth_cap(&mut self, k: usize, max_ctx: usize, depths: &mut StepDepth) -> usize {
        let k = if self.active.constrained() { 0 } else { k };
        depths.sum += k;
        depths.sequences += 1;
        draft_cap(
            k,
            self.active.max_tokens,
            self.active.generated,
            max_ctx,
            self.n_past,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_includes_zero_depths() {
        assert_eq!(StepDepth::default().mean(), None);
        assert_eq!(
            StepDepth {
                sum: 7,
                sequences: 4
            }
            .mean(),
            Some(1.75)
        );
        assert_eq!(
            StepDepth {
                sum: 0,
                sequences: 2
            }
            .mean(),
            Some(0.0)
        );
    }
}

//! Images in a sequence's token stream (#146): where each image's tokens are, their input
//! embedding rows, and the multi-section RoPE positions they and the text after them take.
//!
//! An image of `nx × ny` merged patches is `nx × ny` tokens (`<|image_pad|>` in the token
//! stream) whose input rows come from the vision tower instead of the token embedding table.
//! Qwen-VL's M-RoPE gives them, at the image's first position `p0`, the positions
//! `[p0, p0 + row, p0 + col, 0]` across the four sections, and the image advances the text's
//! position by `max(nx, ny)` only: the text after it is at its token index less what the images
//! before it saved (`nx × ny − max(nx, ny)` each). Text tokens are `[p, p, p, 0]`.
//!
//! The attention stays causal in token order: llama.cpp orders an image's tokens (which share
//! their first section) by (row, column), the same order.
//!
//! Adapted from llama.cpp's `tools/mtmd/mtmd.cpp` (`mtmd_image_tokens_get_decoder_pos`,
//! `mtmd_image_tokens_get_n_pos` for `MTMD_POS_TYPE_MROPE`) (MIT).

use std::sync::Arc;

/// One image's tokens in a sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSpan {
    /// Index of the image's first token in the sequence.
    pub start: usize,
    pub nx: usize,
    pub ny: usize,
    /// `nx × ny` input rows of the model's embedding width, row by row.
    pub rows: Arc<[f32]>,
}

impl ImageSpan {
    pub fn n_tokens(&self) -> usize {
        self.nx * self.ny
    }

    fn end(&self) -> usize {
        self.start + self.n_tokens()
    }

    /// Positions the image saves against one per token.
    fn saved(&self) -> usize {
        self.n_tokens() - self.nx.max(self.ny)
    }
}

/// A sequence's images, in token order (see the module docs).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Media {
    spans: Vec<ImageSpan>,
    n_embd: usize,
}

impl Media {
    /// `spans` in token order, not overlapping, each `nx × ny × n_embd` rows long.
    pub fn new(spans: Vec<ImageSpan>, n_embd: usize) -> Self {
        for w in spans.windows(2) {
            assert!(w[0].end() <= w[1].start, "image spans overlap");
        }
        for s in &spans {
            assert!(s.nx > 0 && s.ny > 0, "an empty image");
            assert_eq!(s.rows.len(), s.n_tokens() * n_embd, "image rows");
        }
        Self { spans, n_embd }
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn spans(&self) -> &[ImageSpan] {
        &self.spans
    }

    /// The width of the image rows.
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// The image span holding token `i`, if any, and the positions saved before it.
    fn locate(&self, i: usize) -> (Option<&ImageSpan>, usize) {
        let mut saved = 0;
        for s in &self.spans {
            if i < s.start {
                break;
            }
            if i < s.end() {
                return (Some(s), saved);
            }
            saved += s.saved();
        }
        (None, saved)
    }

    /// Token `i`'s positions in the four RoPE sections.
    pub fn rope(&self, i: usize) -> [i32; 4] {
        match self.locate(i) {
            (Some(s), saved) => {
                let p0 = (s.start - saved) as i32;
                let k = i - s.start;
                [p0, p0 + (k / s.nx) as i32, p0 + (k % s.nx) as i32, 0]
            }
            (None, saved) => {
                let p = (i - saved) as i32;
                [p, p, p, 0]
            }
        }
    }

    /// Token `i`'s input row, if it is an image token.
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        let (s, _) = self.locate(i);
        let s = s?;
        Some(&s.rows[(i - s.start) * self.n_embd..][..self.n_embd])
    }

    /// How many of tokens `range` are image tokens.
    pub fn image_tokens(&self, range: std::ops::Range<usize>) -> usize {
        self.spans
            .iter()
            .map(|s| {
                s.end()
                    .min(range.end)
                    .saturating_sub(s.start.max(range.start))
            })
            .sum()
    }

    /// The index of the first image token, if any.
    pub fn first_image(&self) -> Option<usize> {
        self.spans.first().map(|s| s.start)
    }
}

#[cfg(test)]
mod tests;

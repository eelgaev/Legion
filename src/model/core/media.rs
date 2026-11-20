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
//! Gemma 4's images ([`Media::sequential`], #1217) take one position per token, as text does,
//! and on the sliding-window layers an image's tokens see the whole image ([`Media::span_end`]:
//! llama.cpp decodes an image's batch non-causally there); the global layers stay causal.
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

/// Positions an image's tokens take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageLayout {
    /// Qwen-VL's M-RoPE (see the module docs).
    #[default]
    MRope,
    /// One position per token, as text (Gemma 4).
    Sequential,
}

/// A sequence's images, in token order (see the module docs).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Media {
    spans: Vec<ImageSpan>,
    n_embd: usize,
    layout: ImageLayout,
    /// Shared by this media and its clones only, not by other media of the same rows (the
    /// image-row cache hands one image's rows to every request that sends it): whether any of
    /// them still exists ([`Self::holders`]).
    held: Arc<()>,
}

impl Media {
    /// `spans` in token order, not overlapping, each `nx × ny × n_embd` rows long, at M-RoPE
    /// positions.
    pub fn new(spans: Vec<ImageSpan>, n_embd: usize) -> Self {
        Self::with_layout(spans, n_embd, ImageLayout::MRope)
    }

    /// [`Self::new`] at one position per token (Gemma 4).
    pub fn sequential(spans: Vec<ImageSpan>, n_embd: usize) -> Self {
        Self::with_layout(spans, n_embd, ImageLayout::Sequential)
    }

    /// [`Self::new`] at the positions of `layout`.
    pub fn with_layout(spans: Vec<ImageSpan>, n_embd: usize, layout: ImageLayout) -> Self {
        for w in spans.windows(2) {
            assert!(w[0].end() <= w[1].start, "image spans overlap");
        }
        for s in &spans {
            assert!(s.nx > 0 && s.ny > 0, "an empty image");
            assert_eq!(s.rows.len(), s.n_tokens() * n_embd, "image rows");
        }
        Self {
            spans,
            n_embd,
            layout,
            held: Arc::default(),
        }
    }

    pub fn layout(&self) -> ImageLayout {
        self.layout
    }

    /// A handle on this media and its clones: its strong count says whether any of them
    /// still exists, and while it is held no other media takes its address.
    pub fn holders(&self) -> std::sync::Weak<()> {
        Arc::downgrade(&self.held)
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
        if self.layout == ImageLayout::Sequential {
            return [i as i32, i as i32, i as i32, 0];
        }
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

    /// The last token of the image holding token `i`, if it is an image token: the last key
    /// it sees on the sliding-window layers of a model whose images are bidirectional there
    /// ([`super::Arch::image_tokens_bidirectional`]).
    pub fn span_end(&self, i: usize) -> Option<usize> {
        let (s, _) = self.locate(i);
        s.map(|s| s.end() - 1)
    }

    /// The end of a prompt chunk of tokens `from..end` that cuts no image (an image whose
    /// tokens see each other must be in one step, as llama.cpp's non-causal image batch):
    /// `end` itself, or the start of the image it cuts, or for an image the chunk starts with,
    /// its end when that is at most `cap` tokens past `from` (else `end`: such an image is
    /// cut).
    pub fn uncut(&self, from: usize, end: usize, cap: usize) -> usize {
        let Some(s) = (self.spans.iter()).find(|s| s.start < end && end < s.end()) else {
            return end;
        };
        if s.start > from {
            s.start
        } else if s.end() - from <= cap {
            s.end()
        } else {
            end
        }
    }

    /// `sizes`, pieces of a prompt chunk from token `from` on, with no cut inside an image (a
    /// step's micro-chunks run one after another, so an image cut there would see only its
    /// first part's keys): a cut inside an image moves back to the image's start, or past its
    /// end when the piece before would be empty (a piece then holds more than its size). The
    /// same pieces when no cut falls inside an image.
    pub fn uncut_sizes(&self, from: usize, sizes: &[usize]) -> Vec<usize> {
        let n: usize = sizes.iter().sum();
        let mut out = Vec::with_capacity(sizes.len());
        let mut prev = 0;
        let mut cut = 0;
        for &len in &sizes[..sizes.len().saturating_sub(1)] {
            cut += len;
            let at = from + cut;
            let c = match (self.spans.iter()).find(|s| s.start < at && at < s.end()) {
                Some(s) if s.start > from + prev => s.start - from,
                Some(s) => s.end() - from,
                None => cut,
            };
            if c > prev && c < n {
                out.push(c - prev);
                prev = c;
            }
        }
        if n > prev {
            out.push(n - prev);
        }
        out
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

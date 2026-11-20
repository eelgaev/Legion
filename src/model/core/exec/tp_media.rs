//! Images under tensor parallelism over machines (#1195): only rank 0 loads the vision tower and
//! encodes a request's images. The other ranks need the images' input rows for the steps that
//! run their tokens, and every step of such a sequence needs its M-RoPE positions
//! ([`Media::rope`]), so rank 0 sends each sequence's images once, before the first step that
//! names them ([`Request::TpMedia`]), and steps name them by id ([`crate::node::WireSeq::media`]).
//! Once no sequence on rank 0 holds them any more, the others forget them
//! ([`Request::TpMediaDrop`]).
//!
//! The rows go as rank 0's F32 values: every rank's steps read the same bits.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use crate::model::core::{ImageLayout, ImageSpan, Media};
use crate::node::Request;

#[cfg(test)]
mod tests;

/// The wire id of no images.
pub const NONE: u32 = 0;

/// What rank 0 has sent ([`SentMedia::id`]).
struct Sent {
    id: u32,
    /// The sent [`Media`]'s [`Media::holders`]: a held [`Weak`] keeps the allocation, so no
    /// other media can take its address while it is here, and its strong count says whether
    /// that media or a clone of it still exists. Not the rows: the image-row cache keeps those
    /// past the sequences, and shares them between requests of the same image.
    held: Weak<()>,
}

impl Sent {
    /// Whether `m` is these images (that media or a clone of it, not other media of the same
    /// rows).
    fn is(&self, m: &Media) -> bool {
        Weak::ptr_eq(&self.held, &m.holders())
    }

    /// Whether a [`Media`] can still hold these images.
    fn alive(&self) -> bool {
        self.held.strong_count() > 0
    }
}

/// Rank 0's images sent to the other ranks.
#[derive(Default)]
pub struct SentMedia {
    last: u32,
    sent: Vec<Sent>,
}

impl SentMedia {
    /// `m`'s wire id, and the request that sends it when the other ranks do not have it yet
    /// (to send before the step that names it).
    pub fn id(&mut self, m: &Media) -> (u32, Option<Request>) {
        if let Some(s) = self.sent.iter().find(|s| s.is(m)) {
            return (s.id, None);
        }
        // Ids wrap past u32::MAX (never NONE): an id still in use then is skipped.
        let mut id = self.last;
        loop {
            id = id.wrapping_add(1);
            if id != NONE && self.sent.iter().all(|s| s.id != id) {
                break;
            }
        }
        self.last = id;
        let spans = m.spans();
        let n: usize = spans.iter().map(|s| s.rows.len()).sum();
        let mut rows = Vec::with_capacity(n);
        for s in spans {
            rows.extend_from_slice(&s.rows);
        }
        self.sent.push(Sent {
            id,
            held: m.holders(),
        });
        let req = Request::TpMedia {
            id,
            n_embd: m.n_embd() as u32,
            spans: (spans.iter())
                .map(|s| (s.start as u32, s.nx as u32, s.ny as u32))
                .collect(),
            rows,
        };
        (id, Some(req))
    }

    /// The images no [`Media`] holds any more, forgotten here, as the request that tells the
    /// other ranks to forget them too (`None`: none).
    pub fn gone(&mut self) -> Option<Request> {
        if self.sent.iter().all(Sent::alive) {
            return None;
        }
        let mut ids = Vec::new();
        self.sent.retain(|s| {
            let keep = s.alive();
            if !keep {
                ids.push(s.id);
            }
            keep
        });
        Some(Request::TpMediaDrop { ids })
    }
}

/// A rank above 0's images from rank 0, by wire id.
#[derive(Default)]
pub struct MediaTable {
    media: HashMap<u32, Media>,
}

impl MediaTable {
    /// A [`Request::TpMedia`]'s images, checked (a bad request fails here, not in a step), at
    /// the positions of `layout` (the model's: rank 0's are the same, so the wire does not
    /// carry it).
    pub fn add(
        &mut self,
        id: u32,
        n_embd: u32,
        spans: &[(u32, u32, u32)],
        rows: &[f32],
        layout: ImageLayout,
    ) -> Result<(), String> {
        if id == NONE {
            return Err("images with the id of none".into());
        }
        let n_embd = n_embd as usize;
        if n_embd == 0 && !spans.is_empty() {
            return Err(format!("images {id}: rows 0 wide"));
        }
        let mut out = Vec::with_capacity(spans.len());
        let (mut at, mut end) = (0usize, 0usize);
        for &(start, nx, ny) in spans {
            let (start, nx, ny) = (start as usize, nx as usize, ny as usize);
            let n = nx
                .checked_mul(ny)
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("images {id}: an image of {nx} x {ny}"))?;
            if start < end {
                return Err(format!(
                    "images {id}: an image at {start} overlaps the one before"
                ));
            }
            end = start
                .checked_add(n)
                .ok_or_else(|| format!("images {id}: an image past the end"))?;
            let len = n
                .checked_mul(n_embd)
                .ok_or_else(|| format!("images {id}: too many rows"))?;
            let r = (at.checked_add(len))
                .and_then(|e| rows.get(at..e))
                .ok_or_else(|| format!("images {id}: {} floats sent, more wanted", rows.len()))?;
            at += len;
            out.push(ImageSpan {
                start,
                nx,
                ny,
                rows: Arc::from(r),
            });
        }
        if at != rows.len() {
            return Err(format!(
                "images {id}: {} floats sent, {at} wanted",
                rows.len()
            ));
        }
        self.media
            .insert(id, Media::with_layout(out, n_embd, layout));
        Ok(())
    }

    /// A [`Request::TpMediaDrop`].
    pub fn forget(&mut self, ids: &[u32]) {
        for id in ids {
            self.media.remove(id);
        }
    }

    /// The images of wire id `id` ([`NONE`]: none).
    pub fn get(&self, id: u32) -> Result<Option<&Media>, String> {
        if id == NONE {
            return Ok(None);
        }
        (self.media.get(&id).map(Some)).ok_or_else(|| format!("images {id}: not sent"))
    }
}

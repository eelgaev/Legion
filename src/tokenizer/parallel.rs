//! Long texts pre-tokenized and merged on several threads. #439: a 123K-token prompt spent about
//! 100 ms on one thread here before its first GPU work.
//!
//! The text is cut at a space before a word, where the GPT-2-style patterns start a pre-token.
//! Each thread walks the pre-tokens from its cut with `find_from_pos`, which is what `find_iter`
//! runs after every non-empty match, on the whole text so look-arounds see the same context. A
//! thread's walk counts only if its last pre-token ends exactly at the next cut. If every walk
//! does, every cut is a point where the one-thread walk also resumes, so the pre-tokens and ids
//! are the same. Anything else (a cut inside a pre-token, an empty match, a regex error) gives
//! `None` and the caller encodes the text on one thread, as before.

use super::{Scratch, Tokenizer};

#[cfg(test)]
mod tests;

/// Texts shorter than this stay on the calling thread.
pub(super) const MIN_BYTES: usize = 64 << 10;
/// Fewest bytes a thread gets.
pub(super) const CHUNK_BYTES: usize = 16 << 10;
/// Most threads one text uses.
pub(super) const MAX_THREADS: usize = 16;

impl Tokenizer {
    /// `encode_text`'s ids for `text` from up to `threads` threads of at least `chunk` bytes each,
    /// or `None` when the text does not cut into at least two exact parts.
    pub(super) fn encode_text_parallel(
        &self,
        text: &str,
        threads: usize,
        chunk: usize,
    ) -> Option<Vec<u32>> {
        let cuts = cut_points(text, threads, chunk);
        if cuts.len() < 3 {
            return None;
        }
        let parts: Vec<Option<Vec<u32>>> = std::thread::scope(|s| {
            let walks: Vec<_> = cuts
                .windows(2)
                .map(|w| {
                    let (start, end) = (w[0], w[1]);
                    s.spawn(move || self.encode_span(text, start, end))
                })
                .collect();
            walks
                .into_iter()
                .map(|w| w.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect()
        });
        let mut out = Vec::with_capacity(parts.iter().flatten().map(Vec::len).sum());
        for part in parts {
            out.extend(part?);
        }
        Some(out)
    }

    /// The ids of the pre-tokens from `start`, a point where the one-thread walk resumes, to
    /// `end`; `None` unless the last one ends exactly at `end`, and at the end of the text unless
    /// no match follows (the one-thread walk would take an empty one there).
    pub(super) fn encode_span(&self, text: &str, start: usize, end: usize) -> Option<Vec<u32>> {
        let mut scratch = Scratch::default();
        let mut out = Vec::with_capacity((end - start) / 2);
        let mut pos = start;
        while pos < end {
            let m = self.regex.find_from_pos(text, pos).ok()??;
            if m.start() == m.end() || m.end() > end {
                return None;
            }
            self.encode_word(m.as_str(), &mut scratch, &mut out);
            pos = m.end();
        }
        if end == text.len() && !matches!(self.regex.find_from_pos(text, end), Ok(None)) {
            return None;
        }
        Some(out)
    }
}

/// `0`, then up to `threads - 1` cuts at least `chunk` bytes apart, each at a space that follows
/// a non-space and precedes an ASCII letter, then `text.len()`.
pub(super) fn cut_points(text: &str, threads: usize, chunk: usize) -> Vec<usize> {
    let bytes = text.as_bytes();
    let n = threads.min(bytes.len() / chunk.max(1)).max(1);
    let mut cuts = vec![0];
    for k in 1..n {
        let last = *cuts.last().unwrap_or(&0);
        let mut p = (bytes.len() * k / n).max(last + chunk);
        while p + 1 < bytes.len()
            && !(bytes[p] == b' '
                && !bytes[p - 1].is_ascii_whitespace()
                && bytes[p + 1].is_ascii_alphabetic())
        {
            p += 1;
        }
        if p + chunk > bytes.len() {
            break;
        }
        cuts.push(p);
    }
    cuts.push(bytes.len());
    cuts
}

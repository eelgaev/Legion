//! A new prompt's first chunk: its table rows read from the request's arrival (#836).
//!
//! A model with a per-layer n-gram (or Engram) table read through a store reads a prompt
//! chunk's rows from the SSD. A chunk's next chunk is prefetched while it computes, but the
//! first chunk's rows were only read inside its own step, after the step's input preparation,
//! and the GPU waited for them at the table's first layer: on a DGX Spark (GB10) about 7 ms of
//! a 120-token prompt's TTFT and 70 to 110 ms of a 1,080-token one's when the rows come from a
//! slow disk. An idle engine now starts those reads as the request arrives, so they run while
//! it admits the prompt and prepares the step (and through the `--deterministic` gather
//! window). The rows are the same bytes, so outputs do not change; a busy engine leaves it to
//! the step (its next fill would wait for the reads).

use super::Worker;

/// How many of a `prompt`-token prompt's first tokens to read ahead: its first step's chunk
/// at most (`chunk` tokens per step, `budget` tokens per step in all).
pub(super) fn ahead_len(prompt: usize, chunk: usize, budget: usize) -> usize {
    prompt.min(chunk.max(1)).min(budget.max(1))
}

impl Worker {
    /// Starts reading the table rows of `prompt`'s first chunk (a hint; nothing without a
    /// table read through a store). Called by an idle engine on a request's arrival.
    pub(super) fn prefetch_first(&mut self, prompt: &[u32]) {
        let n = ahead_len(prompt.len(), self.prefill_chunk, self.max_batch_tokens);
        if n == 0 {
            return;
        }
        let first = &prompt[..n];
        self.exec.model().prefetch_ngram(&[], first);
        self.exec.model().prefetch_engram(&[], first);
        // The other ranks of tensor parallelism (idle too) read theirs at once (#1176).
        self.exec.tp_prefetch_rows(&[], first, false);
    }
}

#[cfg(test)]
mod tests;

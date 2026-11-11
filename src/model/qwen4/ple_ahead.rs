//! A pipelined prompt's next n-gram rows, gathered while the stages compute (#439).
//!
//! The per-layer n-gram embedding's input is a step's table rows, dequantized on the host
//! ([`Model::fill_ple`](crate::model::core::Model)): ~8 ms for a 4,096-token micro-chunk of
//! Qwen3.8-Flash-Next on an AC922. Its stage fills that input once its own graph of the round is
//! done (the inputs may share memory with the graph), and on a long prompt that stage (the
//! first) is the round's slowest one, so the gather sat between its graphs with every GPU
//! waiting. The rows depend only on the micro-chunk's tokens and the slot's token history, so
//! the executor gathers them as soon as the round is launched; the fill then only writes them.
//!
//! What a fill writes is the gather of exactly the rows it hashes: held values are used only
//! for an identical row list, so the input's bytes are the same either way.

#[cfg(test)]
mod tests;

/// Values gathered ahead of a fill, for the rows they were gathered for.
#[derive(Default)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct PleAhead {
    held: Option<(Vec<i32>, Vec<f32>)>,
    /// The values a fill wrote, kept for the next gather to write into (#1183): a new ~40 MB
    /// vector per 4,096-token step faulted all its pages in again, and freeing it unmapped them.
    spare: Vec<f32>,
    /// Fills that took held values.
    hits: u64,
}

impl PleAhead {
    /// A vector of `n` values to gather into: the spare one when there is one (its old values
    /// left in it, for the gather to overwrite).
    pub(crate) fn buffer(&mut self, n: usize) -> Vec<f32> {
        let mut v = std::mem::take(&mut self.spare);
        v.resize(n, 0.0);
        v
    }

    /// Keeps `values` (written) as the spare vector for [`Self::buffer`].
    pub(crate) fn recycle(&mut self, values: Vec<f32>) {
        if values.capacity() >= self.spare.capacity() {
            self.spare = values;
        }
    }

    /// Holds `values`, the gather of `rows`, for the next fill (replacing any held ones).
    pub(crate) fn put(&mut self, rows: Vec<i32>, values: Vec<f32>) {
        self.held = Some((rows, values));
    }

    /// The held values if they are the gather of exactly `rows`. Whatever was held is
    /// dropped either way: a fill is the one place the values go.
    pub(crate) fn take(&mut self, rows: &[i32]) -> Option<Vec<f32>> {
        let (held, values) = self.held.take()?;
        if held != rows {
            return None;
        }
        self.hits += 1;
        Some(values)
    }

    /// Fills that took held values so far.
    #[cfg(test)]
    pub(crate) fn hits(&self) -> u64 {
        self.hits
    }
}

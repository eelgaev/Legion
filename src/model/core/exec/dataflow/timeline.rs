//! A pipelined step's timeline for its `step dataflow` debug line (#785): per stage, the
//! pipeline fill (until its first graph starts), the drain (after its last graph, until the
//! step ends) and the host's lateness (a graph started after both its own previous graph and
//! the residual it takes were done; the first stage's first graph counts from the step's
//! start), so the step's bubble and host time read off the log
//! without the per-micro-chunk trace.

#[cfg(test)]
mod tests;

/// Milliseconds since the step started, per stage and micro-chunk.
#[derive(Debug, Clone)]
pub(super) struct Timeline {
    first: Vec<Option<f64>>,
    done: Vec<Vec<Option<f64>>>,
    late: Vec<f64>,
}

impl Timeline {
    pub fn new(n_stages: usize, n_micro: usize) -> Self {
        Self {
            first: vec![None; n_stages],
            done: vec![vec![None; n_micro]; n_stages],
            late: vec![0.0; n_stages],
        }
    }

    /// Stage `s`'s graph of micro-chunk `m` started at `at`. A graph queued before what it
    /// waits on was done (`--long-dataflow-async`) is never late.
    pub fn launched(&mut self, s: usize, m: usize, at: f64) {
        self.first[s].get_or_insert(at);
        let own = match m.checked_sub(1) {
            Some(k) => self.done[s][k],
            None => Some(0.0),
        };
        let up = match s.checked_sub(1) {
            Some(u) => self.done[u][m],
            None => Some(0.0),
        };
        if let (Some(own), Some(up)) = (own, up) {
            self.late[s] += (at - own.max(up)).max(0.0);
        }
    }

    /// Stage `s`'s graph of micro-chunk `m` was done at `at`.
    pub fn finished(&mut self, s: usize, m: usize, at: f64) {
        self.done[s][m] = Some(at);
    }

    /// Per stage: until its first graph started.
    pub fn fill(&self) -> Vec<f64> {
        self.first.iter().map(|f| f.unwrap_or(0.0)).collect()
    }

    /// Per stage: from its last graph's end to `end`, the step's.
    pub fn drain(&self, end: f64) -> Vec<f64> {
        self.done
            .iter()
            .map(|d| {
                d.iter()
                    .rev()
                    .find_map(|x| *x)
                    .map_or(0.0, |t| (end - t).max(0.0))
            })
            .collect()
    }

    /// Per stage: summed lateness of its graphs' starts (see the module docs).
    pub fn late(&self) -> &[f64] {
        &self.late
    }
}

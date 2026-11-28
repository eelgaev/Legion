//! `--prompt-step-grid` (#1183): prompt chunks of a step bigger than the grid hold whole
//! multiples of it, and a prompt's last part past them runs as its own chunk, as it does in
//! steps of the grid's size. A prompt then has the same chunk boundaries it has in steps of the
//! grid, only fewer of them: an 8,240-token prompt in steps of 12,288 runs 8,192 + 48 tokens
//! (4,096 + 4,096 + 48 in steps of 4,096), not 8,240 at once. The prompt-step kernels give a
//! token the same bits whatever other whole grid cells share its step, so the output keeps the
//! grid-sized steps' bits while the bigger steps read each routed expert once for more tokens.
//! That holds for a prompt alone in its steps: decode rows or other prompts sharing a step take
//! some of its room, so its cuts (and the bits) can differ from those of grid-sized steps (with
//! eight decode rows, steps of 4,096 cut a 20,000-token prompt at 4,088 and 8,176; steps of
//! 12,288 on a 4,096 grid at 8,192).

/// The tokens a prompt chunk of `len` tokens (as the step's room, the prompt's end and its save
/// points left it) takes on a grid of `grid` tokens: `len` up to the grid (the last part of the
/// prompt, or a step no bigger than the grid), else the most whole grid cells it holds.
/// `grid` 0: `len`.
pub(crate) fn grid_take(len: usize, grid: usize) -> usize {
    if grid == 0 || len <= grid {
        return len;
    }
    len / grid * grid
}

#[cfg(test)]
mod tests;

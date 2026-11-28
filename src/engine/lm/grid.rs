//! `--prompt-step-grid` (#1183): prompt chunks of a step bigger than the grid hold whole
//! multiples of it, and a prompt's last part past them runs as its own chunk, as it does in
//! steps of the grid's size. A prompt then has the same chunk boundaries it has in steps of the
//! grid, only fewer of them: an 8,240-token prompt in steps of 12,288 runs 8,192 + 48 tokens
//! (4,096 + 4,096 + 48 in steps of 4,096), not 8,240 at once. The prompt-step kernels give a
//! token the same bits whatever other whole grid cells share its step, so the output keeps the
//! grid-sized steps' bits while the bigger steps read each routed expert once for more tokens.
//!
//! `--prompt-step-grid-fold`: a last part of at least [`FOLD_MIN_TOKENS`] tokens runs in the
//! step of the whole cells before it instead of alone (32,738 tokens in steps of 12,288:
//! 12,288 + 12,288 + 8,162, not 12,288 + 12,288 + 4,096 + 4,066). A part that big already takes
//! the kernels of a whole cell on its own (the small-step paths stop below it: the joined
//! GEMMs at `--joined-prompt-min-tokens`, the BF16 hc down projection at 2,048 tokens, the
//! grouped expert mv kernel at 32 pairs per expert, cuBLAS's split K on short steps), so its
//! tokens keep their bits beside the cells, as the cells keep theirs beside each other.

/// `--prompt-step-grid-fold`'s default: the fewest tokens of a prompt's last part that runs in
/// the step of the whole grid cells before it.
pub const FOLD_MIN_TOKENS: usize = 2048;

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

/// [`grid_take`] with the fold (`--prompt-step-grid-fold`): a chunk that ends the prompt
/// (`at_end`) takes its part past the whole cells too when that part holds `fold` tokens or
/// more. `fold` 0: never.
pub(crate) fn grid_take_folded(len: usize, grid: usize, fold: usize, at_end: bool) -> usize {
    let take = grid_take(len, grid);
    if at_end && fold > 0 && len - take >= fold {
        len
    } else {
        take
    }
}

#[cfg(test)]
mod tests;

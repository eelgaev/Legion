//! `--prompt-step-grid` (#1183): prompt chunks of a step bigger than the grid hold whole
//! multiples of it, and a prompt's last part past them runs as its own chunk, as it does in
//! steps of the grid's size. A prompt then has the same chunk boundaries it has in steps of the
//! grid, only fewer of them: an 8,240-token prompt in steps of 12,288 runs 8,192 + 48 tokens
//! (4,096 + 4,096 + 48 in steps of 4,096), not 8,240 at once. The prompt-step kernels give a
//! token the same bits whatever other whole grid cells share its step, so the output keeps the
//! grid-sized steps' bits while the bigger steps read each routed expert once for more tokens.
//!
//! The fold: where every MoE router's logits come from Legion's kernel (`--legion-router`,
//! [`crate::model::core::Model::routers_on_legion_kernel`]), a last part of at least
//! [`FOLD_MIN_TOKENS`] tokens runs in the step of the whole cells before it instead of alone
//! (32,738 tokens in steps of 12,288: 12,288 + 12,288 + 8,162, not 12,288 + 12,288 + 4,096 +
//! 4,066; one step fewer). A part that big takes the big-step kernels on its own too (the
//! small-step paths stop below it: the joined GEMMs at `--joined-prompt-min-tokens`, the BF16
//! hc down projection at 2,048 tokens, the grouped expert mv kernel at 32 pairs per expert),
//! so its tokens get the same bits beside the cells as alone. Without that kernel the router
//! logits of prompt steps come from cuBLAS, whose algorithm follows the step's token count:
//! folding would move the output, so the last part runs alone, as before.

/// The fewest tokens of a prompt's last part past the whole grid cells that the fold runs in
/// their step: the shortest step that takes every big-step prompt kernel (the BF16 hc down
/// projection starts at 2,048 tokens, the joined GEMMs at 1,024).
pub const FOLD_MIN_TOKENS: usize = 2048;

/// The fold's floor for [`grid_take_folded`]: [`FOLD_MIN_TOKENS`] where every MoE router runs
/// on Legion's kernel (`legion_router`), else 0 (no fold: the grid alone).
pub(crate) fn fold_min(legion_router: bool) -> usize {
    if legion_router { FOLD_MIN_TOKENS } else { 0 }
}

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

/// [`grid_take`] with the fold ([`fold_min`]): a chunk that ends the prompt
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

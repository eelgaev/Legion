//! Which routed experts each GPU owns under `--glm-parallel-decode` (`--glm-expert-split`,
//! #677).
//!
//! The parallel step gives GPU `d` the experts `bounds[d]..bounds[d + 1]` ([`super::expert_bounds`])
//! of every MoE layer. With the file's own expert order, that is a contiguous quarter of the ids
//! per GPU on four GPUs, and the router's choices do not spread evenly over those quarters: on the
//! AC922, GPUs 2 and 3 (experts 144-287 of Maya-S) take 15-35% longer in the expert matvecs per
//! step than GPUs 0 and 1 with the same bytes read, and the others wait for them at every
//! all-reduce (#669, #677).
//!
//! [`ExpertSplit::Interleave`] (opt-in) renumbers the experts at load instead: the step's expert `i` is the
//! file's expert `order[i]` ([`expert_order`]), so GPU `d` owns the file's experts `d`, `d + n`,
//! `d + 2n`, ... (`n` GPUs), a sample of the whole id range. The renumbering is a row
//! permutation of what the loader reads: the router's rows (`ffn_gate_inp`), its selection bias
//! (`exp_probs_b`) and the experts' matrices (`ffn_*_exps`) of every decoder MoE layer, each a
//! [`Recipe::Gather`] of the file's slices in the new order. Everything past the load (the
//! expert cache, its host arena's NUMA halves, the exchange's owners) works on the new ids as
//! before. The router picks the same experts (under new ids) with the same weights; each GPU's
//! partial sum holds other experts than before, so the routed sum adds in another order: greedy
//! output differs from the contiguous split's at near-ties.
//!
//! Measured on the AC922 (Maya-S + MTP, four GPUs, #677): interleaved, GPUs 2 and 3 stay as late
//! as before (the others wait 24.7 and 37.1 ms per step for them, against 24.7 and 38.0 with the
//! contiguous split) and decode is unchanged, so the lateness does not follow the expert ids;
//! the contiguous split stays the default.

#[cfg(test)]
mod tests;

use crate::gguf::{FileRange, GgufFile, Recipe, TensorInfo};

/// How the routed experts of a MoE layer map to the GPUs (`--glm-expert-split`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum ExpertSplit {
    /// GPU `d` owns the file's experts `bounds[d]..bounds[d + 1]` (the default).
    #[default]
    Contiguous,
    /// GPU `d` owns the file's experts `d`, `d + n`, `d + 2n`, ... of `n` GPUs.
    Interleave,
}

/// The tensors of a decoder MoE layer whose last dimension is the expert: renumbered together.
pub const ROUTED: [&str; 5] = [
    "ffn_gate_inp.weight",
    "exp_probs_b.bias",
    "ffn_up_exps.weight",
    "ffn_gate_exps.weight",
    "ffn_down_exps.weight",
];

/// The file's expert behind each of the step's expert ids under `split` (`n_expert` experts
/// over `n_dev` GPUs): `None` for the file's order. Interleaved, the file's experts sorted by
/// `(e % n_dev, e)`: with `n_dev` dividing `n_expert` GPU `d`'s range of
/// [`super::expert_bounds`] is exactly the file's experts `e % n_dev == d`.
pub fn expert_order(split: ExpertSplit, n_expert: usize, n_dev: usize) -> Option<Vec<u32>> {
    match split {
        ExpertSplit::Contiguous => None,
        ExpertSplit::Interleave if n_dev <= 1 => None,
        ExpertSplit::Interleave => Some(
            (0..n_dev)
                .flat_map(|d| (d..n_expert).step_by(n_dev))
                .map(|e| e as u32)
                .collect(),
        ),
    }
}

/// `order` inverted: the step's id of each of the file's experts.
pub fn inverse(order: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; order.len()];
    for (i, &e) in order.iter().enumerate() {
        inv[e as usize] = i as u32;
    }
    inv
}

/// Tensor `t` (last dimension the `order.len()` experts, a plain range of the file) with its
/// expert slices read in `order`: slice `i` is the file's slice `order[i]`.
pub fn permuted(t: &TensorInfo, order: &[u32]) -> Result<Recipe, String> {
    let n = order.len() as u64;
    if t.shape.last() != Some(&n) {
        return Err(format!(
            "`{}`: shape {:?}, not {n} experts last",
            t.name, t.shape
        ));
    }
    if !t.size.is_multiple_of(n) {
        return Err(format!("`{}`: uneven expert slices", t.name));
    }
    let slice = t.size / n;
    Ok(Recipe::Gather(
        order
            .iter()
            .map(|&e| FileRange {
                file: t.file_index,
                offset: t.offset + e as u64 * slice,
                len: slice,
            })
            .collect(),
    ))
}

/// Checks that MoE layer `il` of `g` can be renumbered: its [`ROUTED`] tensors are plain ranges
/// of the file without checkpoint scales, and no other tensor of the layer is per expert.
pub fn check_layer(g: &GgufFile, il: usize) -> Result<(), String> {
    let prefix = format!("blk.{il}.");
    for t in g.tensors() {
        let Some(w) = t.name.strip_prefix(&prefix) else {
            continue;
        };
        if ROUTED.contains(&w) {
            if g.recipe(&t.name).is_some() {
                return Err(format!(
                    "`{}` is synthesized, not a range of the file",
                    t.name
                ));
            }
        } else if w.contains("_exps")
            || w.starts_with("exp_probs")
            || w.starts_with("ffn_gate_inp.")
        {
            return Err(format!(
                "`{}`: a per-expert tensor it cannot renumber",
                t.name
            ));
        }
    }
    Ok(())
}

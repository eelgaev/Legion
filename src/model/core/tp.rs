//! Tensor parallelism over a meta device ([`crate::ggml::meta`]): which axis each weight and
//! cache tensor is split along, by the Megatron layout. Model-neutral: it goes by the standard
//! GGUF tensor names and the layers' head shapes.
//!
//! - Q, K and V projections and the FFN's gate and up: split by output rows (axis 1), in whole
//!   heads (K, V), whole GQA groups (Q: a KV head's query heads stay with it) and whole rows.
//! - The attention output and the FFN's down projection: split by input columns (axis 0), the
//!   same way, so each device's partial product sums with the others' (the meta backend's
//!   all-reduce).
//! - The KV cache: by KV heads (axis 0 of `[head_dim × n_head_kv, cells]`).
//! - Everything else (embeddings, norms, the output head, per-head norm weights): mirrored.
//!
//! A layer whose KV heads don't divide evenly among the devices keeps its attention whole on
//! every device (mirrored weights and cache): splitting would leave a device without heads.

use super::Hparams;
use crate::ggml::meta::{Split, SplitRule};

/// The FFN's rows are split in multiples of this (whole quantization super-blocks along the
/// down projection's input, and kernels' preferred tile).
const FFN_GRANULE: i64 = 256;

/// Per layer: `(head_dim, n_head_kv, n_head)`.
fn heads(hp: &Hparams) -> Vec<(i64, i64, i64)> {
    (0..hp.n_layer)
        .map(|il| {
            let (d, n_kv) = hp.kv_dims(il);
            (d, n_kv.max(1), hp.n_head)
        })
        .collect()
}

/// The split of tensor `name` (shape `ne`) over `n_dev` devices for a model with `layers` =
/// [`heads`].
pub(crate) fn split_of(
    layers: &[(i64, i64, i64)],
    n_dev: usize,
    name: &str,
    ne: [i64; 4],
) -> Split {
    let even = |n_kv: i64| n_kv % n_dev as i64 == 0;
    let axis = |axis, granularity| Split::Axis { axis, granularity };
    if let Some(il) = name
        .strip_prefix("cache_k-")
        .or_else(|| name.strip_prefix("cache_v-"))
        .and_then(|s| s.parse::<usize>().ok())
    {
        return match layers.get(il) {
            Some(&(d, n_kv, _)) if even(n_kv) => axis(0, d),
            _ => Split::Mirrored,
        };
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return Split::Mirrored;
    };
    let mut parts = rest.splitn(2, '.');
    let (Some(il), Some(tensor)) = (
        parts.next().and_then(|s| s.parse::<usize>().ok()),
        parts.next(),
    ) else {
        return Split::Mirrored;
    };
    let layer = layers.get(il).copied();
    // Query-side tensors by their own shape: the layer's heads, else (a tensor not shaped
    // as its layer's queries) those of a cache with that head size: an MTP head's query-only
    // layers read the target's caches, under the target's tensor names but their own layer
    // numbers. A layer's own queries never take another layer's split: with uneven KV heads
    // its cache is whole, and so must they be.
    let by_width = |width: i64| -> Option<i64> {
        let fits = |&(d, n_kv, n_head): &(i64, i64, i64)| {
            (n_head * d == width && even(n_kv)).then_some(d * (n_head / n_kv).max(1))
        };
        match layer {
            Some(own @ (d, _, n_head)) if n_head * d == width => fits(&own),
            _ => layers.iter().find_map(fits),
        }
    };
    match tensor {
        "attn_q.weight" => by_width(ne[1]).map_or(Split::Mirrored, |g| axis(1, g)),
        "attn_output.weight" => by_width(ne[0]).map_or(Split::Mirrored, |g| axis(0, g)),
        "attn_k.weight" | "attn_v.weight" => match layer {
            Some((d, n_kv, _)) if even(n_kv) && ne[1] == n_kv * d => axis(1, d),
            _ => Split::Mirrored,
        },
        "ffn_up.weight" | "ffn_gate.weight" => axis(1, FFN_GRANULE),
        "ffn_down.weight" => axis(0, FFN_GRANULE),
        _ => Split::Mirrored,
    }
}

/// The attention layers of `hp` whose KV heads do not divide among `n_dev` devices (their
/// attention stays whole on every device, see the module doc).
pub fn uneven_kv_layers(hp: &Hparams, n_dev: usize) -> Vec<usize> {
    uneven(&heads(hp), n_dev)
        .into_iter()
        .filter(|&il| hp.owns_kv(il))
        .collect()
}

fn uneven(layers: &[(i64, i64, i64)], n_dev: usize) -> Vec<usize> {
    layers
        .iter()
        .enumerate()
        .filter(|(_, (_, n_kv, _))| n_kv % n_dev as i64 != 0)
        .map(|(il, _)| il)
        .collect()
}

/// The Megatron split rule for `hp`'s layers over `n_dev` devices, for
/// [`crate::ggml::meta::meta_device`].
pub fn megatron_rule(hp: &Hparams, n_dev: usize) -> Box<SplitRule> {
    let layers = heads(hp);
    Box::new(move |name, ne, _ty| {
        compact_split(&layers, n_dev, name, ne)
            .unwrap_or_else(|| split_of(&layers, n_dev, name, ne))
    })
}

/// The split of a Gemma 4 compact cache (#551, [`super::KvPool`]): a `cache_k-{il}` wider than
/// its layer's `head_dim × n_head_kv` values per cell, granules of `head_dim + r` per kv head
/// (the rule is made before the model's weights load, so before it knows its compact layers:
/// the cache's shape marks them). By kv heads like a plain cache, in those granules. `None` for
/// any other tensor, a plain cache included ([`split_of`]).
pub(crate) fn compact_split(
    layers: &[(i64, i64, i64)],
    n_dev: usize,
    name: &str,
    ne: [i64; 4],
) -> Option<Split> {
    let il = name.strip_prefix("cache_k-")?.parse::<usize>().ok()?;
    let (d, n_kv, _) = *layers.get(il)?;
    if ne[0] <= d * n_kv || ne[0] % n_kv != 0 {
        return None;
    }
    Some(if n_kv % n_dev as i64 == 0 {
        Split::Axis {
            axis: 0,
            granularity: ne[0] / n_kv,
        }
    } else {
        Split::Mirrored
    })
}

#[cfg(test)]
mod tests;

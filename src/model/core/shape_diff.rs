//! What `--stage-graph-stats`' build lines say about a new kept graph (#858): its kind, how its
//! shape differs from the nearest shape kept for that kind, and why no kept graph served it.
//! Only computed while the log is on.

use super::GroupShape;

#[cfg(test)]
mod tests;

/// Differing fields shown per build line; past it, `+N more`.
const SHOWN: usize = 6;

/// The kind of a target step's graph of `shape`: `decode` (one token a sequence), `verify` (a
/// verify step, `verify`: no prompt chunk among its groups) or `prompt` (a prompt chunk among
/// them).
pub(crate) fn step_kind(shape: &[GroupShape], verify: bool) -> &'static str {
    if shape.iter().all(|g| g.n_tok == 1) {
        "decode"
    } else if verify {
        "verify"
    } else {
        "prompt"
    }
}

/// The fields of shape `b` that differ from `a`, as `g<i>.<field> <a>-><b>` (`groups <a>-><b>`
/// first when the group counts differ; the groups both have are compared in order).
pub(crate) fn diff(a: &[GroupShape], b: &[GroupShape]) -> Vec<String> {
    let mut out = Vec::new();
    if a.len() != b.len() {
        out.push(format!("groups {}->{}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let mut f = |name: &str, p: String, q: String| {
            if p != q {
                out.push(format!("g{i}.{name} {p}->{q}"));
            }
        };
        f("slot", x.slot.to_string(), y.slot.to_string());
        f("n_seqs", x.n_seqs.to_string(), y.n_seqs.to_string());
        f("n_tok", x.n_tok.to_string(), y.n_tok.to_string());
        f("n_kv", x.n_kv.to_string(), y.n_kv.to_string());
        f("table", x.table.to_string(), y.table.to_string());
        f("ring", x.ring.to_string(), y.ring.to_string());
        // The logits rows: their totals, or each sequence's when only their split moved.
        let (p, q) = if x.out_rows() == y.out_rows() {
            (format!("{:?}", x.logits), format!("{:?}", y.logits))
        } else {
            (x.out_rows().to_string(), y.out_rows().to_string())
        };
        f("logits", p, q);
        f(
            "last_only",
            x.last_only.to_string(),
            y.last_only.to_string(),
        );
        f("verify", x.verify.to_string(), y.verify.to_string());
        f("n_pool", x.n_pool.to_string(), y.n_pool.to_string());
        f("n_new", x.n_new.to_string(), y.n_new.to_string());
        f("pick_all", x.pick_all.to_string(), y.pick_all.to_string());
        f(
            "kv_slots",
            format!("{:?}", x.kv_slots),
            format!("{:?}", y.kv_slots),
        );
        f("par", x.par.to_string(), y.par.to_string());
        f("pair", x.pair.to_string(), y.pair.to_string());
        f(
            "head_half",
            x.head_half.to_string(),
            y.head_half.to_string(),
        );
        f("img", x.img.to_string(), y.img.to_string());
        f(
            "captures",
            format!("{:?}", x.captures),
            format!("{:?}", y.captures),
        );
    }
    out
}

/// Of the `kept` shapes, the differences ([`diff`]) of the one nearest `shape` (the fewest
/// differing fields; the most recently kept, listed last, of equals). `None`: none kept.
pub(crate) fn nearest<'a>(
    kept: impl IntoIterator<Item = &'a [GroupShape]>,
    shape: &[GroupShape],
) -> Option<Vec<String>> {
    kept.into_iter()
        .map(|k| diff(k, shape))
        .reduce(|best, d| if d.len() <= best.len() { d } else { best })
}

/// [`nearest`]'s answer for the log: its first [`SHOWN`] differences (`none kept` without one).
pub(crate) fn describe(nearest: Option<Vec<String>>) -> String {
    let Some(d) = nearest else {
        return "none kept".into();
    };
    let more = d.len().saturating_sub(SHOWN);
    let mut s = d.into_iter().take(SHOWN).collect::<Vec<_>>().join(", ");
    if more > 0 {
        s.push_str(&format!(", +{more} more"));
    }
    s
}

/// Why no kept graph served a new graph's shape: `new shape` the first time a stage built that
/// kind's shape while the log was on (`first`); else it was kept before and dropped since,
/// `at the cap` when the kind keeps its limit (`kept` of `cap`) and the oldest goes now, or
/// `below the cap` (dropped for the byte budget, for memory, or when the shared buffer moved).
pub(crate) fn why(first: bool, kept: usize, cap: usize) -> &'static str {
    if first {
        "new shape"
    } else if cap > 0 && kept >= cap {
        "evicted at the cap"
    } else {
        "dropped below the cap"
    }
}

/// The build line's note ([`crate::ggml::stage_graph::stats::BuildNote`]) for a new graph of
/// `shape` and `kind` on stage `stage`, made before its kind's oldest graph is evicted for it:
/// `kept` lists the shapes its kind keeps now, `n_kept` of `cap` count against its cap.
pub(crate) fn note<'a>(
    kind: &str,
    stage: usize,
    shape: &[GroupShape],
    kept: impl IntoIterator<Item = &'a [GroupShape]>,
    n_kept: usize,
    cap: usize,
) -> crate::ggml::stage_graph::stats::BuildNote {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    shape.hash(&mut h);
    let first = crate::ggml::stage_graph::stats::first_build(kind, stage, h.finish());
    crate::ggml::stage_graph::stats::BuildNote {
        shape: GroupShape::brief(shape),
        nearest: describe(nearest(kept, shape)),
        why: why(first, n_kept, cap),
        kept: n_kept,
        cap,
    }
}

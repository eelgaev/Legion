//! `--tp-moe-ep-place` (#803): which rank holds each routed expert, per layer, under
//! `--tp-moe-ep` ([`super::shard::qwen4::Experts::Placed`]).
//!
//! In decode, each layer's all-reduce waits for the rank that drew more of the token's experts.
//! The contiguous split ([`super::shard::qwen4::expert_range`]) ignores which experts the
//! router picks together; a placement built from recorded routing puts experts that are often
//! picked together on different ranks, so a token's experts spread more evenly and the
//! per-layer max-over-ranks expert count drops. Every rank still holds exactly `E / ranks`
//! experts (memory stays equal), whole, in ascending expert order (its slots).
//!
//! - [`Placement`]: the table, a small JSON file (`legion ep-place` writes it): per layer, the
//!   rank of every expert. Layers it does not name keep the contiguous split.
//! - [`Recorded`]: router top-k ids recorded by `legion serve --moe-route-record`
//!   ([`write_record`]'s format).
//! - [`split`]: the greedy splitter over a layer's recorded tokens.

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::io::Write;

use serde::{Deserialize, Serialize};

/// A per-layer assignment of routed experts to ranks (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    /// Routed experts per layer.
    pub n_expert: usize,
    /// Ranks the experts are shared out among.
    pub ranks: usize,
    /// Layer index (the MTP head's block included, `n_layer`) to the rank of each expert.
    pub layers: BTreeMap<usize, Vec<u8>>,
}

impl Placement {
    /// Reads and checks a placement file.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let p: Self =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        p.check().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(p)
    }

    /// Checks that every listed layer gives each of the `ranks` exactly `n_expert / ranks`
    /// experts.
    pub fn check(&self) -> Result<(), String> {
        if self.ranks == 0 || self.n_expert == 0 || !self.n_expert.is_multiple_of(self.ranks) {
            return Err(format!(
                "{} experts do not divide among {} ranks",
                self.n_expert, self.ranks
            ));
        }
        if self.ranks > u8::MAX as usize {
            return Err(format!("{} ranks", self.ranks));
        }
        let per = self.n_expert / self.ranks;
        for (il, owner) in &self.layers {
            if owner.len() != self.n_expert {
                return Err(format!(
                    "layer {il}: {} entries for {} experts",
                    owner.len(),
                    self.n_expert
                ));
            }
            let mut count = vec![0usize; self.ranks];
            for &r in owner {
                let Some(c) = count.get_mut(r as usize) else {
                    return Err(format!("layer {il}: rank {r} of {}", self.ranks));
                };
                *c += 1;
            }
            if let Some(r) = count.iter().position(|&c| c != per) {
                return Err(format!(
                    "layer {il}: rank {r} holds {} experts, not {per}",
                    count[r]
                ));
            }
        }
        Ok(())
    }

    /// Checks that the placement fits a model of `n_expert` routed experts on `ranks` ranks.
    pub fn fits(&self, n_expert: usize, ranks: usize) -> Result<(), String> {
        if self.n_expert != n_expert || self.ranks != ranks {
            return Err(format!(
                "the placement is for {} experts on {} ranks, the model has {n_expert} on {ranks}",
                self.n_expert, self.ranks
            ));
        }
        Ok(())
    }

    /// Layer `il`'s rank of every expert: the table's, or the contiguous split's.
    pub fn owner(&self, il: usize) -> Vec<u8> {
        self.layers
            .get(&il)
            .cloned()
            .unwrap_or_else(|| contiguous(self.n_expert, self.ranks))
    }

    /// Whether layer `il` has an entry of its own (else the contiguous split).
    pub fn places(&self, il: usize) -> bool {
        self.layers.contains_key(&il)
    }

    /// The experts rank `rank` holds in layer `il`, ascending: its slots in order.
    pub fn experts_of(&self, il: usize, rank: usize) -> Vec<u64> {
        experts_of(&self.owner(il), rank)
    }

    /// Layer `il`'s slot table for rank `rank`: expert `e`'s slot in the rank's stack, -1 for
    /// the experts other ranks hold.
    pub fn slot_table(&self, il: usize, rank: usize) -> Vec<i32> {
        slot_table(&self.owner(il), rank)
    }

    /// Serializes the placement (one line per layer).
    pub fn to_json(&self) -> String {
        let mut s = format!(
            "{{\n  \"n_expert\": {},\n  \"ranks\": {},\n  \"layers\": {{",
            self.n_expert, self.ranks
        );
        for (i, (il, owner)) in self.layers.iter().enumerate() {
            let row: Vec<String> = owner.iter().map(|r| r.to_string()).collect();
            let sep = if i == 0 { "" } else { "," };
            s.push_str(&format!("{sep}\n    \"{il}\": [{}]", row.join(",")));
        }
        s.push_str("\n  }\n}\n");
        s
    }
}

/// The contiguous split: expert `e` on rank `e / (n_expert / ranks)`.
pub fn contiguous(n_expert: usize, ranks: usize) -> Vec<u8> {
    let per = (n_expert / ranks.max(1)).max(1);
    (0..n_expert).map(|e| (e / per) as u8).collect()
}

/// The experts `owner` gives rank `rank`, ascending.
pub fn experts_of(owner: &[u8], rank: usize) -> Vec<u64> {
    owner
        .iter()
        .enumerate()
        .filter(|&(_, &r)| r as usize == rank)
        .map(|(e, _)| e as u64)
        .collect()
}

/// Expert `e`'s slot among rank `rank`'s experts (ascending), -1 for the others'.
pub fn slot_table(owner: &[u8], rank: usize) -> Vec<i32> {
    let mut next = 0;
    owner
        .iter()
        .map(|&r| {
            if r as usize == rank {
                next += 1;
                next - 1
            } else {
                -1
            }
        })
        .collect()
}

/// Ascending expert ids as runs of consecutive ones: `(first, count)`.
pub fn runs(experts: &[u64]) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for &e in experts {
        match out.last_mut() {
            Some((s, n)) if *s + *n == e => *n += 1,
            _ => out.push((e, 1)),
        }
    }
    out
}

/// One layer's recorded routing: `k` expert ids per token, token after token.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rows {
    pub k: usize,
    pub ids: Vec<u16>,
}

impl Rows {
    /// Tokens recorded.
    pub fn tokens(&self) -> usize {
        self.ids.len().checked_div(self.k).unwrap_or(0)
    }

    /// Tokens `range` of these.
    pub fn slice(&self, range: std::ops::Range<usize>) -> Rows {
        Rows {
            k: self.k,
            ids: self.ids[range.start * self.k..range.end * self.k].to_vec(),
        }
    }
}

/// Routing recorded by `legion serve --moe-route-record`, per layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Recorded {
    pub n_expert: usize,
    pub layers: BTreeMap<usize, Rows>,
}

/// A record's header: layer, routed experts, ids per token, tokens (u32 LE each).
const HEADER: usize = 16;

/// Appends one step's routing of layer `il` to a record: `ids` holds `n_used` ids per token.
pub fn write_record(
    w: &mut impl Write,
    il: usize,
    n_expert: usize,
    n_used: usize,
    ids: &[i32],
) -> std::io::Result<()> {
    let n_tok = ids.len().checked_div(n_used).unwrap_or(0);
    let mut b = Vec::with_capacity(HEADER + ids.len() * 4);
    for v in [il, n_expert, n_used, n_tok] {
        b.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for &id in &ids[..n_tok * n_used] {
        b.extend_from_slice(&id.to_le_bytes());
    }
    w.write_all(&b)
}

impl Recorded {
    /// Reads records ([`write_record`]); a truncated last record (a server stopped while
    /// writing) is dropped.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut out = Self::default();
        let mut at = 0;
        let u = |b: &[u8], i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        while at + HEADER <= bytes.len() {
            let (il, n_expert, n_used, n_tok) = (
                u(bytes, at),
                u(bytes, at + 4),
                u(bytes, at + 8),
                u(bytes, at + 12),
            );
            let len = n_used * n_tok * 4;
            if at + HEADER + len > bytes.len() {
                break;
            }
            if n_expert == 0 || n_expert > u16::MAX as usize || n_used == 0 {
                return Err(format!(
                    "record at byte {at}: {n_expert} experts, {n_used} used"
                ));
            }
            if out.n_expert == 0 {
                out.n_expert = n_expert;
            } else if out.n_expert != n_expert {
                return Err(format!(
                    "record at byte {at}: {n_expert} experts, earlier ones {}",
                    out.n_expert
                ));
            }
            let rows = out.layers.entry(il).or_insert_with(|| Rows {
                k: n_used,
                ids: Vec::new(),
            });
            if rows.k != n_used {
                return Err(format!(
                    "record at byte {at}: layer {il} with {n_used} ids per token, earlier {}",
                    rows.k
                ));
            }
            let data = &bytes[at + HEADER..at + HEADER + len];
            for t in 0..n_tok {
                let tok = &data[t * n_used * 4..(t + 1) * n_used * 4];
                let ids: Vec<i32> = tok
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c))
                    .collect();
                // Padding rows (no valid id) are skipped.
                if ids.iter().all(|&e| e >= 0 && (e as usize) < n_expert) {
                    rows.ids.extend(ids.iter().map(|&e| e as u16));
                }
            }
            at += HEADER + len;
        }
        Ok(out)
    }

    /// Adds another record's tokens.
    pub fn extend(&mut self, other: Recorded) -> Result<(), String> {
        if self.n_expert != 0 && other.n_expert != 0 && self.n_expert != other.n_expert {
            return Err(format!(
                "records of {} and {} experts",
                self.n_expert, other.n_expert
            ));
        }
        if self.n_expert == 0 {
            self.n_expert = other.n_expert;
        }
        for (il, rows) in other.layers {
            let mine = self.layers.entry(il).or_insert_with(|| Rows {
                k: rows.k,
                ids: Vec::new(),
            });
            if mine.k != rows.k {
                return Err(format!(
                    "layer {il}: {} vs {} ids per token",
                    mine.k, rows.k
                ));
            }
            mine.ids.extend(rows.ids);
        }
        Ok(())
    }
}

/// The mean over `rows`' tokens of the most experts any one rank holds of the token's
/// (`owner`: each expert's rank). `k / ranks` is the floor; the contiguous split's value is
/// what a rank waits on in decode today.
pub fn mean_max(rows: &Rows, owner: &[u8], ranks: usize) -> f64 {
    let n = rows.tokens();
    if n == 0 {
        return 0.0;
    }
    let mut total = 0usize;
    let mut count = vec![0usize; ranks];
    for tok in rows.ids.chunks_exact(rows.k) {
        count.fill(0);
        for &e in tok {
            count[owner[e as usize] as usize] += 1;
        }
        total += count.iter().max().copied().unwrap_or(0);
    }
    total as f64 / n as f64
}

/// Splits a layer's `n_expert` experts among `ranks` ranks, `n_expert / ranks` each, to
/// minimise [`mean_max`] over `rows`:
///
/// 1. Greedy by co-occurrence: experts in order of how often they are picked (most first),
///    each to the rank (with room) holding the fewest of its co-picks so far, then the least
///    picked load: experts often picked together end up on different ranks.
/// 2. Swaps: an expert of one rank for one of another while that lowers the summed
///    max-over-ranks count of the tokens (the exact objective), up to `passes` sweeps.
///
/// Deterministic: ties go to the lower expert id and the lower rank. With no tokens, the
/// contiguous split.
pub fn split(rows: &Rows, n_expert: usize, ranks: usize, passes: usize) -> Vec<u8> {
    let n = rows.tokens();
    if n == 0 || ranks < 2 || !n_expert.is_multiple_of(ranks) {
        return contiguous(n_expert, ranks);
    }
    let cap = n_expert / ranks;
    // Picks and co-picks.
    let mut freq = vec![0u64; n_expert];
    let mut co = vec![0u32; n_expert * n_expert];
    for tok in rows.ids.chunks_exact(rows.k) {
        for (i, &a) in tok.iter().enumerate() {
            freq[a as usize] += 1;
            for &b in &tok[i + 1..] {
                co[a as usize * n_expert + b as usize] += 1;
                co[b as usize * n_expert + a as usize] += 1;
            }
        }
    }
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_by(|&a, &b| freq[b].cmp(&freq[a]).then(a.cmp(&b)));
    const NONE: u8 = u8::MAX;
    let mut owner = vec![NONE; n_expert];
    let mut size = vec![0usize; ranks];
    let mut load = vec![0u64; ranks];
    // Per rank, the co-picks of each expert with the experts placed there so far.
    let mut with = vec![0u64; ranks * n_expert];
    for &e in &order {
        let r = (0..ranks)
            .filter(|&r| size[r] < cap)
            .min_by_key(|&r| (with[r * n_expert + e], load[r], r))
            .expect("room on some rank");
        owner[e] = r as u8;
        size[r] += 1;
        load[r] += freq[e];
        for f in 0..n_expert {
            with[r * n_expert + f] += co[e * n_expert + f] as u64;
        }
    }
    improve(rows, &mut owner, ranks, passes);
    owner
}

/// Step 2 of [`split`]: pairwise swaps on the exact objective.
fn improve(rows: &Rows, owner: &mut [u8], ranks: usize, passes: usize) {
    let n_expert = owner.len();
    let n = rows.tokens();
    // The tokens that picked each expert.
    let mut inc: Vec<Vec<u32>> = vec![Vec::new(); n_expert];
    for (t, tok) in rows.ids.chunks_exact(rows.k).enumerate() {
        for &e in tok {
            inc[e as usize].push(t as u32);
        }
    }
    let mut count = vec![0i32; n * ranks];
    for (t, tok) in rows.ids.chunks_exact(rows.k).enumerate() {
        for &e in tok {
            count[t * ranks + owner[e as usize] as usize] += 1;
        }
    }
    let max_of = |c: &[i32]| c.iter().copied().max().unwrap_or(0);
    // The change of token t's max when one of its experts moves from rank `from` to `to`.
    let moved = |count: &[i32], t: usize, from: usize, to: usize| -> i32 {
        let c = &count[t * ranks..(t + 1) * ranks];
        let before = max_of(c);
        let mut after = 0;
        for (r, &v) in c.iter().enumerate() {
            let v = v - i32::from(r == from) + i32::from(r == to);
            after = after.max(v);
        }
        after - before
    };
    // Co-picks, for the bound below.
    let mut co = vec![0i32; n_expert * n_expert];
    for tok in rows.ids.chunks_exact(rows.k) {
        for (i, &a) in tok.iter().enumerate() {
            for &b in &tok[i + 1..] {
                co[a as usize * n_expert + b as usize] += 1;
                co[b as usize * n_expert + a as usize] += 1;
            }
        }
    }
    // gain[e * ranks + r]: the change of the objective if expert e alone moved to rank r. A
    // swap of a (to b's rank) and b (to a's) changes it by gain(a) + gain(b), less what the
    // tokens that picked both added there (they keep their counts): at most 2 per such token,
    // so a pair whose bound cannot beat the best swap so far is not looked at.
    let mut gain = vec![0i32; n_expert * ranks];
    let mut stale = true;
    let mut mark = vec![0u32; n];
    let mut stamp = 0u32;
    for _ in 0..passes {
        let mut better = false;
        for a in 0..n_expert {
            if inc[a].is_empty() {
                continue;
            }
            if stale {
                gain.fill(0);
                for (t, tok) in rows.ids.chunks_exact(rows.k).enumerate() {
                    for &e in tok {
                        let from = owner[e as usize] as usize;
                        for to in (0..ranks).filter(|&to| to != from) {
                            gain[e as usize * ranks + to] += moved(&count, t, from, to);
                        }
                    }
                }
                stale = false;
            }
            let ra = owner[a] as usize;
            let mut best = (0i32, usize::MAX);
            for b in 0..n_expert {
                let rb = owner[b] as usize;
                if rb == ra {
                    continue;
                }
                let bound = gain[a * ranks + rb] + gain[b * ranks + ra] - 2 * co[a * n_expert + b];
                if bound >= best.0 {
                    continue;
                }
                // Tokens that picked both keep their counts.
                stamp += 1;
                for &t in &inc[b] {
                    mark[t as usize] = stamp;
                }
                let mut d = 0;
                for &t in &inc[a] {
                    if mark[t as usize] != stamp {
                        d += moved(&count, t as usize, ra, rb);
                    }
                }
                stamp += 1;
                for &t in &inc[a] {
                    mark[t as usize] = stamp;
                }
                for &t in &inc[b] {
                    if mark[t as usize] != stamp {
                        d += moved(&count, t as usize, rb, ra);
                    }
                }
                if d < best.0 {
                    best = (d, b);
                }
            }
            if best.1 == usize::MAX {
                continue;
            }
            let b = best.1;
            let rb = owner[b] as usize;
            stamp += 1;
            for &t in &inc[b] {
                mark[t as usize] = stamp;
            }
            let both: Vec<bool> = inc[a].iter().map(|&t| mark[t as usize] == stamp).collect();
            for (&t, both) in inc[a].iter().zip(both) {
                if !both {
                    count[t as usize * ranks + ra] -= 1;
                    count[t as usize * ranks + rb] += 1;
                }
            }
            stamp += 1;
            for &t in &inc[a] {
                mark[t as usize] = stamp;
            }
            for &t in &inc[b] {
                if mark[t as usize] != stamp {
                    count[t as usize * ranks + rb] -= 1;
                    count[t as usize * ranks + ra] += 1;
                }
            }
            owner.swap(a, b);
            better = true;
            stale = true;
        }
        if !better {
            break;
        }
    }
}

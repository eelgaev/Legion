//! Routed experts' host copies split by NUMA node (expert parallelism over GPUs on several
//! sockets, GLM's `--glm-parallel-decode`): each GPU reads the experts it owns from its own
//! socket's memory. The arena (`cuda::host_arena_split`) gives every expert tensor one
//! physical piece per run of experts on the same node.

#[cfg(test)]
mod tests;

/// The pieces `(bytes, node)` of an expert tensor of `alloc` bytes (a multiple of `gran`)
/// holding experts of `per` bytes each back to back, expert `e` wanted on node `node[e]`: one
/// piece per run of experts on the same node, cut at the granule at or below the run's first
/// byte (so a run's first expert may start in the piece before). A run shorter than a granule
/// stays with the run before it.
pub fn expert_pieces(node: &[u32], per: usize, alloc: usize, gran: usize) -> Vec<(usize, u32)> {
    assert!(gran > 0 && alloc.is_multiple_of(gran));
    let mut out: Vec<(usize, u32)> = Vec::new();
    let mut start = 0usize;
    for (e, &n) in node.iter().enumerate() {
        let at = (e * per / gran * gran).min(alloc);
        match out.last_mut() {
            Some(last) if last.1 == n => {}
            Some(last) if at > start => {
                last.0 = at - start;
                start = at;
                out.push((0, n));
            }
            Some(_) => {}
            None => out.push((0, n)),
        }
    }
    match out.last_mut() {
        Some(last) => last.0 = alloc - start,
        None => out.push((alloc, 0)),
    }
    out.retain(|p| p.0 > 0);
    out
}

/// The bytes of the experts `range` (of `per` bytes each) that lie in pieces on a node other
/// than `want`: what a GPU on node `want` reading those experts reads across the sockets.
pub fn foreign_bytes(
    pieces: &[(usize, u32)],
    per: usize,
    range: std::ops::Range<usize>,
    want: u32,
) -> usize {
    let (lo, hi) = (range.start * per, range.end * per);
    let mut at = 0;
    let mut out = 0;
    for &(size, n) in pieces {
        let (a, b) = (at.max(lo), (at + size).min(hi));
        if n != want && b > a {
            out += b - a;
        }
        at += size;
    }
    out
}

/// The bytes `range` of a tensor whose host bytes lie in `pieces` (`(bytes, node)` back to back
/// from the tensor's first byte, [`expert_pieces`]) cut where the pieces' nodes change: each
/// cut with its node (#720: the loader fills each piece from its own node's CPUs, since a GPU
/// reads memory written by the other socket's CPUs at about two thirds of the speed). Bytes
/// past the pieces stay with the last piece's node; without pieces, `range` whole on `node`.
pub fn cut_by_node(
    range: std::ops::Range<usize>,
    pieces: &[(usize, u32)],
    node: Option<u32>,
) -> Vec<(std::ops::Range<usize>, Option<u32>)> {
    let mut out: Vec<(std::ops::Range<usize>, Option<u32>)> = Vec::new();
    let mut at = 0usize;
    for (i, &(size, n)) in pieces.iter().enumerate() {
        let end = if i + 1 == pieces.len() {
            usize::MAX
        } else {
            at + size
        };
        let (a, b) = (at.max(range.start), end.min(range.end));
        if b > a {
            match out.last_mut() {
                Some(last) if last.1 == Some(n) && last.0.end == a => last.0.end = b,
                _ => out.push((a..b, Some(n))),
            }
        }
        at = end;
        if at >= range.end {
            break;
        }
    }
    if out.is_empty() && !range.is_empty() {
        out.push((range, node));
    }
    out
}

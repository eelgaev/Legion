//! Per-node row digests of prompt-step graphs, for finding the op whose rows depend on the step
//! they run in (#1183, development only): with `LEGION_NODE_DIGEST=<dir>` (CUDA), a graph of
//! `LEGION_NODE_DIGEST_MIN` (1024) tokens or more runs node by node like the op profile
//! (`op_profile::compute`, so ggml's CUDA graphs and fusions are off for it), and after each
//! node its output is read back and hashed per token row. Two files per graph go to the
//! directory: `g<seq>.idx` (one text line per hashed node: index, op, name, shape, type, sources,
//! the token dimension and the digest count) and `g<seq>.bin` (little-endian u32 digests: per
//! node the digest of its first `LEGION_NODE_DIGEST_HEAD` (4096) rows, then one per row for its
//! last `LEGION_NODE_DIGEST_TAIL` (2086) rows; a node without a token dimension has one digest
//! of all its bytes). Run the same prompt as one step and as two, then compare the tail rows:
//! the first node whose digests differ is the op that made them differ.
//!
//! At most `LEGION_NODE_DIGEST_GRAPHS` (8) graphs and their first `LEGION_NODE_DIGEST_NODES`
//! (all) nodes; graphs past that, and smaller graphs, compute as without the variable.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::raw::{c_int, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::external::op_of;
use super::ffi;

unsafe extern "C" {
    fn cudaMallocHost(p: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFreeHost(p: *mut c_void) -> c_int;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
}

/// `cudaMemcpyDefault`: the direction from the pointers (unified addressing).
const MEMCPY_DEFAULT: c_int = 4;

struct Config {
    dir: String,
    min_tokens: i64,
    head: usize,
    tail: usize,
    graphs: usize,
    nodes: usize,
}

fn env_num(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn config() -> Option<&'static Config> {
    static C: OnceLock<Option<Config>> = OnceLock::new();
    C.get_or_init(|| {
        let dir = std::env::var("LEGION_NODE_DIGEST")
            .ok()
            .filter(|d| !d.is_empty())?;
        std::fs::create_dir_all(&dir).ok()?;
        Some(Config {
            dir,
            min_tokens: env_num("LEGION_NODE_DIGEST_MIN", 1024) as i64,
            head: env_num("LEGION_NODE_DIGEST_HEAD", 4096),
            tail: env_num("LEGION_NODE_DIGEST_TAIL", 2086),
            graphs: env_num("LEGION_NODE_DIGEST_GRAPHS", 8),
            nodes: env_num("LEGION_NODE_DIGEST_NODES", usize::MAX),
        })
    })
    .as_ref()
}

/// Whether `LEGION_NODE_DIGEST` is set.
pub(crate) fn enabled() -> bool {
    config().is_some()
}

static GRAPHS: AtomicUsize = AtomicUsize::new(0);

/// Mixes one 64-bit word into a hash.
fn mix(h: u64, w: u64) -> u64 {
    (h ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29)
}

/// A hash of `bytes` (whole 8-byte words, then the rest zero-padded).
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ bytes.len() as u64;
    let mut it = bytes.chunks_exact(8);
    for c in &mut it {
        h = mix(h, u64::from_le_bytes(c.try_into().unwrap()));
    }
    let r = it.remainder();
    if !r.is_empty() {
        let mut w = [0u8; 8];
        w[..r.len()].copy_from_slice(r);
        h = mix(h, u64::from_le_bytes(w));
    }
    h
}

fn fold(h: u64) -> u32 {
    (h ^ (h >> 32)) as u32
}

/// One digest per index of dimension `d` (1..=3) of a tensor of shape `ne` and byte strides
/// `nb` whose dimension-0 rows are `row` contiguous bytes at those strides in `bytes`: every
/// row with that index hashed into it, in order.
pub(crate) fn row_digests(
    bytes: &[u8],
    ne: [i64; 4],
    nb: [usize; 4],
    row: usize,
    d: usize,
) -> Vec<u64> {
    assert!((1..=3).contains(&d), "token dimension {d}");
    let n = ne[d].max(0) as usize;
    // The digest of index `t`: its rows in memory order (i3, then i2, then i1).
    let one = |t: usize| {
        let mut h = 0x8422_2325_cbf2_9ce4u64;
        let range = |k: usize| {
            if k == d {
                t..t + 1
            } else {
                0..ne[k].max(1) as usize
            }
        };
        for i3 in range(3) {
            for i2 in range(2) {
                for i1 in range(1) {
                    let off = i1 * nb[1] + i2 * nb[2] + i3 * nb[3];
                    h = mix(h, hash_bytes(&bytes[off..off + row]));
                }
            }
        }
        h
    };
    let mut out = vec![0u64; n];
    // Large tensors (hundreds of MB at a 6K-token step) hash on several threads.
    let threads = if bytes.len() > 8 << 20 { 16 } else { 1 };
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (c, chunk) in out.chunks_mut(per).enumerate() {
            let one = &one;
            s.spawn(move || {
                for (j, o) in chunk.iter_mut().enumerate() {
                    *o = one(c * per + j);
                }
            });
        }
    });
    out
}

/// The step's tokens of graph `g`: the most common `ne[1]` of its row-wise ops (adds, muls,
/// norms), which have shape `[width, tokens]`. Not the matmuls: `--prompt-step-grid-split`
/// runs a folded step's projections over parts of its tokens.
///
/// # Safety
/// `g` must be a valid graph.
unsafe fn step_tokens(g: *mut ffi::ggml_cgraph) -> i64 {
    unsafe {
        let mut counts: std::collections::HashMap<i64, usize> = Default::default();
        for i in 0..ffi::ggml_graph_n_nodes(g) {
            let n = ffi::ggml_graph_node(g, i);
            if matches!(
                (*n).op,
                ffi::ggml_op_GGML_OP_ADD | ffi::ggml_op_GGML_OP_MUL | ffi::ggml_op_GGML_OP_RMS_NORM
            ) && (*n).ne[2] == 1
                && (*n).ne[3] == 1
            {
                *counts.entry((*n).ne[1]).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .max_by_key(|&(t, c)| (c, t))
            .map_or(0, |(t, _)| t)
    }
}

fn cstr(p: *const std::os::raw::c_char) -> String {
    // SAFETY: ggml names and descriptions are NUL-terminated.
    let s = unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .replace(' ', "_");
    if s.is_empty() { "-".to_string() } else { s }
}

/// One graph's digest files and a pinned staging buffer.
pub(crate) struct Sink {
    cfg: &'static Config,
    tokens: i64,
    idx: BufWriter<File>,
    bin: BufWriter<File>,
    host: *mut c_void,
    cap: usize,
    done: usize,
}

impl Drop for Sink {
    fn drop(&mut self) {
        let _ = self.idx.flush();
        let _ = self.bin.flush();
        if !self.host.is_null() {
            // SAFETY: allocated by cudaMallocHost below.
            unsafe { cudaFreeHost(self.host) };
        }
    }
}

/// A sink for graph `g` if digests are on and it is a prompt step to hash.
///
/// # Safety
/// `g` must be a valid graph.
pub(crate) unsafe fn begin(g: *mut ffi::ggml_cgraph) -> Option<Sink> {
    let cfg = config()?;
    let tokens = unsafe { step_tokens(g) };
    if tokens < cfg.min_tokens {
        return None;
    }
    let seq = GRAPHS.fetch_add(1, Ordering::Relaxed);
    if seq >= cfg.graphs {
        return None;
    }
    let n = unsafe { ffi::ggml_graph_n_nodes(g) };
    let base = format!("{}/g{seq:03}", cfg.dir);
    let mut idx = BufWriter::new(File::create(format!("{base}.idx")).ok()?);
    let bin = BufWriter::new(File::create(format!("{base}.bin")).ok()?);
    writeln!(idx, "graph {seq} tokens {tokens} nodes {n}").ok()?;
    eprintln!("node digest: graph {seq}, {tokens} tokens, {n} nodes -> {base}.idx");
    Some(Sink {
        cfg,
        tokens,
        idx,
        bin,
        host: std::ptr::null_mut(),
        cap: 0,
        done: 0,
    })
}

impl Sink {
    /// Hashes node `i` (`node`, just launched on `stream`) once it is done.
    ///
    /// # Safety
    /// `node` is a computed node of the sink's graph whose data is live; `stream` its stream.
    pub(crate) unsafe fn node(
        &mut self,
        i: c_int,
        node: *const ffi::ggml_tensor,
        stream: *mut c_void,
    ) {
        unsafe {
            if self.done >= self.cfg.nodes {
                return;
            }
            self.done += 1;
            let t = &*node;
            if matches!(
                t.op,
                ffi::ggml_op_GGML_OP_NONE
                    | ffi::ggml_op_GGML_OP_VIEW
                    | ffi::ggml_op_GGML_OP_RESHAPE
                    | ffi::ggml_op_GGML_OP_PERMUTE
                    | ffi::ggml_op_GGML_OP_TRANSPOSE
            ) || t.data.is_null()
            {
                return;
            }
            let bytes = ffi::ggml_nbytes(node);
            if bytes == 0 {
                return;
            }
            let ne = t.ne;
            let nb = t.nb;
            let row = ffi::ggml_row_size(t.type_, ne[0]);
            let tsize = ffi::ggml_type_size(t.type_);
            let d = (1..=3usize)
                .rev()
                .find(|&d| ne[d] == self.tokens)
                .filter(|_| nb[0] == tsize || ne[0] == 1);
            let op = match op_of(node) {
                Some(ext) => ext.name.to_string(),
                None => cstr(ffi::ggml_op_desc(node)),
            };
            let srcs: Vec<String> = t
                .src
                .iter()
                .filter(|s| !s.is_null())
                .map(|&s| {
                    format!(
                        "{}:{}",
                        cstr((*s).name.as_ptr()),
                        cstr(ffi::ggml_type_name((*s).type_))
                    )
                })
                .collect();
            // Without rows (a cache written in place, a flat buffer), only small tensors.
            let skip = bytes > if d.is_some() { 4 << 30 } else { 64 << 20 };
            if !skip && bytes > self.cap {
                if !self.host.is_null() {
                    cudaFreeHost(self.host);
                    self.host = std::ptr::null_mut();
                }
                let cap = bytes.next_power_of_two();
                if cudaMallocHost(&mut self.host, cap) != 0 {
                    self.host = std::ptr::null_mut();
                    self.cap = 0;
                    return;
                }
                self.cap = cap;
            }
            if !skip && cudaMemcpyAsync(self.host, t.data, bytes, MEMCPY_DEFAULT, stream) != 0 {
                return;
            }
            cudaStreamSynchronize(stream);
            let data = if skip {
                &[][..]
            } else {
                std::slice::from_raw_parts(self.host as *const u8, bytes)
            };
            let digests: Vec<u32> = match d {
                _ if skip => Vec::new(),
                Some(d) => {
                    let rows = row_digests(data, ne, nb, row, d);
                    let head = rows.len().min(self.cfg.head);
                    let h = rows[..head].iter().fold(0u64, |h, &r| mix(h, r));
                    let tail = rows.len().min(self.cfg.tail);
                    std::iter::once(fold(h))
                        .chain(rows[rows.len() - tail..].iter().map(|&r| fold(r)))
                        .collect()
                }
                None => vec![fold(hash_bytes(data))],
            };
            let _ = writeln!(
                self.idx,
                "{i} {op} {} {}x{}x{}x{} {} {} d={} {}",
                cstr(t.name.as_ptr()),
                ne[0],
                ne[1],
                ne[2],
                ne[3],
                cstr(ffi::ggml_type_name(t.type_)),
                if srcs.is_empty() {
                    "-".to_string()
                } else {
                    srcs.join(",")
                },
                d.map_or(-1, |d| d as i64),
                digests.len()
            );
            for v in digests {
                let _ = self.bin.write_all(&v.to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests;

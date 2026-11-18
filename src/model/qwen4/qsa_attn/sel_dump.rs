//! `LEGION_DEBUG_QSA_SEL` (docs/debug-env.md, #1176): how much neighbouring prompt tokens'
//! sparse-attention selections overlap, per prompt-step call of [`super::attention`], and the
//! selection rows of chosen calls for `legion qsa-bench --sel`.
//!
//! `LEGION_DEBUG_QSA_SEL=<prefix>[,keep=<call>.<call>...]`: every prompt step's attention call
//! (more than [`super::DECODE_TOKENS`] tokens) copies its selection rows to the host after the
//! launch (a stream wait, so for measurements only) and prints one `qsa-sel` line of
//! [`Overlap`]s; calls numbered in `keep` also go to `<prefix>-<call>.bin` (two I32s, the rows
//! and the row length, then the rows as I32). A capture never holds the op while this is set.

use std::sync::OnceLock;

#[cfg(test)]
mod tests;

/// The parsed switch.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Config {
    prefix: String,
    keep: Vec<u64>,
}

fn config() -> Option<&'static Config> {
    static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let v = std::env::var("LEGION_DEBUG_QSA_SEL").ok()?;
            let mut parts = v.split(',');
            let prefix = parts.next()?.to_string();
            let mut keep = Vec::new();
            for p in parts {
                match p.strip_prefix("keep=") {
                    Some(list) => {
                        keep.extend(list.split('.').filter_map(|x| x.parse::<u64>().ok()))
                    }
                    None => eprintln!("LEGION_DEBUG_QSA_SEL: ignored {p}"),
                }
            }
            Some(Config { prefix, keep })
        })
        .as_ref()
}

/// Whether selections are being measured.
pub fn enabled() -> bool {
    config().is_some()
}

/// The overlap of groups of `n` neighbouring tokens' selections (sets of positions).
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Overlap {
    pub n: usize,
    /// Groups measured (whole groups of `n` neighbours in one sequence).
    pub groups: u64,
    /// Selected keys over the groups' tokens.
    pub sum: u64,
    /// Distinct keys per group, summed.
    pub union: u64,
    /// Keys every token of a group selects, summed.
    pub all: u64,
    /// Rows a CTA of the group would load if its tokens ran their chunks of 32 in lock step and
    /// kept the rows the next chunks share: the union of the tokens' chunk `i`, less what
    /// chunk `i - 1`'s union already holds, summed over `i`.
    pub lock: u64,
    /// Keys the group's first and last tokens share, by how far apart (in chunks of 32) they
    /// sit in the two lists: 0, 1, 2 or more.
    pub drift: [u64; 3],
}

/// [`Overlap`] for groups of `n` of `rows` (each `row` I32s: the count, then the positions
/// ascending), `n_tok` rows a sequence.
pub fn overlap(rows: &[i32], row: usize, n_tok: usize, n: usize) -> Overlap {
    let list = |r: usize| {
        let x = &rows[r * row..(r + 1) * row];
        let c = (x[0].max(0) as usize).min(row - 1);
        &x[1..1 + c]
    };
    let mut o = Overlap {
        n,
        ..Overlap::default()
    };
    let n_rows = rows.len() / row;
    let mut r = 0;
    while r + n <= n_rows {
        // A group stays inside one sequence.
        if r / n_tok != (r + n - 1) / n_tok {
            r = (r / n_tok + 1) * n_tok;
            continue;
        }
        let lists: Vec<&[i32]> = (r..r + n).map(list).collect();
        let mut u: Vec<i32> = lists.iter().flat_map(|l| l.iter().copied()).collect();
        o.sum += u.len() as u64;
        u.sort_unstable();
        let mut run = 0;
        for i in 0..u.len() {
            run = if i > 0 && u[i] == u[i - 1] {
                run + 1
            } else {
                1
            };
            if run == n {
                o.all += 1;
            }
        }
        u.dedup();
        o.union += u.len() as u64;
        let chunks = lists
            .iter()
            .map(|l| l.len().div_ceil(32))
            .max()
            .unwrap_or(0);
        let mut prev: Vec<i32> = Vec::new();
        for i in 0..chunks {
            let mut cur: Vec<i32> = lists
                .iter()
                .flat_map(|l| l.iter().skip(32 * i).take(32).copied())
                .collect();
            cur.sort_unstable();
            cur.dedup();
            o.lock += cur
                .iter()
                .filter(|p| prev.binary_search(p).is_err())
                .count() as u64;
            prev = cur;
        }
        let (a, b) = (lists[0], lists[n - 1]);
        for (ia, p) in a.iter().enumerate() {
            if let Ok(ib) = b.binary_search(p) {
                o.drift[(ia / 32).abs_diff(ib / 32).min(2)] += 1;
            }
        }
        o.groups += 1;
        r += n;
    }
    o
}

/// The K / V rows `attn_gp_kernel` loads for `rows` (as [`overlap`]'s) and its steps: each pair
/// of neighbouring tokens of a sequence (an odd last token on its own) runs the position window
/// (the pending chunk of 32 that ends first, the other too if it starts by then), keeping the
/// rows of a step the next step needs again.
pub fn pair_window(rows: &[i32], row: usize, n_tok: usize) -> (u64, u64) {
    let list = |r: usize| {
        let x = &rows[r * row..(r + 1) * row];
        let c = (x[0].max(0) as usize).min(row - 1);
        &x[1..1 + c]
    };
    fn chunk(l: &[i32], i: usize) -> &[i32] {
        &l[i.min(l.len())..(i + 32).min(l.len())]
    }
    let (mut loaded, mut steps) = (0u64, 0u64);
    for s in 0..rows.len() / row / n_tok.max(1) {
        for a in (0..n_tok).step_by(2) {
            let la = list(s * n_tok + a);
            let lb: &[i32] = if a + 1 < n_tok {
                list(s * n_tok + a + 1)
            } else {
                &[]
            };
            let (mut ia, mut ib) = (0, 0);
            let mut held: Vec<i32> = Vec::new();
            loop {
                let (ca, cb) = (chunk(la, ia), chunk(lb, ib));
                let ea = ca.last().copied().unwrap_or(i32::MAX);
                let eb = cb.last().copied().unwrap_or(i32::MAX);
                let aa = !ca.is_empty() && (ea <= eb || ca[0] <= eb);
                let ab = !cb.is_empty() && (eb <= ea || cb[0] <= ea);
                if !aa && !ab {
                    break;
                }
                let mut need: Vec<i32> = Vec::with_capacity(64);
                if aa {
                    need.extend_from_slice(ca);
                    ia += 32;
                }
                if ab {
                    need.extend_from_slice(cb);
                    ib += 32;
                }
                need.sort_unstable();
                need.dedup();
                loaded += need.iter().filter(|p| !held.contains(p)).count() as u64;
                held = need;
                steps += 1;
            }
        }
    }
    (loaded, steps)
}

/// One `qsa-sel` line: the call, its rows, the positions of its first and last tokens, the
/// mean keys a token, and per group size: shared by all (% of the sum), union / sum, lock-step
/// rows / sum, drift of shared keys in chunks (% at 0 / 1 / 2+).
pub fn line(call: u64, rows: &[i32], row: usize, n_tok: usize) -> String {
    let n_rows = rows.len() / row;
    let last = |r: usize| {
        let c = rows[r * row].max(0) as usize;
        if c == 0 { -1 } else { rows[r * row + c] }
    };
    let keys: u64 = (0..n_rows).map(|r| rows[r * row].max(0) as u64).sum();
    let mut s = format!(
        "qsa-sel call {call} rows {n_rows} pos {}..{} keys {:.1}",
        last(0),
        last(n_rows.saturating_sub(1)),
        keys as f64 / n_rows.max(1) as f64
    );
    for n in [2, 4, 8] {
        let o = overlap(rows, row, n_tok, n);
        let sum = o.sum.max(1) as f64;
        let d = o.drift.iter().sum::<u64>().max(1) as f64;
        s += &format!(
            " | N{n} all {:.1}% union/sum {:.3} lock/sum {:.3} drift {:.0}/{:.0}/{:.0}%",
            o.all as f64 * n as f64 / sum * 100.0,
            o.union as f64 / sum,
            o.lock as f64 / sum,
            o.drift[0] as f64 / d * 100.0,
            o.drift[1] as f64 / d * 100.0,
            o.drift[2] as f64 / d * 100.0
        );
    }
    s
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicU64, Ordering};

    unsafe extern "C" {
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int) -> c_int;
        fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
    }

    const DEVICE_TO_HOST: c_int = 2;

    /// After a prompt step's launch: copies `rows` selection rows of `row` I32s at `sel` (device
    /// memory) to the host, prints their overlap and keeps them when asked. Never fails the op.
    ///
    /// # Safety
    /// `sel` holds `rows × row` I32s on `stream`'s device.
    pub(in super::super) unsafe fn after(
        sel: *const i32,
        rows: usize,
        row: usize,
        n_tok: usize,
        stream: *mut c_void,
    ) {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        let Some(cfg) = super::config() else {
            return;
        };
        let call = CALLS.fetch_add(1, Ordering::Relaxed);
        let mut v = vec![0i32; rows * row];
        // SAFETY: the caller's; the stream is synchronized first.
        let e = unsafe {
            match cudaStreamSynchronize(stream) {
                0 => cudaMemcpy(
                    v.as_mut_ptr().cast(),
                    sel.cast(),
                    v.len() * 4,
                    DEVICE_TO_HOST,
                ),
                e => e,
            }
        };
        if e != 0 {
            eprintln!("LEGION_DEBUG_QSA_SEL: CUDA error {e}");
            return;
        }
        eprintln!("{}", super::line(call, &v, row, n_tok));
        if cfg.keep.contains(&call) {
            let path = format!("{}-{call}.bin", cfg.prefix);
            let bytes: Vec<u8> = [rows as i32, row as i32]
                .iter()
                .chain(v.iter())
                .flat_map(|x| x.to_le_bytes())
                .collect();
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("LEGION_DEBUG_QSA_SEL: {path}: {e}");
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub(super) use cuda::after;

/// Selection rows kept by `LEGION_DEBUG_QSA_SEL` (the rows, the row length, the I32s).
pub fn read(path: &str) -> std::io::Result<(usize, usize, Vec<i32>)> {
    let b = std::fs::read(path)?;
    let x: Vec<i32> = b
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes")))
        .collect();
    let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, "not a selection dump");
    let (&rows, &row) = (x.first().ok_or_else(bad)?, x.get(1).ok_or_else(bad)?);
    let (rows, row) = (rows as usize, row as usize);
    if row == 0 || x.len() != 2 + rows * row {
        return Err(bad());
    }
    Ok((rows, row, x[2..].to_vec()))
}

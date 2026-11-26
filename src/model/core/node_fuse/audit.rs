//! The up-mix audit (#1183): after the node fusion, every qwen4exp hyper-connection up-mix node
//! of a graph with what it reads and writes per token and who reads each output (node, op,
//! the dtype it reads), then the totals by stream. The up-mix kernel of prompt steps is bound
//! by DRAM bytes, so each line's share of the bytes is its share of the kernel's time.
//!
//! Logged at INFO for the first graph of each step size (up to [`MAX_GRAPHS`]) when the
//! target [`TARGET`] is on at DEBUG: `RUST_LOG=info,legion::model::core::node_fuse::audit=debug`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::{Graph, Node, is_view_op};
use crate::ggml::{ffi, op_of};

/// The log target (the module's path).
pub(super) const TARGET: &str = "legion::model::core::node_fuse::audit";

/// Graphs audited at most (one per step size).
const MAX_GRAPHS: usize = 4;

/// The step sizes audited so far.
static SIZES: Mutex<Vec<i64>> = Mutex::new(Vec::new());

/// Whether the audit is asked for ([`TARGET`] at DEBUG).
pub(super) fn enabled() -> bool {
    tracing::enabled!(target: TARGET, tracing::Level::DEBUG)
}

/// `t`'s name, or `-`.
///
/// # Safety
/// A valid tensor.
unsafe fn name(t: Node) -> String {
    // SAFETY: the caller's; the name is a NUL-terminated array.
    let s = unsafe { std::ffi::CStr::from_ptr((*t).name.as_ptr()) }.to_string_lossy();
    if s.is_empty() { "-".into() } else { s.into() }
}

/// The op `n` runs: an external op's name, else ggml's.
///
/// # Safety
/// A valid tensor.
unsafe fn op_name(n: Node) -> String {
    // SAFETY: the caller's; ggml's names are static strings.
    unsafe {
        match op_of(n) {
            Some(op) => op.name.into(),
            None => std::ffi::CStr::from_ptr(ffi::ggml_op_desc(n))
                .to_string_lossy()
                .into_owned(),
        }
    }
}

/// `t`'s type name.
///
/// # Safety
/// A valid tensor.
unsafe fn ty(t: Node) -> String {
    // SAFETY: the caller's; ggml's names are static strings.
    unsafe {
        std::ffi::CStr::from_ptr(ffi::ggml_type_name((*t).type_))
            .to_string_lossy()
            .into_owned()
    }
}

/// Whether `s` is `t` or a view of it (down the chain).
///
/// # Safety
/// Valid tensors.
unsafe fn is_of(mut s: Node, t: Node) -> bool {
    // SAFETY: the caller's; view sources are valid tensors.
    unsafe {
        loop {
            if s == t {
                return true;
            }
            if (*s).view_src.is_null() {
                return false;
            }
            s = (*s).view_src;
        }
    }
}

/// Every node of `g` but `writer` (views left out) reading `t` or a view of it: "op name reads
/// TYPE", with "(off)" for a node turned off (its values written by another).
///
/// # Safety
/// `t` a valid tensor; `g`'s nodes valid tensors.
unsafe fn readers(g: &Graph, t: Node, writer: Node) -> Vec<String> {
    // SAFETY: the caller's.
    unsafe {
        let mut out = Vec::new();
        for &n in &g.nodes {
            if is_view_op((*n).op) || n == t || n == writer {
                continue;
            }
            for &s in (*n).src.iter() {
                if !s.is_null() && is_of(s, t) {
                    let off = if (*n).op == ffi::ggml_op_GGML_OP_NONE {
                        " (off)"
                    } else {
                        ""
                    };
                    out.push(format!("{} {} reads {}{off}", op_name(n), name(n), ty(s)));
                    break;
                }
            }
        }
        out
    }
}

/// Where the up-mix's BF16 `lo` comes from: its `hc_up_lob` node `lob` and the chain before it.
///
/// # Safety
/// `lob` null or a valid tensor whose sources are valid.
unsafe fn lo_source(lob: Node) -> String {
    // SAFETY: the caller's.
    unsafe {
        if lob.is_null() {
            return "F32 (Q8_0 up: quantized by the kernel)".into();
        }
        let act = (*lob).src[0];
        if (*lob).op == ffi::ggml_op_GGML_OP_NONE {
            let down = if act.is_null() {
                std::ptr::null_mut()
            } else {
                (*act).src[0]
            };
            let by = if down.is_null() {
                "-".into()
            } else {
                op_name(down)
            };
            return format!("written by {by} (activation and BF16 rounding fused)");
        }
        let mut chain = vec!["hc_up_lob".to_string()];
        let mut t = act;
        for _ in 0..2 {
            if t.is_null() {
                break;
            }
            chain.push(op_name(t));
            t = (*t).src[0];
        }
        chain.join(" <- ")
    }
}

/// Logs the audit of `g` if it is the first graph of its step size with up-mix nodes (and not
/// more than [`MAX_GRAPHS`] were logged).
///
/// # Safety
/// `g` a graph of valid tensors, allocated and rewritten, not run.
pub(super) unsafe fn log(g: &Graph) {
    use crate::model::qwen4::mix_audit;
    // SAFETY: the caller's.
    unsafe {
        let mixes: Vec<(usize, crate::model::qwen4::MixAudit)> = g
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| mix_audit(t).map(|a| (i, a)))
            .collect();
        let Some(n) = mixes.iter().map(|(_, a)| a.n).max() else {
            return;
        };
        {
            let mut seen = SIZES.lock().unwrap_or_else(|e| e.into_inner());
            if seen.len() >= MAX_GRAPHS || seen.contains(&n) {
                return;
            }
            seen.push(n);
        }
        tracing::info!(
            target: TARGET,
            "up-mix audit: {} up-mix nodes, {n} tokens (bytes a token; the weights left out)",
            mixes.len()
        );
        // Bytes a token by stream, summed over the nodes, and by mode.
        let mut by_stream: BTreeMap<String, usize> = BTreeMap::new();
        let mut by_mode: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for (k, (i, a)) in mixes.iter().enumerate() {
            let mut total = 0;
            let reads: Vec<String> = a
                .reads
                .iter()
                .map(|&(what, b)| {
                    total += b;
                    *by_stream.entry(format!("read {what}")).or_default() += b;
                    format!("{what} {b}")
                })
                .collect();
            let mut outs = Vec::new();
            let mut written = Vec::new();
            for &(what, b, t) in &a.writes {
                total += b;
                *by_stream.entry(format!("write {what}")).or_default() += b;
                if t.is_null() {
                    outs.push(format!("{what}: not written"));
                    continue;
                }
                written.push(what);
                let r = readers(g, t, g.nodes[*i]);
                outs.push(format!(
                    "{what} {b} -> [{}]",
                    if r.is_empty() {
                        "no reader in this graph".into()
                    } else {
                        r.join("; ")
                    }
                ));
            }
            let mode = format!("{} writing {}", a.mode, written.join(" + "));
            let m = by_mode.entry(mode.clone()).or_default();
            m.0 += 1;
            m.1 += total;
            let node = g.nodes[*i];
            tracing::info!(
                target: TARGET,
                "up-mix {k}: node {i} {} (up {}), {} tokens, {mode}: reads {}; lo from {}; \
                 writes {}; {total} B a token, {:.1} MB a call",
                name(node),
                name((*node).src[0]),
                a.n,
                reads.join(", "),
                lo_source(a.lob),
                outs.join("; "),
                (total as f64 * a.n as f64) / 1e6
            );
        }
        let all: usize = by_stream.values().sum();
        for (mode, (count, bytes)) in &by_mode {
            tracing::info!(
                target: TARGET,
                "up-mix audit: {count} x {mode}: {} B a token each, {:.1} MB a call each",
                bytes / count.max(&1),
                (*bytes as f64 / *count.max(&1) as f64) * n as f64 / 1e6
            );
        }
        for (what, bytes) in &by_stream {
            tracing::info!(
                target: TARGET,
                "up-mix audit: {what}: {bytes} B a token over all nodes, {:.1}% of the bytes \
                 ({:.1} MB a {n}-token step)",
                100.0 * *bytes as f64 / all.max(1) as f64,
                (*bytes as f64 * n as f64) / 1e6
            );
        }
    }
}

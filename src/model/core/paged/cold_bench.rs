//! Cold calls for `legion paged-qm-bench --grid --cold` (#689 r19): the wide-grid kernels timed
//! per call the way a model's verify step meets them. In a model each attention layer's call
//! follows several milliseconds of other layers' work (their weights, their own K/V), so it
//! starts with its K/V out of L2 and its pages' translations gone; a bench that repeats one call
//! over one pool does not. Here one graph runs `calls` calls back to back on the stream, each
//! after what the mode puts in front of it:
//! - `warm`: nothing, layer 0 each time (the plain bench);
//! - `rot`: nothing, the layers' pools in turn (one buffer, as a model's KvPool);
//! - `read`: a 1 GiB streaming read (a different GiB of an 8 GiB buffer each call), layer 0;
//! - `rotread`: the read and the pools in turn (closest to the model);
//! - `l2`: a 48 MiB read (twice the GB10's L2, few pages), layer 0;
//! - `tlb`: one 4-byte read every 2 MiB over the 8 GiB (4,096 pages, ~128 KiB of L2), layer 0;
//! - `tlb64k`: one every 64 KiB over the 8 GiB;
//! - `pt64k` / `pt2m`: one element every 64 KiB / 2 MiB of the call's own K and V pools, right
//!   before it.
//!
//! Parts join with `+` (`rot+read+pt64k`). A call's time is the graph's time less the same graph
//! without the calls (the flushes alone), over the calls; the flushes are ggml's `sum_rows` /
//! `cont`. Nothing here changes a kernel.

use super::grid_bench;
use super::wmma_bench::{Group, fill};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::KvPool;

/// The flush buffer's bytes.
const FLUSH_BYTES: usize = 8 << 30;

/// The flushes' buffer (F32, [`FLUSH_BYTES`], zeroed), kept for every graph.
pub struct Flush {
    ctx: Context,
}

impl Flush {
    pub fn new(backend: &Backend) -> Self {
        let ctx = Context::new(1, 1);
        ctx.new_tensor(types::F32, &[(FLUSH_BYTES / 4) as i64])
            .named("cold_flush");
        ctx.alloc_tensors(backend.buffer_type())
            .expect("flush buffer");
        ctx.clear_buffers(0);
        Self { ctx }
    }

    fn buf(&self) -> Tensor<'_> {
        self.ctx.tensor("cold_flush").expect("flush tensor")
    }
}

/// The parts a mode joins with `+` (`rotread` is `rot+read`); `pt64k` / `pt2m` touch the call's
/// own layer's K and V pools (one element every 64 KiB / 2 MiB of each) right before it, after
/// the other parts.
pub const PARTS: [&str; 8] = [
    "warm", "rot", "read", "l2", "tlb", "tlb64k", "pt64k", "pt2m",
];

/// `mode`'s parts, checked.
fn parts(mode: &str) -> Vec<&str> {
    let m = if mode == "rotread" { "rot+read" } else { mode };
    let p: Vec<&str> = m.split('+').collect();
    for x in &p {
        assert!(
            PARTS.contains(x),
            "no cold mode part {x}; {PARTS:?} joined with +"
        );
    }
    p
}

/// Whether `mode` is valid ([`MODES`] or [`PARTS`] joined with `+`).
pub fn valid(mode: &str) -> bool {
    let m = if mode == "rotread" { "rot+read" } else { mode };
    m.split('+').all(|x| PARTS.contains(&x))
}

/// The nodes in front of call `c` (on layer pools `kp`, `vp`) in `mode`.
fn flush_nodes<'a>(
    ctx: &'a Context,
    buf: Tensor<'a>,
    kp: Tensor<'a>,
    vp: Tensor<'a>,
    mode: &str,
    c: usize,
) -> Vec<Tensor<'a>> {
    let read = |bytes: usize, slot: usize| {
        let rows = (bytes / 4096) as i64;
        let off = (slot % (FLUSH_BYTES / bytes)) * bytes;
        ctx.sum_rows(ctx.view_2d(buf, [1024, rows], 4096, off))
    };
    let touch = |t: Tensor<'a>, bytes: usize, stride: usize, off: usize| {
        let n = (bytes / stride) as i64;
        ctx.cont(ctx.view_2d(t, [1, n], stride, off))
    };
    let mut out = Vec::new();
    let p = parts(mode);
    for x in &p {
        match *x {
            "read" => out.push(read(1 << 30, c)),
            "l2" => out.push(read(48 << 20, c)),
            "tlb" => out.push(touch(buf, FLUSH_BYTES, 2 << 20, (c % 16) * 128)),
            "tlb64k" => out.push(touch(buf, FLUSH_BYTES, 64 << 10, (c % 16) * 128)),
            _ => {}
        }
    }
    for x in &p {
        let stride = match *x {
            "pt64k" => 64 << 10,
            "pt2m" => 2 << 20,
            _ => continue,
        };
        out.push(touch(kp, kp.nbytes(), stride, 0));
        out.push(touch(vp, vp.nbytes(), stride, 0));
    }
    out
}

/// One graph of `calls` calls of `kernel` on `g` in `mode` (`op` false: the flushes alone),
/// computed `reps` times a round: the best round's ms per call, and call 0's output (empty
/// without the calls).
#[allow(clippy::too_many_arguments)]
fn graph_ms(
    backend: &mut Backend,
    kv: &KvPool,
    layers: usize,
    flush: &Flush,
    g: &Group,
    kernel: &str,
    mode: &str,
    op: bool,
    calls: usize,
    reps: usize,
    rounds: usize,
    gap_ms: u64,
) -> (f64, Vec<f32>) {
    let (d, n_head, _) = g.heads;
    let ns = g.n_past.len();
    let rotate = parts(mode).contains(&"rot");
    let ctx = Context::new(64 * calls + 64, 16 * calls + 16);
    let q = ctx
        .new_tensor(
            types::F32,
            &[d as i64, n_head as i64, g.n_tok as i64, ns as i64],
        )
        .as_input();
    let tab = ctx
        .new_tensor(types::I32, &[g.width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let scratch = ctx.new_tensor(types::F32, &[1, 1, 1, 1]).as_input();
    let buf = flush.buf();
    let mut graph = ctx.new_graph(16 * calls + 16);
    let mut first = None;
    let mut flushes = 0;
    for c in 0..calls {
        let il = if rotate { c % layers.max(1) } else { 0 };
        let (kp, vp) = kv.pools(il);
        for f in flush_nodes(&ctx, buf, kp, vp, mode, c) {
            graph.expand(f.as_output());
            flushes += 1;
        }
        if op {
            let o = attention_tiered(
                &ctx, q, kp, vp, tab, past, scratch, g.scale, g.window, g.ring, None,
            )
            .as_output();
            graph.expand(o);
            first.get_or_insert(o);
        }
    }
    if first.is_none() && flushes == 0 {
        return (0.0, Vec::new());
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .expect("graph allocation");
    if let Some(o) = first {
        fill(q, tab, past, g);
        o.set_f32(&vec![f32::NAN; o.elements()]);
    }
    grid_bench::set_here(kernel);
    backend.compute(&graph).expect("compute");
    let out = first.map(|o| o.to_f32()).unwrap_or_default();
    let mut best = f64::INFINITY;
    for _ in 0..rounds.max(1) {
        let start = std::time::Instant::now();
        for _ in 0..reps.max(1) {
            if gap_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(gap_ms));
            }
            backend.compute(&graph).expect("compute");
        }
        let busy = start.elapsed().as_secs_f64() * 1e3 - (gap_ms * reps.max(1) as u64) as f64;
        let ms = busy / (reps.max(1) * calls) as f64;
        best = best.min(ms);
    }
    grid_bench::set_here("");
    (best, out)
}

/// The cold calls of `kernel` on `g` in `mode`, each graph run `gap_ms` after the host sleeps
/// that long (an idle GPU in between): (ms per call, the flush's ms per call, call 0's output).
/// Wall clock: a flush that runs slower beside the calls counts as the calls' time, so take a
/// call's own time from a kernel trace.
#[allow(clippy::too_many_arguments)]
pub fn time(
    backend: &mut Backend,
    kv: &KvPool,
    layers: usize,
    flush: &Flush,
    g: &Group,
    kernel: &str,
    mode: &str,
    calls: usize,
    reps: usize,
    rounds: usize,
    gap_ms: u64,
) -> (f64, f64, Vec<f32>) {
    let (both, out) = graph_ms(
        backend, kv, layers, flush, g, kernel, mode, true, calls, reps, rounds, gap_ms,
    );
    let (alone, _) = graph_ms(
        backend, kv, layers, flush, g, kernel, mode, false, calls, reps, rounds, gap_ms,
    );
    (both - alone, alone, out)
}

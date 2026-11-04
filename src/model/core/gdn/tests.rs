//! [`rows`] and [`commit`] against ggml's `gated_delta_net`: on the CPU (the reference
//! implementation, within rounding), on CUDA (ignored tests) bit for bit per token count, and
//! a microbenchmark of both against ggml's op (and its state copy).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// A case: `h` value heads over `nk` q / k heads, `nt` tokens, `ns` sequences.
#[derive(Clone, Copy, Debug)]
struct Case {
    h: i64,
    nk: i64,
    nt: i64,
    ns: i64,
}

/// What a case's run returns: (op rows, op states, ggml rows, ggml states).
type Out = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// The op's [`rows`] and [`commit`] and ggml's `gated_delta_net` over the same inputs on
/// `backend`. q and k are views of a pool one token longer (strided, as a commit reads them).
fn run(backend: &mut Backend, c: Case) -> Out {
    let Case { h, nk, nt, ns } = c;
    let ctx = Context::new(32, 32);
    let qp = ctx.new_tensor(types::F32, &[S, nk, nt + 1, ns]).as_input();
    let kp = ctx.new_tensor(types::F32, &[S, nk, nt + 1, ns]).as_input();
    let v = ctx.new_tensor(types::F32, &[S, h, nt, ns]).as_input();
    let g = ctx.new_tensor(types::F32, &[1, h, nt, ns]).as_input();
    let beta = ctx.new_tensor(types::F32, &[1, h, nt, ns]).as_input();
    let st_ggml = ctx.new_tensor(types::F32, &[S, S, h, ns]).as_input();
    let st_op = ctx.new_tensor(types::F32, &[S, S, h, ns]).as_input();
    let (nq, nk_) = (qp.nb(), kp.nb());
    let q = ctx.view_4d(qp, [S, nk, nt, ns], [nq[1], nq[2], nq[3]], 0);
    let k = ctx.view_4d(kp, [S, nk, nt, ns], [nk_[1], nk_[2], nk_[3]], 0);
    let r = rows(&ctx, q, k, v, g, beta, st_op).as_output();
    let cm = commit(&ctx, q, k, v, g, beta, st_op).as_output();
    let reference = ctx
        .gated_delta_net(q, k, v, g, beta, st_ggml, 1)
        .as_output();
    let mut graph = ctx.new_graph(32);
    // The rows read the input state, so they run before the commit overwrites it.
    for t in [r, cm, reference] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let n_qk = (S * nk * (nt + 1) * ns) as usize;
    let scaled = |n: usize, seed: u64, a: f32| -> Vec<f32> {
        noise(n, seed).iter().map(|x| x * a).collect()
    };
    // q, k about unit length per head (as l2_norm leaves them).
    qp.set_f32(&scaled(n_qk, 1, 0.15));
    kp.set_f32(&scaled(n_qk, 2, 0.15));
    v.set_f32(&noise((S * h * nt * ns) as usize, 3));
    let n_gb = (h * nt * ns) as usize;
    // Log decays in (-1, 0], betas in (0, 1).
    g.set_f32(
        &noise(n_gb, 4)
            .iter()
            .map(|x| -0.5 * (x + 1.0))
            .collect::<Vec<_>>(),
    );
    beta.set_f32(
        &noise(n_gb, 5)
            .iter()
            .map(|x| 0.5 * (x + 1.0))
            .collect::<Vec<_>>(),
    );
    let s0 = scaled((S * S * h * ns) as usize, 6, 0.1);
    st_ggml.set_f32(&s0);
    st_op.set_f32(&s0);
    backend.compute(&graph).unwrap();
    let all = reference.to_f32();
    let n_rows = (S * h * nt * ns) as usize;
    (
        r.to_f32(),
        st_op.to_f32(),
        all[..n_rows].to_vec(),
        all[n_rows..].to_vec(),
    )
}

/// Max |a - b| over max |b|.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale
}

fn cases(h: i64, nk: i64) -> Vec<Case> {
    let mut v = Vec::new();
    for nt in 1..=9 {
        for ns in [1, 3] {
            v.push(Case { h, nk, nt, ns });
        }
    }
    v
}

#[test]
fn cpu_matches_ggml() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for c in cases(4, 2) {
        let (r, s, gr, gs) = run(&mut backend, c);
        let e = rel(&r, &gr);
        assert!(e < 1e-5, "rows {c:?}: {e:e}");
        let e = rel(&s, &gs);
        assert!(e < 1e-5, "states {c:?}: {e:e}");
    }
}

/// [`commit_slots`] over pools of `nt` tokens for `keep.len()` slots, and the reference per slot
/// with its kept tokens alone: [`commit`] (`ggml: false`) or ggml's `gated_delta_net` with its
/// final states (`ggml: true`). Returns (all-slot states, reference states).
fn run_slots(
    backend: &mut Backend,
    h: i64,
    nk: i64,
    nt: i64,
    keep: &[i32],
    ggml: bool,
) -> (Vec<f32>, Vec<f32>) {
    let ns = keep.len() as i64;
    let n_nodes = 16 * keep.len() + 32;
    let ctx = Context::new(n_nodes, n_nodes);
    let t = |ne: &[i64]| ctx.new_tensor(types::F32, ne).as_input();
    let (q, k, v) = (t(&[S, nk, nt, ns]), t(&[S, nk, nt, ns]), t(&[S, h, nt, ns]));
    let (g, beta) = (t(&[1, h, nt, ns]), t(&[1, h, nt, ns]));
    // Outputs too: the ops write them in place, so the allocator must not reuse them.
    let (st_ref, st_op) = (t(&[S, S, h, ns]).as_output(), t(&[S, S, h, ns]).as_output());
    let keep_t = ctx.new_tensor(types::I32, &[ns]).as_input();
    let mut graph = ctx.new_graph(n_nodes);
    graph.expand(commit_slots(&ctx, q, k, v, g, beta, st_op, keep_t).as_output());
    for (s, &kp) in keep.iter().enumerate() {
        if kp == 0 {
            continue;
        }
        let kp = kp as i64;
        let view = |x| slot_view(&ctx, x, kp, s);
        let ns_ = st_ref.nb();
        let st = ctx.view_4d(st_ref, [S, S, h, 1], [ns_[1], ns_[2], ns_[3]], s * ns_[3]);
        let (gs, bs) = (ctx.cont(view(g)), ctx.cont(view(beta)));
        if ggml {
            let out = ctx.gated_delta_net(view(q), view(k), view(v), gs, bs, st, 1);
            let es = out.element_size();
            let fin = ctx.view_4d(
                out,
                st.ne(),
                [
                    S as usize * es,
                    (S * S) as usize * es,
                    (S * S * h) as usize * es,
                ],
                (S * h * kp) as usize * es,
            );
            graph.expand(ctx.cpy(fin, st).as_output());
        } else {
            graph.expand(commit(&ctx, view(q), view(k), view(v), gs, bs, st).as_output());
        }
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let scaled = |n: usize, seed: u64, a: f32| -> Vec<f32> {
        noise(n, seed).iter().map(|x| x * a).collect()
    };
    q.set_f32(&scaled(q.elements(), 1, 0.15));
    k.set_f32(&scaled(k.elements(), 2, 0.15));
    v.set_f32(&noise(v.elements(), 3));
    g.set_f32(
        &noise(g.elements(), 4)
            .iter()
            .map(|x| -0.5 * (x + 1.0))
            .collect::<Vec<_>>(),
    );
    beta.set_f32(
        &noise(beta.elements(), 5)
            .iter()
            .map(|x| 0.5 * (x + 1.0))
            .collect::<Vec<_>>(),
    );
    let s0 = scaled(st_op.elements(), 6, 0.1);
    st_ref.set_f32(&s0);
    st_op.set_f32(&s0);
    keep_t.set_i32(keep);
    backend.compute(&graph).unwrap();
    (st_op.to_f32(), st_ref.to_f32())
}

/// Slot `s`'s first `kp` tokens of a pool tensor `[a, b, tokens, slots]`.
fn slot_view<'a>(ctx: &'a Context, x: Tensor<'a>, kp: i64, s: usize) -> Tensor<'a> {
    let [a, b, _, _] = x.ne();
    let nb = x.nb();
    ctx.view_4d(x, [a, b, kp, 1], [nb[1], nb[2], nb[3]], s * nb[3])
}

/// The keep counts the all-slot commit tests use: idle slots, every unrolled count and the
/// generic loop past it.
const SLOT_KEEPS: [&[i32]; 4] = [
    &[2],
    &[0, 3, 1, 0],
    &[1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    &[0, 0, 5],
];

#[test]
fn cpu_commit_slots_matches_commit_per_slot() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for keep in SLOT_KEEPS {
        let (ours, reference) = run_slots(&mut backend, 4, 2, 9, keep, false);
        assert!(
            ours.iter()
                .zip(&reference)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{keep:?}: rel {:e}",
            rel(&ours, &reference)
        );
    }
}

/// A slot keeping 0 tokens keeps its state as it was.
#[test]
fn cpu_commit_slots_leaves_idle_slots_alone() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(2);
    let (ours, _) = run_slots(&mut backend, 4, 2, 3, &[0, 2, 0], false);
    let per_slot = ours.len() / 3;
    let s0: Vec<f32> = noise(ours.len(), 6).iter().map(|x| x * 0.1).collect();
    assert_eq!(ours[..per_slot], s0[..per_slot], "slot 0");
    assert_ne!(
        ours[per_slot..2 * per_slot],
        s0[per_slot..2 * per_slot],
        "slot 1"
    );
    assert_eq!(ours[2 * per_slot..], s0[2 * per_slot..], "slot 2");
}

/// What [`run_replay`] returns: the reference (commit, then rows) rows and states, the fused
/// verify's rows and states, the states of the deferred commit run on its own, and the packed
/// rows after the fused verify with what they must hold (its tokens' k, v, g and beta over the
/// first `nt2` of each slot, part 39), and its counters.
struct ReplayOut {
    rows: Vec<f32>,
    states: Vec<f32>,
    fused_rows: Vec<f32>,
    fused_states: Vec<f32>,
    flushed: Vec<f32>,
    packed: Vec<f32>,
    stored: Vec<f32>,
    arrive: Vec<i32>,
    /// The states the fused verify must leave: the reference's, but a leaving slot's as it was
    /// (#836).
    left: Vec<f32>,
}

/// A round of a deferred commit (#836 part 14) over `keep.len()` slots: the last verify step's
/// `nt` tokens per slot (`keep[s]` of them kept), then a verify step of `nt2` tokens. The
/// reference commits ([`commit_slots`]) and then reads the rows ([`rows`]); the fused verify
/// ([`rows_replay`]) and the deferred commit alone ([`commit_replay`]) read the kept tokens'
/// inputs packed as a stash leaves them ([`replay_row`]).
fn run_replay(
    backend: &mut Backend,
    h: i64,
    nk: i64,
    nt: i64,
    nt2: i64,
    keep: &[i32],
) -> ReplayOut {
    run_replay_at(backend, h, nk, nt, nt2, keep, 0, false)
}

/// [`run_replay`] with the kept tokens packed from row `start` of each slot (a flush stored
/// them there, #836) and, with `leave`, the count words letting the fused verify leave the
/// states of slots where `keep + nt2` tokens fit the `nt` rows past `start` (it then stores
/// its tokens after the kept ones).
#[allow(clippy::too_many_arguments)]
fn run_replay_at(
    backend: &mut Backend,
    h: i64,
    nk: i64,
    nt: i64,
    nt2: i64,
    keep: &[i32],
    start: i64,
    leave: bool,
) -> ReplayOut {
    run_replay_pad(backend, h, nk, nt, nt2, keep, start, leave, &[])
}

/// [`run_replay_at`] with a verify step's row map (#836, unless `real` is empty): slot `s` has
/// `real[s]` real tokens of the `nt2` (at least 1), the rest padding. The fused verify must give
/// the reference's rows of the real tokens and zeros for the padding ones, and store only the
/// real tokens' inputs (the packed rows past them keep what they held).
#[allow(clippy::too_many_arguments)]
fn run_replay_pad(
    backend: &mut Backend,
    h: i64,
    nk: i64,
    nt: i64,
    nt2: i64,
    keep: &[i32],
    start: i64,
    leave: bool,
    real: &[usize],
) -> ReplayOut {
    let ns = keep.len() as i64;
    assert!(real.is_empty() || real.len() == keep.len());
    let live = |s: usize| real.get(s).map_or(nt2 as usize, |&r| r.min(nt2 as usize));
    let room = nt + start;
    let leaves = |s: usize| leave && i64::from(keep[s]) + nt2 <= nt;
    let data = Context::new(32, 0);
    let t = |ne: &[i64]| data.new_tensor(types::F32, ne);
    let prev = [
        t(&[S, nk, nt, ns]),
        t(&[S, nk, nt, ns]),
        t(&[S, h, nt, ns]),
        t(&[1, h, nt, ns]),
        t(&[1, h, nt, ns]),
    ];
    let cur = [
        t(&[S, nk, nt2, ns]),
        t(&[S, nk, nt2, ns]),
        t(&[S, h, nt2, ns]),
        t(&[1, h, nt2, ns]),
        t(&[1, h, nt2, ns]),
    ];
    let row = replay_row(nk, h);
    let replay = t(&[row, room, ns]);
    let states = [t(&[S, S, h, ns]), t(&[S, S, h, ns]), t(&[S, S, h, ns])];
    let keep_t = data.new_tensor(types::I32, &[ns]);
    let words_t = data.new_tensor(types::I32, &[ns]);
    let map_t = data.new_tensor(types::I32, &[nt2 * ns]);
    // One slot more than the op's, which it must not touch.
    let arrive = data.new_tensor(types::I32, &[nk * (ns + 1)]);
    data.alloc_tensors(backend.buffer_type()).unwrap();
    let scaled = |n: usize, seed: u64, a: f32| -> Vec<f32> {
        noise(n, seed).iter().map(|x| x * a).collect()
    };
    let mut host = Vec::new();
    for (i, set) in [prev, cur].iter().enumerate() {
        let seed = 10 * i as u64;
        let [q, k, v, g, beta] = *set;
        let x = [
            scaled(q.elements(), seed + 1, 0.15),
            scaled(k.elements(), seed + 2, 0.15),
            noise(v.elements(), seed + 3),
            noise(g.elements(), seed + 4)
                .iter()
                .map(|x| -0.5 * (x + 1.0))
                .collect::<Vec<_>>(),
            noise(beta.elements(), seed + 5)
                .iter()
                .map(|x| 0.5 * (x + 1.0))
                .collect::<Vec<_>>(),
        ];
        for (t, x) in set.iter().zip(&x) {
            t.set_f32(x);
        }
        host.push(x);
    }
    // The stash's layout: per slot and token, [q | k | v | g | beta]; tokens past the kept
    // ones hold garbage the ops must not read.
    let widths = [S * nk, S * nk, S * h, h, h].map(|w| w as usize);
    let mut packed = noise((row * room * ns) as usize, 99);
    let (nt_u, room_u, start_u) = (nt as usize, room as usize, start as usize);
    for at in 0..(nt * ns) as usize {
        let (s, t) = (at / nt_u, at % nt_u);
        let mut off = (s * room_u + start_u + t) * row as usize;
        for (x, &w) in host[0].iter().zip(&widths) {
            packed[off..off + w].copy_from_slice(&x[at * w..(at + 1) * w]);
            off += w;
        }
    }
    replay.set_f32(&packed);
    let mut stored = packed.clone();
    for at in 0..(nt2 * ns) as usize {
        let (s, t) = (at / nt2 as usize, at % nt2 as usize);
        if t >= live(s) {
            // A padding token: not stored.
            continue;
        }
        let base = if leaves(s) {
            start_u + keep[s] as usize
        } else {
            0
        };
        let mut off = (s * room_u + base + t) * row as usize;
        for (i, (x, &w)) in host[1].iter().zip(&widths).enumerate() {
            if i > 0 {
                stored[off..off + w].copy_from_slice(&x[at * w..(at + 1) * w]);
            }
            off += w;
        }
    }
    arrive.set_i32(&vec![0; arrive.elements()]);
    let s0 = scaled(states[0].elements(), 6, 0.1);
    for s in states {
        s.set_f32(&s0);
    }
    keep_t.set_i32(keep);
    let words: Vec<i32> = keep
        .iter()
        .map(|&k| count_word(k as usize, start_u, leave))
        .collect();
    words_t.set_i32(&words);
    // The row map as `pad_rows::map` makes it: a padding token takes its sequence's first row.
    let map: Vec<i32> = (0..(nt2 * ns) as usize)
        .map(|at| {
            let (s, t) = (at / nt2 as usize, at % nt2 as usize);
            (s * nt2 as usize + if t < live(s) { t } else { 0 }) as i32
        })
        .collect();
    map_t.set_i32(&map);
    let pad = (!real.is_empty()).then_some(map_t);
    let [st_ref, st_fused, st_flush] = states;
    // The reference commit first, on its own graph.
    let ctx = Context::new(32, 32);
    let mut graph = ctx.new_graph(32);
    let [q, k, v, g, beta] = prev;
    graph.expand(commit_slots(&ctx, q, k, v, g, beta, st_ref, keep_t).as_output());
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    backend.compute(&graph).unwrap();
    let ctx = Context::new(32, 32);
    let mut graph = ctx.new_graph(32);
    let [q, k, v, g, beta] = cur;
    let r = rows(&ctx, q, k, v, g, beta, st_ref).as_output();
    // The deferred commit alone first: the fused verify then stores its own tokens there.
    let flush = commit_replay(&ctx, replay, st_flush, words_t, nk).as_output();
    let fused = rows_replay(
        &ctx,
        q,
        k,
        v,
        g,
        beta,
        st_fused,
        words_t,
        replay,
        (arrive, 1),
        pad,
    )
    .as_output();
    for t in [r, flush, fused] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    backend.compute(&graph).unwrap();
    let states = st_ref.to_f32();
    let per = (S * S * h) as usize;
    let mut left = states.clone();
    for s in (0..ns as usize).filter(|&s| leaves(s)) {
        left[s * per..(s + 1) * per].copy_from_slice(&s0[s * per..(s + 1) * per]);
    }
    // A padding token's row: zeros.
    let mut rows = r.to_f32();
    let width = (S * h) as usize;
    for (at, row) in rows.chunks_mut(width).enumerate() {
        if at % nt2 as usize >= live(at / nt2 as usize) {
            row.fill(0.0);
        }
    }
    ReplayOut {
        rows,
        states,
        fused_rows: fused.to_f32(),
        fused_states: st_fused.to_f32(),
        flushed: st_flush.to_f32(),
        packed: replay.to_f32(),
        stored,
        arrive: arrive.to_i32(),
        left,
    }
}

/// The fused verify is the commit and then the rows, bit for bit, and so is the deferred commit
/// run on its own; slots keeping 0 tokens start from their state as it was.
#[test]
fn cpu_rows_replay_is_commit_then_rows() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for keep in SLOT_KEEPS {
        for nt2 in [1, 4] {
            let o = run_replay(&mut backend, 4, 2, 9, nt2, keep);
            let same = |a: &[f32], b: &[f32]| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            };
            assert!(same(&o.fused_rows, &o.rows), "{keep:?} x {nt2}: rows");
            assert!(same(&o.fused_states, &o.states), "{keep:?} x {nt2}: states");
            assert!(same(&o.flushed, &o.states), "{keep:?} x {nt2}: flush");
            assert!(same(&o.packed, &o.stored), "{keep:?} x {nt2}: store");
            assert!(
                o.arrive.iter().all(|&c| c == 0),
                "{keep:?} x {nt2}: counters"
            );
        }
    }
}

/// Count words that let the fused verify leave the states (#836): its rows are still the commit
/// and then the rows, a leaving slot's state is as it was, its tokens go after the kept ones,
/// and the deferred commit on its own (a flush) reads the kept tokens from their start row.
#[test]
fn cpu_rows_replay_leaves_the_states_it_may() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for keep in SLOT_KEEPS {
        for (nt2, start) in [(1, 0), (4, 0), (4, 3), (9, 2)] {
            let o = run_replay_at(&mut backend, 4, 2, 9, nt2, keep, start, true);
            let same = |a: &[f32], b: &[f32]| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            };
            let at = format!("{keep:?} x {nt2} from {start}");
            assert!(same(&o.fused_rows, &o.rows), "{at}: rows");
            assert!(same(&o.fused_states, &o.left), "{at}: states");
            assert!(same(&o.flushed, &o.states), "{at}: flush");
            assert!(same(&o.packed, &o.stored), "{at}: store");
            assert!(o.arrive.iter().all(|&c| c == 0), "{at}: counters");
        }
    }
}

/// Real tokens per slot for the padding tests: 1 to `nt2`, varied over the slots.
fn pad_reals(n_slots: usize, nt2: i64) -> Vec<usize> {
    (0..n_slots).map(|s| 1 + (s * 3) % nt2 as usize).collect()
}

/// The verify step's row map (#836): a slot's real tokens' rows and stores are the reference's
/// bit for bit, its padding tokens' rows are zeros and their inputs are not stored; the commit,
/// the left states and the counters are as without the map.
#[test]
fn cpu_rows_replay_skips_padding_tokens() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for keep in SLOT_KEEPS {
        for (nt2, start, leave) in [(4, 0, false), (9, 0, false), (4, 3, true), (9, 2, true)] {
            let real = pad_reals(keep.len(), nt2);
            let o = run_replay_pad(&mut backend, 4, 2, 9, nt2, keep, start, leave, &real);
            let same = |a: &[f32], b: &[f32]| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            };
            let at = format!("{keep:?} x {nt2} from {start}, real {real:?}");
            let states = if leave { &o.left } else { &o.states };
            assert!(same(&o.fused_rows, &o.rows), "{at}: rows");
            assert!(same(&o.fused_states, states), "{at}: states");
            assert!(same(&o.flushed, &o.states), "{at}: flush");
            assert!(same(&o.packed, &o.stored), "{at}: store");
            assert!(o.arrive.iter().all(|&c| c == 0), "{at}: counters");
        }
    }
}

/// [`decode`]'s output (rows, then the final states) and ggml's over the same inputs on
/// `backend`, `ns` sequences of one token.
fn run_decode(backend: &mut Backend, h: i64, nk: i64, ns: i64) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new(32, 32);
    let t = |ne: &[i64]| ctx.new_tensor(types::F32, ne).as_input();
    let (q, k, v) = (t(&[S, nk, 1, ns]), t(&[S, nk, 1, ns]), t(&[S, h, 1, ns]));
    let (g, beta) = (t(&[1, h, 1, ns]), t(&[1, h, 1, ns]));
    let (st_ggml, st_op) = (t(&[S, S, h, ns]), t(&[S, S, h, ns]));
    let ours = decode(&ctx, q, k, v, g, beta, st_op).as_output();
    let reference = ctx
        .gated_delta_net(q, k, v, g, beta, st_ggml, 1)
        .as_output();
    assert_eq!(ours.ne(), reference.ne());
    let mut graph = ctx.new_graph(32);
    graph.expand(ours);
    graph.expand(reference);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let scaled = |n: usize, seed: u64, a: f32| -> Vec<f32> {
        noise(n, seed).iter().map(|x| x * a).collect()
    };
    q.set_f32(&scaled(q.elements(), 1, 0.15));
    k.set_f32(&scaled(k.elements(), 2, 0.15));
    v.set_f32(&noise(v.elements(), 3));
    g.set_f32(&scaled(g.elements(), 4, 0.5));
    beta.set_f32(&scaled(beta.elements(), 5, 0.5));
    let s0 = scaled(st_op.elements(), 6, 0.1);
    st_ggml.set_f32(&s0);
    st_op.set_f32(&s0);
    backend.compute(&graph).unwrap();
    // The input states are only read.
    assert_eq!(st_op.to_f32(), s0);
    (ours.to_f32(), reference.to_f32())
}

#[test]
fn cpu_decode_matches_ggml() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for ns in [1, 3] {
        let (ours, reference) = run_decode(&mut backend, 4, 2, ns);
        let e = rel(&ours, &reference);
        assert!(e < 1e-5, "{ns} sequences: {e:e}");
    }
}

#[test]
fn decode_needs_a_gpu_it_is_chosen_for_and_one_token() {
    let ctx = Context::new(32, 0);
    let t = |ne: &[i64]| ctx.new_tensor(types::F32, ne);
    let (q, k, v) = (t(&[S, 2, 1, 3]), t(&[S, 2, 1, 3]), t(&[S, 4, 1, 3]));
    let (g, beta) = (t(&[1, 4, 1, 3]), t(&[1, 4, 1, 3]));
    let state = t(&[S, S, 4, 3]);
    let (q2, k2, v2) = (t(&[S, 2, 2, 3]), t(&[S, 2, 2, 3]), t(&[S, 4, 2, 3]));
    let (g2, beta2) = (t(&[1, 4, 2, 3]), t(&[1, 4, 2, 3]));
    for cc in [None, Some(7), Some(8), Some(9), Some(12)] {
        set_gpu(cc);
        let gpu = cc.is_some_and(|m| m >= 9) && cfg!(feature = "cuda");
        assert_eq!(decode_fits(q, k, v, g, beta, state), gpu, "{cc:?}");
        assert!(!decode_fits(q2, k2, v2, g2, beta2, state), "two tokens");
        set_gdn_decode(false);
        assert!(!decode_fits(q, k, v, g, beta, state), "switched off");
        set_gdn_decode(true);
    }
    set_gpu(None);
}

#[test]
fn layouts_the_kernel_does_not_take_do_not_fit() {
    let ctx = Context::new(32, 0);
    let t = |ne: &[i64]| ctx.new_tensor(types::F32, ne);
    let (q, k) = (t(&[S, 2, 3, 1]), t(&[S, 2, 3, 1]));
    let v = t(&[S, 4, 3, 1]);
    let (g, beta) = (t(&[1, 4, 3, 1]), t(&[1, 4, 3, 1]));
    let state = t(&[S, S, 4, 1]);
    assert!(shapes_fit(q, k, v, g, beta, state));
    // Heads of 64, a KDA gate, a state of the wrong size, 64 tokens (ggml's chunked paths).
    let (q64, k64, v64) = (t(&[64, 2, 3, 1]), t(&[64, 2, 3, 1]), t(&[64, 4, 3, 1]));
    assert!(!shapes_fit(q64, k64, v64, g, beta, t(&[64, 64, 4, 1])));
    let kda = t(&[S, 4, 3, 1]);
    assert!(!shapes_fit(q, k, v, kda, kda, state));
    assert!(!shapes_fit(q, k, v, g, beta, t(&[S, S, 2, 1])));
    let (q, k) = (t(&[S, 2, 64, 1]), t(&[S, 2, 64, 1]));
    let v = t(&[S, 4, 64, 1]);
    let (g, beta) = (t(&[1, 4, 64, 1]), t(&[1, 4, 64, 1]));
    assert!(!shapes_fit(q, k, v, g, beta, state));
}

#[cfg(feature = "cuda")]
fn gpu() -> Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

#[cfg(feature = "cuda")]
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Per token count 1..=9 (the exact-T kernels and the generic one), one and three sequences,
/// every CTA shape: rows and states bit for bit ggml-cuda's.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_is_bit_identical_to_ggml_per_token_count() {
    let mut gpu = gpu();
    for warps in [0, 1, 2, 4] {
        force_warps(warps);
        // qwen3.8-flash-next's heads (32 v over 16 q / k), and a small case.
        for c in cases(32, 16).into_iter().chain(cases(4, 2)) {
            let (r, s, gr, gs) = run(&mut gpu, c);
            assert!(
                same_bits(&r, &gr),
                "rows {c:?} warps {warps}: rel {:e}",
                rel(&r, &gr)
            );
            assert!(
                same_bits(&s, &gs),
                "states {c:?} warps {warps}: rel {:e}",
                rel(&s, &gs)
            );
        }
    }
    force_warps(0);
}

/// [`commit_slots`]: every slot's final state bit for bit ggml-cuda's over its kept tokens alone.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_commit_slots_is_bit_identical_to_ggml_per_slot() {
    let mut gpu = gpu();
    for (h, nk) in [(48, 16), (4, 2)] {
        for keep in SLOT_KEEPS {
            let (ours, reference) = run_slots(&mut gpu, h, nk, 9, keep, true);
            assert!(
                same_bits(&ours, &reference),
                "{h} heads, keep {keep:?}: rel {:e}",
                rel(&ours, &reference)
            );
        }
    }
}

/// [`rows_replay`] on the GPU, every CTA shape: the commit and then the rows bit for bit, its
/// tokens' inputs stored over the packed rows by the CTAs (the last of a q / k head's storing
/// what they all read, part 39), and the counters back at 0.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rows_replay_is_commit_then_rows_and_stores() {
    let mut gpu = gpu();
    for warps in [0, 1, 2, 4] {
        force_warps(warps);
        for (h, nk) in [(48, 16), (4, 2)] {
            for keep in SLOT_KEEPS {
                for nt2 in [1, 4, 9] {
                    let o = run_replay(&mut gpu, h, nk, 9, nt2, keep);
                    let at = format!("warps {warps}, {h} heads, keep {keep:?} x {nt2}");
                    assert!(same_bits(&o.fused_rows, &o.rows), "{at}: rows");
                    assert!(same_bits(&o.fused_states, &o.states), "{at}: states");
                    assert!(same_bits(&o.flushed, &o.states), "{at}: flush");
                    assert!(same_bits(&o.packed, &o.stored), "{at}: store");
                    assert!(o.arrive.iter().all(|&c| c == 0), "{at}: counters");
                }
            }
        }
    }
    force_warps(0);
}

/// [`rows_replay`]'s kernel variants on the GPU (inputs through shared memory, #836): the
/// same checks as the per-lane kernel's above.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rows_replay_staged_variants_are_commit_then_rows_and_stores() {
    let mut gpu = gpu();
    for (variant, name) in bench::REPLAY_VARIANTS {
        set_replay_variant(variant);
        for (h, nk) in [(48, 16), (4, 2)] {
            for keep in SLOT_KEEPS {
                for nt2 in [1, 4, 9] {
                    let o = run_replay(&mut gpu, h, nk, 9, nt2, keep);
                    let at = format!("{name}, {h} heads, keep {keep:?} x {nt2}");
                    assert!(same_bits(&o.fused_rows, &o.rows), "{at}: rows");
                    assert!(same_bits(&o.fused_states, &o.states), "{at}: states");
                    assert!(same_bits(&o.packed, &o.stored), "{at}: store");
                    assert!(o.arrive.iter().all(|&c| c == 0), "{at}: counters");
                }
            }
        }
    }
    set_replay_variant(-1);
}

/// [`rows_replay`] on the GPU with count words that let it leave the states (#836), every
/// kernel: the checks of [`cpu_rows_replay_leaves_the_states_it_may`], bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rows_replay_leaves_the_states_it_may() {
    let mut gpu = gpu();
    let mut kernels: Vec<(i32, i32, &str)> = [0, 1, 2, 4].map(|w| (w, -1, "per lane")).to_vec();
    kernels.extend(bench::REPLAY_VARIANTS.iter().map(|&(v, name)| (0, v, name)));
    for (warps, variant, name) in kernels {
        force_warps(warps);
        set_replay_variant(variant);
        for (h, nk) in [(48, 16), (4, 2)] {
            for keep in SLOT_KEEPS {
                for (nt2, start) in [(1, 0), (4, 0), (4, 3), (9, 2)] {
                    let o = run_replay_at(&mut gpu, h, nk, 9, nt2, keep, start, true);
                    let at = format!("{name} {warps}w, {h}h, {keep:?} x {nt2} @{start}");
                    assert!(same_bits(&o.fused_rows, &o.rows), "{at}: rows");
                    assert!(same_bits(&o.fused_states, &o.left), "{at}: states");
                    assert!(same_bits(&o.flushed, &o.states), "{at}: flush");
                    assert!(same_bits(&o.packed, &o.stored), "{at}: store");
                    assert!(o.arrive.iter().all(|&c| c == 0), "{at}: counters");
                }
            }
        }
    }
    force_warps(0);
    set_replay_variant(-1);
}

/// [`rows_replay`] on the GPU with a verify step's row map (#836), every kernel: the checks of
/// [`cpu_rows_replay_skips_padding_tokens`], bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rows_replay_skips_padding_tokens() {
    let mut gpu = gpu();
    let mut kernels: Vec<(i32, i32, &str)> = [0, 1, 2, 4].map(|w| (w, -1, "per lane")).to_vec();
    kernels.extend(bench::REPLAY_VARIANTS.iter().map(|&(v, name)| (0, v, name)));
    for (warps, variant, name) in kernels {
        force_warps(warps);
        set_replay_variant(variant);
        for (h, nk) in [(48, 16), (4, 2)] {
            for keep in SLOT_KEEPS {
                for (nt2, start, leave) in
                    [(4, 0, false), (9, 0, false), (4, 3, true), (9, 2, true)]
                {
                    let real = pad_reals(keep.len(), nt2);
                    let o = run_replay_pad(&mut gpu, h, nk, 9, nt2, keep, start, leave, &real);
                    let at = format!("{name} {warps}w, {h}h, {keep:?} x {nt2} @{start} {real:?}");
                    let states = if leave { &o.left } else { &o.states };
                    assert!(same_bits(&o.fused_rows, &o.rows), "{at}: rows");
                    assert!(same_bits(&o.fused_states, states), "{at}: states");
                    assert!(same_bits(&o.flushed, &o.states), "{at}: flush");
                    assert!(same_bits(&o.packed, &o.stored), "{at}: store");
                    assert!(o.arrive.iter().all(|&c| c == 0), "{at}: counters");
                }
            }
        }
    }
    force_warps(0);
    set_replay_variant(-1);
}

/// [`decode`]: every kernel variant, rows and final states bit for bit ggml-cuda's.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_decode_is_bit_identical_to_ggml() {
    let mut gpu = gpu();
    for (variant, name) in bench::VARIANTS {
        set_decode_variant(variant);
        for (h, nk, ns) in [(48, 16, 1), (48, 16, 5), (32, 16, 16), (4, 2, 3)] {
            let (ours, reference) = run_decode(&mut gpu, h, nk, ns);
            assert!(
                same_bits(&ours, &reference),
                "{name}: {h} heads x {ns} sequences: rel {:e}",
                rel(&ours, &reference)
            );
        }
    }
    set_decode_variant(-1);
}

/// Kernel time per layer of [`rows`] (a verify step) and [`commit`] against ggml's op (with the
/// state copy a commit adds), over `LAYERS` layers' states in one graph:
/// `cargo test --release --features cuda gdn::tests::cuda_bench -- --ignored --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a benchmark"]
fn cuda_bench() {
    const LAYERS: usize = 36;
    const REPS: usize = 50;
    let mut gpu = gpu();
    let (h, nk) = (32i64, 16i64);
    for (what, nt, ns) in [
        ("verify", 3, 1),
        ("verify", 3, 4),
        ("verify", 3, 16),
        ("commit", 1, 1),
        ("commit", 2, 1),
        ("commit", 3, 1),
        ("commit", 3, 4),
        ("commit", 2, 16),
    ] {
        // Inputs are outputs too: the allocator would hand their memory to later nodes.
        let time = |gpu: &mut Backend, which: &str, warps: i32| -> f64 {
            force_warps(warps);
            let ctx = Context::new(16 * LAYERS + 16, 16 * LAYERS + 16);
            let q = ctx
                .new_tensor(types::F32, &[S, nk, nt, ns])
                .as_input()
                .as_output();
            let k = ctx
                .new_tensor(types::F32, &[S, nk, nt, ns])
                .as_input()
                .as_output();
            let v = ctx
                .new_tensor(types::F32, &[S, h, nt, ns])
                .as_input()
                .as_output();
            let g = ctx
                .new_tensor(types::F32, &[1, h, nt, ns])
                .as_input()
                .as_output();
            let beta = ctx
                .new_tensor(types::F32, &[1, h, nt, ns])
                .as_input()
                .as_output();
            let mut graph = ctx.new_graph(16 * LAYERS + 16);
            let mut states = Vec::new();
            for _ in 0..LAYERS {
                let st = ctx
                    .new_tensor(types::F32, &[S, S, h, ns])
                    .as_input()
                    .as_output();
                states.push(st);
                let out = match (which, what) {
                    ("op", "verify") => rows(&ctx, q, k, v, g, beta, st),
                    ("op", _) => commit(&ctx, q, k, v, g, beta, st),
                    (_, "verify") => ctx.gated_delta_net(q, k, v, g, beta, st, 1),
                    _ => {
                        let out = ctx.gated_delta_net(q, k, v, g, beta, st, 1);
                        let es = out.element_size();
                        let fin = ctx.view_4d(
                            out,
                            st.ne(),
                            [
                                S as usize * es,
                                (S * S) as usize * es,
                                (S * S * h) as usize * es,
                            ],
                            (S * h * nt * ns) as usize * es,
                        );
                        ctx.cpy(fin, st)
                    }
                };
                graph.expand(out);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            q.set_f32(&noise((S * nk * nt * ns) as usize, 1));
            k.set_f32(&noise((S * nk * nt * ns) as usize, 2));
            v.set_f32(&noise((S * h * nt * ns) as usize, 3));
            g.set_f32(&vec![-0.1; (h * nt * ns) as usize]);
            beta.set_f32(&vec![0.5; (h * nt * ns) as usize]);
            for st in &states {
                st.set_f32(&vec![0.0; (S * S * h * ns) as usize]);
            }
            gpu.compute(&graph).unwrap();
            gpu.compute(&graph).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..REPS {
                gpu.compute(&graph).unwrap();
            }
            t0.elapsed().as_secs_f64() * 1e6 / (REPS * LAYERS) as f64
        };
        let ggml = time(&mut gpu, "ggml", 0);
        let ops: Vec<String> = [0, 1, 2, 4]
            .iter()
            .map(|&w| format!("w{w} {:.2}", time(&mut gpu, "op", w)))
            .collect();
        println!(
            "gdn {what} T={nt} seqs={ns}: ggml {ggml:.2} us/layer, op {} us/layer",
            ops.join(", ")
        );
    }
    force_warps(0);
}

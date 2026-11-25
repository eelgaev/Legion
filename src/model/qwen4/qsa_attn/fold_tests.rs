//! [`fold_q`] (#1183): the gated prompt attention making its query's head norm and RoPE itself
//! (`node_fuse`'s rewrite) against ggml's nodes of them, codes and scales bit for bit (CUDA,
//! ignored tests), and where the rewrite leaves the nodes.

use super::tests::{f16_bytes, noise};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::ROPE_IMROPE;
use crate::model::core::node_fuse::rewrite_graph;

const EPS: f32 = 1e-6;

/// A prompt step: the projection's rows of `total` tokens, the attention of `n` of them from
/// token `off`, RoPE positions from `base`.
struct Step {
    d: i64,
    n_head: i64,
    n_head_kv: i64,
    total: i64,
    off: i64,
    n: i64,
    base: i32,
    imrope: bool,
}

/// Per token: the positions it attends to, ascending (itself always).
fn picks(tokens: usize) -> Vec<Vec<i32>> {
    (0..tokens)
        .map(|t| {
            (0..=t)
                .filter(|&p| p == t || (p * 5 + t) % 4 != 0)
                .map(|p| p as i32)
                .collect()
        })
        .collect()
}

/// The packed activations of [`gated_codes`] over qwen4exp's query chain (`rms_norm`, `mul`,
/// `rope_multi`, a view of the step's tokens) on `backend`, after `node_fuse`'s rewrite (as on
/// CUDA). `keep`: the RoPE's output is a graph output, so the rewrite must leave the chain.
/// Also whether the node's query was folded and the run's [`Fold`].
fn run(s: &Step, backend: &mut Backend, keep: bool) -> (Vec<u8>, bool, Fold) {
    let (d, nh, nkv, n) = (s.d, s.n_head, s.n_head_kv, s.n);
    let k = d * nh;
    let blocks = (n as usize).div_ceil(KV_BLOCK);
    let cells = (blocks * KV_BLOCK) as i64;
    let ctx = Context::new(64, 64);
    let input = |ty, ne: &[i64]| ctx.new_tensor(ty, ne).as_input();
    // Kept for the whole graph, as weights and the positions (read by every layer) are.
    let kept = |ty, ne: &[i64]| ctx.new_tensor(ty, ne).as_input().as_output();
    // The joined projection's rows: [q_h | gate_h] per head, as attention_res views them.
    let qg = kept(types::F32, &[2 * k, s.total]);
    let nb = [(2 * d) as usize * 4, qg.nb()[1]];
    let raw = ctx.view_3d(qg, [d, nh, s.total], nb, 0);
    let w = kept(types::F32, &[d]);
    let pos = kept(types::I32, &[4 * s.total]);
    let mode = if s.imrope { ROPE_IMROPE } else { 8 };
    let rope = ctx.rope_multi(
        ctx.mul(ctx.rms_norm(raw, EPS), w),
        pos,
        64,
        [11, 11, 10, 0],
        mode,
        262_144,
        10_000_000.0,
        1.0,
        0.0,
        1.0,
        32.0,
        1.0,
    );
    let rope = if keep { rope.as_output() } else { rope };
    // The step's tokens, as `attend` views a group's.
    let rnb = rope.nb();
    let q = ctx.view_3d(rope, [d, nh, n], [rnb[1], rnb[2]], s.off as usize * rnb[2]);
    let q = ctx.reshape_4d(q, [d, nh, n, 1]);
    let gate = ctx.view_3d(qg, [d, nh, n], nb, s.off as usize * nb[1] + d as usize * 4);
    let kp = input(types::F16, &[d * nkv, cells]);
    let vp = input(types::F16, &[d * nkv, cells]);
    let table = input(types::I32, &[blocks as i64, 1]);
    let sel = input(types::I32, &[1 + n, n]);
    let att = attention(&ctx, q, kp, vp, table, sel);
    let o = ctx.reshape_2d(att, k, n);
    let got = gated_codes(&ctx, o, gate).expect("a prompt step's attention output");
    // SAFETY: a view of the op's node.
    let node = ctx.wrap(unsafe { (*got.raw()).view_src }).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(node);
    if keep {
        graph.expand(rope);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // SAFETY: an allocated graph that has not run.
    unsafe { rewrite_graph(graph.raw(), true) };
    // SAFETY: the op's node, allocated.
    let (folded, off) = unsafe { (is_q_folded(node.raw()), fold_q_off(node.raw())) };
    let fold = Fold {
        off,
        cc12: (kp.cuda_device().and_then(crate::hw::per_device::cc)).is_some_and(|c| c.0 == 12),
    };

    let rows: Vec<f32> = noise((2 * k * s.total) as usize, 21)
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if (i as i64 / d) % 2 == 0 {
                3.0 * x
            } else {
                4.0 * x
            }
        })
        .collect();
    qg.set_f32(&rows);
    let wv: Vec<f32> = noise(d as usize, 22)
        .iter()
        .map(|x| 1.0 + 0.5 * x)
        .collect();
    w.set_f32(&wv);
    let pv: Vec<i32> = (0..4)
        .flat_map(|sec| (0..s.total as i32).map(move |t| s.base + t + 3 * sec))
        .collect();
    pos.set_i32(&pv);
    kp.set_bytes(&f16_bytes(&noise(kp.elements(), 23)));
    vp.set_bytes(&f16_bytes(&noise(vp.elements(), 24)));
    let tab: Vec<i32> = (0..blocks as i32).rev().collect();
    table.set_i32(&tab);
    let mut srows = vec![0i32; ((1 + n) * n) as usize];
    for (t, p) in picks(n as usize).iter().enumerate() {
        let row = &mut srows[t * (1 + n as usize)..];
        row[0] = p.len() as i32;
        row[1..=p.len()].copy_from_slice(p);
    }
    sel.set_i32(&srows);
    backend.compute(&graph).unwrap();
    // `got` is not in the graph (so has no memory of its own): its bytes are the node's first.
    (node.bytes_prefix(got.nbytes()), folded, fold)
}

/// What decides a run's fold besides the graph's chain.
struct Fold {
    /// [`fold_q_off`] of the node.
    off: Option<&'static str>,
    /// The graph's device (its KV cache's) is of compute capability 12.x.
    cc12: bool,
}

impl Fold {
    /// A debug override (`LEGION_DEBUG_QSA_PROMPT_KERNEL`, `LEGION_DEBUG_QSA_GW_L1=0`) keeps the
    /// fold off on any device.
    fn overridden(&self) -> bool {
        self.off
            .is_some_and(|w| w.contains("LEGION_DEBUG_QSA_PROMPT_KERNEL") || w.contains("GW_L1"))
    }
}

/// Where two runs' packed activations first differ.
fn first_diff(a: &[u8], b: &[u8], k: usize, n: usize) -> Option<String> {
    let i = a.iter().zip(b).position(|(x, y)| x != y)?;
    Some(if i < k * n {
        format!("code of token {}, column {}", i / k, i % k)
    } else {
        format!("scale of token {}", (i - k * n) / 4)
    })
}

/// Off the GPU the rewrite leaves the chain (the fold is a GB10 kernel's), and the codes are
/// the same.
#[test]
fn cpu_fold_q_stays_off() {
    let mut backend = super::tests::cpu();
    let s = Step {
        d: 64,
        n_head: 8,
        n_head_kv: 2,
        total: 80,
        off: 0,
        n: 80,
        base: 0,
        imrope: true,
    };
    let (want, kept, _) = run(&s, &mut backend, true);
    let (got, folded, _) = run(&s, &mut backend, false);
    assert!(!kept && !folded);
    assert_eq!(
        first_diff(&want, &got, (s.d * s.n_head) as usize, s.n as usize),
        None
    );
}

/// On a GPU, the attention with its query folded (on a compute capability 12.x GPU) against
/// ggml's norm and RoPE nodes feeding it: qwen4exp's shape (head size 256, 24 heads over 2 kv
/// heads), a token count that is not a multiple of anything, the step at token 0 and later in
/// the rows, positions from 0 and from 199,999, IMROPE and MROPE. Elsewhere nothing is folded.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_fold_q_matches_ggml_norm_rope() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (off, total, base, imrope) in [
        (0, 777, 0, true),
        (5, 790, 199_999, true),
        (3, 781, 77, false),
    ] {
        let s = Step {
            d: 256,
            n_head: 24,
            n_head_kv: 2,
            total,
            off,
            n: 777,
            base,
            imrope,
        };
        let (want, kept, _) = run(&s, &mut backend, true);
        let (got, folded, fold) = run(&s, &mut backend, false);
        assert!(!kept, "a RoPE output with another reader was folded");
        if fold.overridden() {
            assert!(!folded, "folded under a debug override");
            eprintln!("skipped the folded check: {}", fold.off.unwrap_or_default());
        } else {
            assert_eq!(folded, fold.cc12, "folded on 12.x only");
        }
        let (k, n) = ((s.d * s.n_head) as usize, s.n as usize);
        assert_eq!(
            first_diff(&want, &got, k, n),
            None,
            "token {off} of {total}, positions from {base}, imrope {imrope}"
        );
    }
}

//! `--tp-moe-ep`'s tables, and a CPU oracle: the ranks' partial MoE sums over their halves of
//! the experts add up to the MoE over all of them, on the fused op's path (table) and on the
//! `mul_mat_id` path (remapped ids, masked weights) and with expert-parallel placement
//! (`--tp-moe-ep-place`, #803) on top: a placed layer's ranks hold scattered experts.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::moe_ffn::{ExpertRefs, PairOut, moe_ffn};
use crate::tp_net::ep_place::{self, Placement};

#[test]
fn tables_map_the_share() {
    let (slot, ids, keep) = tables(8, 4, 4);
    assert_eq!(slot, [-1, -1, -1, -1, 0, 1, 2, 3]);
    assert_eq!(ids, [0, 0, 0, 0, 0, 1, 2, 3]);
    assert_eq!(keep, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let (slot, ids, keep) = tables(6, 0, 3);
    assert_eq!(slot, [0, 1, 2, -1, -1, -1]);
    assert_eq!(ids, [0, 1, 2, 0, 0, 0]);
    assert_eq!(keep, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
}

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

const N_EMBD: usize = 256;
const N_FF: usize = 128;
const N_EXPERT: usize = 8;
const N_USED: usize = 3;
const N_TOK: usize = 5;

/// One expert stack's quantized bytes and type: `[k, m, N_EXPERT]`.
struct Stack {
    ty: ffi::ggml_type,
    k: usize,
    m: usize,
    q: Vec<u8>,
}

impl Stack {
    fn new(ty: ffi::ggml_type, k: usize, m: usize, seed: u64) -> Self {
        let vals: Vec<f32> = noise(k * m * N_EXPERT, seed)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let q = crate::ggml::quantize(ty, &vals, k);
        Self { ty, k, m, q }
    }

    /// The bytes of experts `which`, in that order.
    fn share(&self, which: &[u64]) -> Vec<u8> {
        let per = self.q.len() / N_EXPERT;
        which
            .iter()
            .flat_map(|&e| &self.q[e as usize * per..][..per])
            .copied()
            .collect()
    }
}

struct Experts {
    up: Stack,
    gate: Stack,
    down: Stack,
    x: Vec<f32>,
    ids: Vec<i32>,
    w: Vec<f32>,
}

fn experts() -> Experts {
    Experts {
        up: Stack::new(types::Q4_K, N_EMBD, N_FF, 1),
        gate: Stack::new(types::Q4_K, N_EMBD, N_FF, 2),
        down: Stack::new(types::Q5_1, N_FF, N_EMBD, 3),
        x: noise(N_EMBD * N_TOK, 4),
        // Distinct experts per token, from both halves.
        ids: (0..N_TOK * N_USED)
            .map(|i| ((i / N_USED * 3 + i % N_USED * 5) % N_EXPERT) as i32)
            .collect(),
        w: noise(N_USED * N_TOK, 5)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect(),
    }
}

/// The experts `which` as weights in `wctx` (slot `i` holds `which[i]`), allocated on the CPU.
fn load<'a>(wctx: &'a Context, e: &Experts, which: &[u64]) -> [Tensor<'a>; 3] {
    let n = which.len();
    let t = |s: &Stack| wctx.new_tensor(s.ty, &[s.k as i64, s.m as i64, n as i64]);
    let w = [t(&e.up), t(&e.gate), t(&e.down)];
    wctx.alloc_tensors(Backend::cpu().unwrap().buffer_type())
        .unwrap();
    for (t, s) in w.iter().zip([&e.up, &e.gate, &e.down]) {
        t.set_bytes(&s.share(which));
    }
    w
}

/// The layer the placed tests place.
const IL: usize = 2;

/// The MoE of `e` as layer [`IL`] over all eight experts, or (`ep`: rank `r` of 2) over the
/// rank's share: its half of the contiguous split, or the experts `place` gives it. On the
/// fused op or (`fused` false) on `mul_mat_id` with the weighted sum on the host.
fn partial(e: &Experts, ep: Option<usize>, place: Option<&Placement>, fused: bool) -> Vec<f32> {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(2);
    let owner = place.map_or_else(|| ep_place::contiguous(N_EXPERT, 2), |p| p.owner(IL));
    let which: Vec<u64> = match ep {
        Some(r) => ep_place::experts_of(&owner, r),
        None => (0..N_EXPERT as u64).collect(),
    };
    let wctx = Context::new(4, 0);
    let [up, gate, down] = load(&wctx, e, &which);
    let share = ep.map(|r| TpEp::new(up, N_EXPERT, r, 2, place).unwrap());
    let ctx = Context::new(64, 64);
    let (nt, nu) = (N_TOK as i64, N_USED as i64);
    let xt = ctx.new_tensor(types::F32, &[N_EMBD as i64, nt]).as_input();
    let idt = ctx.new_tensor(types::I32, &[nu, nt]).as_input();
    let wt = ctx.new_tensor(types::F32, &[nu, nt]).as_input();
    let mut graph = ctx.new_graph(64);
    let (out, masked) = if fused {
        let refs = ExpertRefs {
            up,
            gate,
            down,
            table: share.as_ref().map(|s| s.table(IL)),
        };
        let op = moe_ffn(&ctx, xt, idt, wt, refs, PairOut::F32).as_output();
        graph.expand(op);
        (op, None)
    } else {
        let (ids, w) = match &share {
            Some(s) => s.remap(&ctx, IL, idt, wt, nt),
            None => (idt, wt),
        };
        let x3 = ctx.reshape_3d(xt, N_EMBD as i64, 1, nt);
        let u = ctx.mul_mat_id(up, x3, ids);
        let g = ctx.mul_mat_id(gate, x3, ids);
        let act = ctx.mul(ctx.silu(g), u);
        let d = ctx.mul_mat_id(down, act, ids).as_output();
        let w = w.as_output();
        graph.expand(d);
        graph.expand(w);
        (d, Some(w))
    };
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    xt.set_f32(&e.x);
    idt.set_i32(&e.ids);
    wt.set_f32(&e.w);
    backend.compute(&graph).unwrap();
    let out = out.to_f32();
    let Some(masked) = masked else {
        return out;
    };
    // [N_EMBD, N_USED, N_TOK] per-pair outputs, weighted and summed per token.
    let masked = masked.to_f32();
    let mut sum = vec![0f32; N_EMBD * N_TOK];
    for t in 0..N_TOK {
        for j in 0..N_USED {
            let wj = masked[t * N_USED + j];
            let row = &out[(t * N_USED + j) * N_EMBD..][..N_EMBD];
            for (s, v) in sum[t * N_EMBD..][..N_EMBD].iter_mut().zip(row) {
                *s += wj * v;
            }
        }
    }
    sum
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "{what}: all zero");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5 * scale + 1e-6,
            "{what}: value {i}: {a} vs {b}"
        );
    }
}

#[test]
fn halves_add_up_to_the_full_moe() {
    let e = experts();
    for fused in [true, false] {
        let full = partial(&e, None, None, fused);
        let (a, b) = (
            partial(&e, Some(0), None, fused),
            partial(&e, Some(1), None, fused),
        );
        // Each half alone is not the whole (both halves are routed to).
        assert!(a.iter().zip(&full).any(|(p, q)| (p - q).abs() > 1e-3));
        let sum: Vec<f32> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
        assert_close(&sum, &full, if fused { "fused" } else { "mul_mat_id" });
    }
}

#[test]
fn refuses_a_stack_of_the_wrong_size() {
    let e = experts();
    let wctx = Context::new(4, 0);
    let all: Vec<u64> = (0..N_EXPERT as u64).collect();
    let [up, ..] = load(&wctx, &e, &all);
    assert!(TpEp::new(up, N_EXPERT, 0, 2, None).is_err());
    assert!(TpEp::new(up, N_EXPERT, 0, 3, None).is_err());
    let share = TpEp::new(up, N_EXPERT, 0, 1, None).unwrap();
    // One rank holds them all: nothing is cut.
    assert!(!share.cuts(up));
}

/// A placement giving layer [`IL`] the experts {1, 2, 4, 7} on rank 0 and {0, 3, 5, 6} on
/// rank 1: a permutation of the contiguous split.
fn scattered() -> Placement {
    let mut layers = std::collections::BTreeMap::new();
    layers.insert(IL, vec![1, 0, 0, 1, 0, 1, 1, 0]);
    let p = Placement {
        n_expert: N_EXPERT,
        ranks: 2,
        layers,
    };
    p.check().unwrap();
    p
}

#[test]
fn placed_tables_follow_the_placement() {
    let place = scattered();
    let e = experts();
    let wctx = Context::new(4, 0);
    let [up, ..] = load(&wctx, &e, &ep_place::experts_of(&place.owner(IL), 0));
    let share = TpEp::new(up, N_EXPERT, 0, 2, Some(&place)).unwrap();
    assert_eq!(share.table(IL).to_i32(), [-1, 0, 1, -1, 2, -1, -1, 3]);
    // Other layers keep the contiguous split.
    assert_eq!(share.table(0).to_i32(), [0, 1, 2, 3, -1, -1, -1, -1]);
    let s = share.share(IL);
    assert_eq!(s.first, 0);
    assert!(s.map.is_some());
    let s = share.share(0);
    assert_eq!((s.first, s.map.is_none()), (0, true));
    // A placement for another expert count is refused.
    let other = Placement {
        n_expert: 16,
        ranks: 2,
        layers: Default::default(),
    };
    assert!(TpEp::new(up, N_EXPERT, 0, 2, Some(&other)).is_err());
}

#[test]
fn placed_shares_add_up_to_the_full_moe() {
    let place = scattered();
    let e = experts();
    for fused in [true, false] {
        let full = partial(&e, None, None, fused);
        let (a, b) = (
            partial(&e, Some(0), Some(&place), fused),
            partial(&e, Some(1), Some(&place), fused),
        );
        // The placed halves differ from the contiguous ones.
        let c = partial(&e, Some(0), None, fused);
        assert!(a.iter().zip(&c).any(|(p, q)| (p - q).abs() > 1e-3));
        let sum: Vec<f32> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
        assert_close(
            &sum,
            &full,
            if fused {
                "placed fused"
            } else {
                "placed mul_mat_id"
            },
        );
    }
}

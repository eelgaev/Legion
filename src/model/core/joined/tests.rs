//! Joining weight pairs on the CPU: which pairs join, and that a joined matmul's halves are the
//! separate matmuls' results.

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

const E: i64 = 64;

/// Weights in one CPU buffer, in this order: layer 0's alpha / beta, layer 1's router / gate
/// (a 1-D gate, as the models store it), layer 2's alpha (F16) / beta, layer 3's alpha, a
/// filler, then its beta.
fn weights() -> Context {
    weights_in(Backend::cpu().unwrap().buffer_type(), E)
}

/// [`weights`] in buffers of `buft`, rows of `e`.
fn weights_in(buft: crate::ggml::BufferType, e: i64) -> Context {
    let ctx = Context::new(16, 0);
    for (name, ty, shape) in [
        ("blk.0.ssm_alpha.weight", types::F32, vec![e, 6]),
        ("blk.0.ssm_beta.weight", types::F32, vec![e, 6]),
        ("blk.1.ffn_gate_inp.weight", types::F32, vec![e, 8]),
        ("blk.1.ffn_gate_inp_shexp.weight", types::F32, vec![e]),
        ("blk.2.ssm_alpha.weight", types::F16, vec![e, 6]),
        ("blk.2.ssm_beta.weight", types::F32, vec![e, 6]),
        ("blk.3.ssm_alpha.weight", types::F32, vec![e, 6]),
        ("filler", types::F32, vec![e]),
        ("blk.3.ssm_beta.weight", types::F32, vec![e, 6]),
    ] {
        ctx.new_tensor(ty, &shape).named(name);
    }
    ctx.alloc_tensors(buft).unwrap();
    ctx
}

const NAMES: [&str; 9] = [
    "blk.0.ssm_alpha.weight",
    "blk.0.ssm_beta.weight",
    "blk.1.ffn_gate_inp.weight",
    "blk.1.ffn_gate_inp_shexp.weight",
    "blk.2.ssm_alpha.weight",
    "blk.2.ssm_beta.weight",
    "blk.3.ssm_alpha.weight",
    "filler",
    "blk.3.ssm_beta.weight",
];

fn build(w: &Context, host: bool) -> Option<Joined> {
    Joined::build(
        NAMES.iter().filter_map(|n| w.tensor(n)),
        |n| w.tensor(n),
        host,
    )
}

#[test]
fn only_adjacent_f32_pairs_join() {
    let w = weights();
    let j = build(&w, true).expect("nothing joined");
    let mut names: Vec<String> = j.tensors().map(|t| t.name()).collect();
    names.sort();
    assert_eq!(
        names,
        [
            joined_name("blk.0.ssm_alpha.weight"),
            joined_name("blk.1.ffn_gate_inp.weight"),
        ]
    );
    let t = |n: &str| w.tensor(n).unwrap();
    let ab = j
        .get(t("blk.0.ssm_alpha.weight"), t("blk.0.ssm_beta.weight"))
        .unwrap();
    assert_eq!(ab.ne(), [E, 12, 1, 1]);
    let rg = j
        .get(
            t("blk.1.ffn_gate_inp.weight"),
            t("blk.1.ffn_gate_inp_shexp.weight"),
        )
        .unwrap();
    assert_eq!(rg.ne(), [E, 9, 1, 1]);
    // Not the tensors it was placed over (another order, another pair, a replaced weight).
    assert!(
        j.get(t("blk.0.ssm_beta.weight"), t("blk.0.ssm_alpha.weight"))
            .is_none()
    );
    assert!(
        j.get(t("blk.3.ssm_alpha.weight"), t("blk.3.ssm_beta.weight"))
            .is_none()
    );
    let copy = Context::new(1, 0);
    copy.new_tensor(types::F32, &[E, 6])
        .named("blk.0.ssm_alpha.weight");
    copy.alloc_tensors(Backend::cpu().unwrap().buffer_type())
        .unwrap();
    assert!(
        j.get(
            copy.tensor("blk.0.ssm_alpha.weight").unwrap(),
            t("blk.0.ssm_beta.weight")
        )
        .is_none()
    );
}

/// BF16 pairs (an NVFP4 checkpoint's shared-expert gate / up and DeltaNet alpha / beta) join
/// as BF16 matrices, only stored in that order and of one type.
#[test]
fn adjacent_bf16_pairs_join_as_bf16() {
    let ctx = Context::new(16, 0);
    let weights = [
        ("blk.0.ffn_gate_shexp.weight", types::BF16, 10),
        ("blk.0.ffn_up_shexp.weight", types::BF16, 10),
        ("blk.1.ssm_alpha.weight", types::BF16, 4),
        ("blk.1.ssm_beta.weight", types::BF16, 4),
        // stored the other way round
        ("blk.2.ffn_up_shexp.weight", types::BF16, 10),
        ("blk.2.ffn_gate_shexp.weight", types::BF16, 10),
        // two types
        ("blk.3.ffn_gate_shexp.weight", types::BF16, 10),
        ("blk.3.ffn_up_shexp.weight", types::F16, 10),
    ];
    for (name, ty, rows) in weights {
        ctx.new_tensor(ty, &[E, rows]).named(name);
    }
    ctx.alloc_tensors(Backend::cpu().unwrap().buffer_type())
        .unwrap();
    let all = weights.iter().filter_map(|(n, _, _)| ctx.tensor(n));
    let j = Joined::build(all, |n| ctx.tensor(n), true).expect("nothing joined");
    let mut joined: Vec<String> = j.tensors().map(|t| t.name()).collect();
    joined.sort();
    assert_eq!(
        joined,
        [
            joined_name("blk.0.ffn_gate_shexp.weight"),
            joined_name("blk.1.ssm_alpha.weight"),
        ]
    );
    let t = |n: &str| ctx.tensor(n).unwrap();
    let gu = j
        .get(
            t("blk.0.ffn_gate_shexp.weight"),
            t("blk.0.ffn_up_shexp.weight"),
        )
        .unwrap();
    assert_eq!((gu.ty(), gu.ne()), (types::BF16, [E, 20, 1, 1]));
}

/// A DeltaNet layer's BF16 z, alpha and beta stored back to back join as one run (and alpha /
/// beta as their pair too); a run with anything between its members or of mixed types does
/// not, and `get_run` takes only the very members in their order.
#[test]
fn bf16_z_alpha_beta_join_as_a_run() {
    let ctx = Context::new(16, 0);
    let weights = [
        ("blk.0.attn_gate.weight", types::BF16, 32),
        ("blk.0.ssm_alpha.weight", types::BF16, 4),
        ("blk.0.ssm_beta.weight", types::BF16, 4),
        // something between z and alpha
        ("blk.1.attn_gate.weight", types::BF16, 32),
        ("filler", types::BF16, 2),
        ("blk.1.ssm_alpha.weight", types::BF16, 4),
        ("blk.1.ssm_beta.weight", types::BF16, 4),
        // z of another type
        ("blk.2.attn_gate.weight", types::F16, 32),
        ("blk.2.ssm_alpha.weight", types::BF16, 4),
        ("blk.2.ssm_beta.weight", types::BF16, 4),
    ];
    for (name, ty, rows) in weights {
        ctx.new_tensor(ty, &[E, rows]).named(name);
    }
    ctx.alloc_tensors(Backend::cpu().unwrap().buffer_type())
        .unwrap();
    let all = weights.iter().filter_map(|(n, _, _)| ctx.tensor(n));
    let j = Joined::build(all, |n| ctx.tensor(n), true).expect("nothing joined");
    let mut joined: Vec<String> = j.tensors().map(|t| t.name()).collect();
    joined.sort();
    assert_eq!(
        joined,
        [
            joined_name("blk.0.attn_gate.weight"),
            joined_name("blk.0.ssm_alpha.weight"),
            joined_name("blk.1.ssm_alpha.weight"),
            joined_name("blk.2.ssm_alpha.weight"),
        ]
    );
    let t = |n: &str| ctx.tensor(n).unwrap();
    let [z, a, b] = ["attn_gate", "ssm_alpha", "ssm_beta"].map(|m| t(&format!("blk.0.{m}.weight")));
    let zab = j.get_run(&[z, a, b]).expect("the run");
    assert_eq!((zab.ty(), zab.ne()), (types::BF16, [E, 40, 1, 1]));
    assert_eq!(j.get(a, b).unwrap().ne(), [E, 8, 1, 1]);
    assert!(j.get_run(&[z, a]).is_none(), "a part of the run");
    assert!(j.get_run(&[z, b, a]).is_none(), "another order");
    let [z1, a1, b1] =
        ["attn_gate", "ssm_alpha", "ssm_beta"].map(|m| t(&format!("blk.1.{m}.weight")));
    assert!(j.get_run(&[z1, a1, b1]).is_none(), "not back to back");
}

/// A full-attention layer's BF16 query + gate, key and value stored back to back join as one
/// run; with the output projection between key and value they do not.
#[test]
fn bf16_attention_q_k_v_join_as_a_run() {
    let ctx = Context::new(16, 0);
    let weights = [
        ("blk.3.attn_q.weight", types::BF16, 48),
        ("blk.3.attn_k.weight", types::BF16, 8),
        ("blk.3.attn_v.weight", types::BF16, 8),
        ("blk.3.attn_output.weight", types::BF16, 16),
        ("blk.4.attn_q.weight", types::BF16, 48),
        ("blk.4.attn_k.weight", types::BF16, 8),
        ("blk.4.attn_output.weight", types::BF16, 16),
        ("blk.4.attn_v.weight", types::BF16, 8),
    ];
    for (name, ty, rows) in weights {
        ctx.new_tensor(ty, &[E, rows]).named(name);
    }
    ctx.alloc_tensors(Backend::cpu().unwrap().buffer_type())
        .unwrap();
    let all = weights.iter().filter_map(|(n, _, _)| ctx.tensor(n));
    let j = Joined::build(all, |n| ctx.tensor(n), true).expect("nothing joined");
    let joined: Vec<String> = j.tensors().map(|t| t.name()).collect();
    assert_eq!(joined, [joined_name("blk.3.attn_q.weight")]);
    let t = |n: &str| ctx.tensor(n).unwrap();
    let [q, k, v] = ["attn_q", "attn_k", "attn_v"].map(|m| t(&format!("blk.3.{m}.weight")));
    let qkv = j.get_run(&[q, k, v]).expect("the run");
    assert_eq!((qkv.ty(), qkv.ne()), (types::BF16, [E, 64, 1, 1]));
    assert_eq!(
        RUNS[RUN_QKV],
        ["attn_q.weight", "attn_k.weight", "attn_v.weight"]
    );
    assert_eq!(RUNS[RUN_Z][0], "attn_gate.weight");
}

/// A joined result splits into one row view per member, rows the joined height apart.
#[test]
fn split_run_views_each_members_rows() {
    let ctx = Context::new(8, 0);
    let r = ctx.new_tensor(types::F32, &[40, 3]);
    let parts = split_run(&ctx, r, &[32, 4, 4]);
    assert_eq!(parts.len(), 3);
    for (p, (rows, off)) in parts.iter().zip([(32, 0), (4, 32), (4, 36)]) {
        assert_eq!(p.ne(), [rows, 3, 1, 1]);
        assert_eq!(p.nb()[1], r.nb()[1]);
        // SAFETY: a live view; reads a field.
        assert_eq!(unsafe { (*p.raw()).view_offs }, off * 4);
        assert!(!p.is_contiguous(), "rows of the joined height apart");
    }
}

/// F32 pairs serve small steps only, BF16 pairs prompt steps of the default size only (never
/// without CUDA).
#[test]
fn pair_types_serve_their_steps() {
    let min = i64::from(PROMPT_MIN_TOKENS);
    assert_eq!(prompt_min_tokens(), PROMPT_MIN_TOKENS);
    for n in [1, MAX_TOKENS, MAX_TOKENS + 1, 64, min - 1, min, 4096] {
        let cuda = cfg!(feature = "cuda");
        assert!(!serves(types::F32, n) || n <= MAX_TOKENS, "F32, {n}");
        assert_eq!(serves(types::BF16, n), cuda && n >= min, "BF16, {n}");
        assert!(!serves(types::F16, n), "F16, {n}");
    }
}

/// Host memory joins only when asked to (the model joins GPU weights only).
#[test]
fn host_weights_do_not_join_by_default() {
    let w = weights();
    assert!(build(&w, false).is_none());
}

/// A joined matmul of `w`'s pairs (rows of `e`) on `backend`, with the separate matmuls:
/// per pair and step size, (joined first, joined second, first, second).
fn halves(w: &Context, e: i64, host: bool, backend: &mut Backend) -> Vec<[Vec<f32>; 4]> {
    for (k, n) in NAMES.iter().enumerate() {
        let t = w.tensor(n).unwrap();
        if t.ty() == types::F32 {
            t.set_f32(&noise(t.elements(), k as u64 + 1));
        }
    }
    let j = build(w, host).expect("nothing joined");
    let t = |n: &str| w.tensor(n).unwrap();
    let mut out = Vec::new();
    for (a, b) in [
        ("blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"),
        (
            "blk.1.ffn_gate_inp.weight",
            "blk.1.ffn_gate_inp_shexp.weight",
        ),
    ] {
        let (a, b) = (t(a), t(b));
        for n in 1..=MAX_TOKENS {
            let ctx = Context::new(16, 16);
            let x = ctx.new_tensor(types::F32, &[e, n]).as_input();
            let r = ctx.mul_mat(j.get(a, b).unwrap(), x);
            let [ja, jb] = split_run(&ctx, r, &[a.ne()[1], b.ne()[1]])[..] else {
                panic!("two halves");
            };
            let outs = [
                ctx.cont(ja).as_output(),
                ctx.cont(jb).as_output(),
                ctx.mul_mat(a, x).as_output(),
                ctx.mul_mat(b, x).as_output(),
            ];
            let mut graph = ctx.new_graph(16);
            for o in outs {
                graph.expand(o);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                .unwrap();
            x.set_f32(&noise((e * n) as usize, 99));
            backend.compute(&graph).unwrap();
            out.push(outs.map(|o| o.to_f32()));
        }
    }
    out
}

/// A joined matmul's two halves are the separate matmuls' results (on the CPU up to rounding:
/// ggml-cpu blocks rows by the matrix's height).
#[test]
fn joined_halves_are_the_separate_matmuls() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let w = weights();
    for [ja, jb, a, b] in halves(&w, E, true, &mut backend) {
        for (x, y) in [(ja, a), (jb, b)] {
            assert_eq!(x.len(), y.len());
            for (p, q) in x.iter().zip(&y) {
                assert!((p - q).abs() <= 1e-5 * q.abs().max(1.0), "{p} vs {q}");
            }
        }
    }
}

/// On a GPU, bit for bit: ggml-cuda's F32 vector kernel computes each row on its own.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_joined_halves_are_the_separate_matmuls() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let dev = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device");
    let mut gpu = dev.init().unwrap();
    for e in [64, 2560] {
        let w = weights_in(dev.buffer_type(), e);
        for (i, [ja, jb, a, b]) in halves(&w, e, false, &mut gpu).into_iter().enumerate() {
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&ja), bits(&a), "e {e}, case {i}: first");
            assert_eq!(bits(&jb), bits(&b), "e {e}, case {i}: second");
        }
    }
}

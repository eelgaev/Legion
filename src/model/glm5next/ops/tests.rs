//! The GLM ops' CPU references on small inputs against direct formulas. (The whole layer stack
//! is checked against llama.cpp by `tests/glm5next.rs`.)

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

#[derive(Clone)]
enum Data {
    F32(Vec<f32>),
    I32(Vec<i32>),
    /// A tensor of type `.0` with these bytes (a quantized weight).
    Raw(ffi::ggml_type, Vec<u8>),
}

/// Builds `f` over input tensors of the given shapes and values, runs it on the CPU and
/// returns its output (I32 and F16 outputs converted).
fn run(
    inputs: &[(&[i64], Data)],
    f: impl for<'a> Fn(&'a Context, &[Tensor<'a>]) -> Tensor<'a>,
) -> Vec<f32> {
    run_on(&mut Backend::cpu().unwrap(), inputs, f)
}

/// [`run`] on `backend`.
fn run_on(
    backend: &mut Backend,
    inputs: &[(&[i64], Data)],
    f: impl for<'a> Fn(&'a Context, &[Tensor<'a>]) -> Tensor<'a>,
) -> Vec<f32> {
    let ctx = Context::new(16, 16);
    let ts: Vec<Tensor<'_>> = inputs
        .iter()
        .map(|(ne, d)| {
            let ty = match d {
                Data::F32(_) => types::F32,
                Data::I32(_) => types::I32,
                Data::Raw(ty, _) => *ty,
            };
            ctx.new_tensor(ty, ne).as_input()
        })
        .collect();
    let out = f(&ctx, &ts).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    for (t, (_, d)) in ts.iter().zip(inputs) {
        match d {
            Data::F32(v) => t.set_f32(v),
            Data::I32(v) => t.set_i32(v),
            Data::Raw(_, b) => t.set_bytes(b),
        }
    }
    backend.compute(&graph).unwrap();
    if out.ty() == types::I32 {
        out.to_i32().into_iter().map(|x| x as f32).collect()
    } else if out.ty() == types::F16 {
        out.bytes()
            .as_chunks::<2>()
            .0
            .iter()
            // SAFETY: a plain conversion.
            .map(|b| unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes(*b)) })
            .collect()
    } else {
        out.to_f32()
    }
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{i}: {x} vs {y}");
    }
}

#[test]
fn bf16_rounds_to_nearest_even() {
    assert_eq!(to_bf16(1.0), 0x3F80);
    // Exactly halfway: to the even mantissa.
    assert_eq!(to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
    assert_eq!(to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
    // Above halfway rounds up.
    assert_eq!(to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
    assert!(f32::from_bits((to_bf16(f32::NAN) as u32) << 16).is_nan());
}

#[test]
fn sinkhorn_is_nearly_doubly_stochastic() {
    let hc = 4;
    let mut c = noise(hc * hc, 7)
        .iter()
        .map(|x| 3.0 * x)
        .collect::<Vec<_>>();
    sinkhorn(&mut c, hc, 1e-6, 20);
    for i in 0..hc {
        let col: f32 = (0..hc).map(|j| c[i * hc + j]).sum();
        let row: f32 = (0..hc).map(|j| c[j * hc + i]).sum();
        assert!(
            (col - 1.0).abs() < 1e-3 && (row - 1.0).abs() < 1e-3,
            "{col} {row}"
        );
    }
    assert!(c.iter().all(|&x| x > 0.0));
}

#[test]
fn glu_clamps_gate_above_and_up_both_ways() {
    let g = vec![-20.0, -1.0, 0.5, 30.0];
    let u = vec![50.0, -50.0, 2.0, 3.0];
    let got = run(
        &[(&[4], Data::F32(g.clone())), (&[4], Data::F32(u.clone()))],
        |ctx, t| glu(ctx, t[0], t[1], 10.0),
    );
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&g, &u)| silu(g.min(10.0)) * u.clamp(-10.0, 10.0))
        .collect();
    close(&got, &want, 1e-6);
    let plain = run(
        &[(&[4], Data::F32(g.clone())), (&[4], Data::F32(u.clone()))],
        |ctx, t| glu(ctx, t[0], t[1], 0.0),
    );
    close(
        &plain,
        &[
            silu(-20.0) * 50.0,
            silu(-1.0) * -50.0,
            silu(0.5) * 2.0,
            silu(30.0) * 3.0,
        ],
        1e-6,
    );
}

#[test]
fn router_picks_biased_top_k_and_weighs_unbiased() {
    let (ne, k) = (6usize, 2i64);
    let logits = vec![
        0.0, 2.0, 1.0, 2.0, -1.0, 0.5, /* token 1 */ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let bias = vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0];
    let ids = run(
        &[
            (&[ne as i64, 2], Data::F32(logits.clone())),
            (&[ne as i64], Data::F32(bias.clone())),
        ],
        |ctx, t| router_ids(ctx, t[0], t[1], k),
    );
    // Token 0: expert 4 by its bias, then 1 (tied with 3: the lower id). Token 1: all tied
    // but 4.
    assert_eq!(ids, vec![4.0, 1.0, 4.0, 0.0]);
    let w = run(
        &[
            (&[ne as i64, 2], Data::F32(logits.clone())),
            (&[k, 2], Data::I32(ids.iter().map(|&x| x as i32).collect())),
        ],
        |ctx, t| router_w(ctx, t[0], t[1], true, 2.5),
    );
    let (a, b) = (sigm(-1.0), sigm(2.0));
    let (c, d) = (sigm(1.0), sigm(1.0));
    close(
        &w,
        &[
            2.5 * a / (a + b),
            2.5 * b / (a + b),
            2.5 * c / (c + d),
            2.5 * d / (c + d),
        ],
        1e-6,
    );
}

#[test]
fn route_bias_favours_cached_experts_in_selection_only() {
    let (ne, k) = (6usize, 2i64);
    let logits = vec![0.0, 2.0, 1.0, 1.9, -1.0, 0.5];
    let bias = vec![0.0; ne];
    // Experts 2 and 3 hold slots 0 and 1 (of 2); the others are host experts (2 + e).
    let table = vec![2, 3, 0, 1, 6, 7];
    let pick = |bonus: f32| {
        run(
            &[
                (&[ne as i64, 1], Data::F32(logits.clone())),
                (&[ne as i64], Data::F32(bias.clone())),
                (&[ne as i64], Data::I32(table.clone())),
            ],
            |ctx, t| router_ids_resident(ctx, t[0], t[1], t[2], 2, bonus, k),
        )
    };
    let plain = run(
        &[
            (&[ne as i64, 1], Data::F32(logits.clone())),
            (&[ne as i64], Data::F32(bias.clone())),
        ],
        |ctx, t| router_ids(ctx, t[0], t[1], k),
    );
    // No bonus: the plain router's choice.
    assert_eq!(pick(0.0), plain);
    assert_eq!(plain, vec![1.0, 3.0]);
    // A small bonus lifts cached expert 3 over host expert 1 (2.0 vs 1.9 in logits).
    assert_eq!(pick(0.1), vec![3.0, 1.0]);
    // A big one takes both cached experts.
    assert_eq!(pick(1.0), vec![3.0, 2.0]);
}

#[test]
fn weighted_sum_and_argmax() {
    let (e, k, n) = (3usize, 2usize, 2usize);
    let y = noise(e * k * n, 3);
    let w = noise(k * n, 4);
    let got = run(
        &[
            (&[e as i64, k as i64, n as i64], Data::F32(y.clone())),
            (&[k as i64, n as i64], Data::F32(w.clone())),
        ],
        |ctx, t| weighted_sum(ctx, t[0], t[1]),
    );
    let want: Vec<f32> = (0..n)
        .flat_map(|t| {
            let (y, w) = (&y, &w);
            (0..e).map(move |i| (0..k).map(|j| y[(t * k + j) * e + i] * w[t * k + j]).sum())
        })
        .collect();
    close(&got, &want, 1e-6);
    let am = run(
        &[(&[3, 2], Data::F32(vec![1.0, 3.0, 3.0, -1.0, -2.0, -3.0]))],
        |ctx, t| argmax(ctx, t[0]),
    );
    assert_eq!(am, vec![1.0, 0.0]);
}

/// Two tokens of one head from a zero state, against the delta rule written out.
#[test]
fn kda_scan_follows_the_gated_delta_rule() {
    let d = 8usize;
    let (q, k, v) = (noise(2 * d, 11), noise(2 * d, 12), noise(2 * d, 13));
    let g: Vec<f32> = noise(2 * d, 14).iter().map(|x| -0.5 * x.abs()).collect();
    let beta = vec![0.7, 0.4];
    let got = run(
        &[
            (&[d as i64, 1, 2], Data::F32(q.clone())),
            (&[d as i64, 1, 2], Data::F32(k.clone())),
            (&[d as i64, 1, 2], Data::F32(v.clone())),
            (&[d as i64, 1, 2], Data::F32(g.clone())),
            (&[1, 2], Data::F32(beta.clone())),
            (&[d as i64, d as i64, 1, 1], Data::F32(vec![0.0; d * d])),
        ],
        |ctx, t| kda_scan(ctx, t[0], t[1], t[2], t[3], t[4], t[5], 2),
    );
    let l2 = |x: &[f32], s: f32| {
        let n = (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        x.iter().map(|v| v / n * s).collect::<Vec<_>>()
    };
    let mut s = vec![vec![0f32; d]; d]; // s[j][i]: value j, key i
    let mut want = Vec::new();
    for t in 0..2 {
        let qn = l2(&q[t * d..][..d], 1.0 / (d as f32).sqrt());
        let kn = l2(&k[t * d..][..d], 1.0);
        for j in 0..d {
            for i in 0..d {
                s[j][i] *= g[t * d + i].exp();
            }
            let sk: f32 = (0..d).map(|i| s[j][i] * kn[i]).sum();
            let delta = (v[t * d + j] - sk) * beta[t];
            for i in 0..d {
                s[j][i] += delta * kn[i];
            }
            want.push((0..d).map(|i| s[j][i] * qn[i]).sum());
        }
    }
    close(&got, &want, 1e-5);
}

#[test]
fn gated_rms_normalizes_rows() {
    let d = 4usize;
    let o = noise(2 * d, 21);
    let w = vec![1.0, 2.0, 0.5, -1.0];
    let gate = noise(2 * d, 22);
    let got = run(
        &[
            (&[d as i64, 2], Data::F32(o.clone())),
            (&[d as i64], Data::F32(w.clone())),
            (&[d as i64, 2], Data::F32(gate.clone())),
        ],
        |ctx, t| gated_rms(ctx, t[0], t[1], Some(t[2]), 1e-5),
    );
    let mut want = Vec::new();
    for r in 0..2 {
        let row = &o[r * d..][..d];
        let s = 1.0 / (row.iter().map(|x| x * x).sum::<f32>() / d as f32 + 1e-5).sqrt();
        for i in 0..d {
            want.push(row[i] * s * w[i] * sigm(gate[r * d + i]));
        }
    }
    close(&got, &want, 1e-5);
}

/// [`run`] with several outputs: each output's values, then every input's after the run
/// (the states an op advances in place, the buffers it stores into).
#[allow(clippy::type_complexity)]
fn run_all(
    inputs: &[(&[i64], Data)],
    f: impl for<'a> Fn(&'a Context, &[Tensor<'a>]) -> Vec<Tensor<'a>>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    run_all_on(&mut Backend::cpu().unwrap(), inputs, f)
}

/// [`run_all`] on `backend`.
#[allow(clippy::type_complexity)]
fn run_all_on(
    backend: &mut Backend,
    inputs: &[(&[i64], Data)],
    f: impl for<'a> Fn(&'a Context, &[Tensor<'a>]) -> Vec<Tensor<'a>>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let ctx = Context::new(64, 64);
    let ts: Vec<Tensor<'_>> = inputs
        .iter()
        .map(|(ne, d)| {
            let ty = match d {
                Data::F32(_) => types::F32,
                Data::I32(_) => types::I32,
                Data::Raw(ty, _) => *ty,
            };
            ctx.new_tensor(ty, ne).as_input()
        })
        .collect();
    let outs: Vec<Tensor<'_>> = f(&ctx, &ts).into_iter().map(|t| t.as_output()).collect();
    let mut graph = ctx.new_graph(64);
    // The inputs a graph does not read get memory too (they are read back).
    for &t in outs.iter().chain(&ts) {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    for (t, (_, d)) in ts.iter().zip(inputs) {
        match d {
            Data::F32(v) => t.set_f32(v),
            Data::I32(v) => t.set_i32(v),
            Data::Raw(_, b) => t.set_bytes(b),
        }
    }
    backend.compute(&graph).unwrap();
    let read = |t: &Tensor<'_>| {
        if t.ty() == types::I32 {
            t.to_i32().into_iter().map(|x| x as f32).collect()
        } else if t.ty() == types::F32 {
            t.to_f32()
        } else {
            Vec::new() // a quantized weight
        }
    };
    (
        outs.iter().map(read).collect(),
        ts.iter().map(read).collect(),
    )
}

/// The MTP head's input: the normed embedding (times `enorm`) first, then the normed hidden
/// row (times `hnorm`), per token.
#[test]
fn nextn_in_stacks_the_normed_embedding_over_the_normed_hidden_row() {
    let (d, n) = (4usize, 3usize);
    let (e, h) = (noise(d * n, 31), noise(d * n, 32));
    let (we, wh) = (vec![1.0, 2.0, 3.0, 4.0], vec![-1.0, 0.5, 0.25, 2.0]);
    let got = run(
        &[
            (&[d as i64, n as i64], Data::F32(e.clone())),
            (&[d as i64, n as i64], Data::F32(h.clone())),
            (&[d as i64], Data::F32(we.clone())),
            (&[d as i64], Data::F32(wh.clone())),
        ],
        |ctx, t| nextn_in(ctx, t[0], t[1], t[2], t[3], 1e-5),
    );
    let normed = |x: &[f32], w: &[f32]| {
        let s = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-5).sqrt();
        x.iter().zip(w).map(|(v, w)| v * s * w).collect::<Vec<_>>()
    };
    let mut want = Vec::new();
    for t in 0..n {
        want.extend(normed(&e[t * d..][..d], &we));
        want.extend(normed(&h[t * d..][..d], &wh));
    }
    close(&got, &want, 1e-6);
}

#[test]
fn get_cols_add_and_pmax() {
    let (d, n) = (3usize, 4usize);
    let x = noise(d * n, 41);
    let got = run(
        &[
            (&[d as i64, n as i64], Data::F32(x.clone())),
            (&[2], Data::I32(vec![3, 1])),
        ],
        |ctx, t| get_cols(ctx, t[0], t[1]),
    );
    let want: Vec<f32> = [3, 1]
        .iter()
        .flat_map(|&c| x[c * d..][..d].to_vec())
        .collect();
    assert_eq!(got, want);

    let y = noise(d * n, 42);
    let got = run(
        &[
            (&[d as i64, n as i64], Data::F32(x.clone())),
            (&[d as i64, n as i64], Data::F32(y.clone())),
        ],
        |ctx, t| add(ctx, t[0], t[1]),
    );
    let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_eq!(got, want);

    let got = run(
        &[(&[d as i64, n as i64], Data::F32(x.clone()))],
        |ctx, t| pmax(ctx, t[0]),
    );
    let want: Vec<f32> = (0..n)
        .map(|t| {
            let r = &x[t * d..][..d];
            let m = r.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            1.0 / r.iter().map(|v| (v - m).exp()).sum::<f32>()
        })
        .collect();
    close(&got, &want, 1e-6);
}

/// A verify group's conv: the decode outputs, the states untouched, the inputs stored; the
/// stored inputs of the kept tokens, convolved again, advance the states as decoding them
/// did.
#[test]
fn kda_conv_verify_stores_inputs_and_keeps_the_state() {
    let (c, n_tok, n_seqs, keep) = (5usize, 3usize, 2usize, 2usize);
    let n = n_tok * n_seqs;
    let x = noise(c * n, 51);
    let w = noise(4 * c, 52);
    let st = noise(3 * c * n_seqs, 53);
    let shapes: [Vec<i64>; 4] = [
        vec![c as i64, n as i64],
        vec![4, 1, c as i64],
        vec![3, c as i64, n_seqs as i64],
        vec![c as i64, n_tok as i64, n_seqs as i64],
    ];
    let data = |store: &[f32]| {
        vec![
            (&shapes[0][..], Data::F32(x.clone())),
            (&shapes[1][..], Data::F32(w.clone())),
            (&shapes[2][..], Data::F32(st.clone())),
            (&shapes[3][..], Data::F32(store.to_vec())),
        ]
    };
    let zeros = vec![0.0; c * n];
    // Decode: the outputs and the advanced state.
    let (o, after) = run_all(&data(&zeros), |ctx, t| {
        vec![kda_conv(ctx, t[0], t[1], t[2], 0, n_tok as i64)]
    });
    // Verify: the same outputs, the state as it was, the inputs stored.
    let (ov, v_after) = run_all(&data(&zeros), |ctx, t| {
        vec![kda_conv_verify(
            ctx,
            t[0],
            t[1],
            t[2],
            0,
            n_tok as i64,
            t[3],
        )]
    });
    assert_eq!(ov[0], o[0]);
    assert_eq!(v_after[2], st);
    assert_eq!(v_after[3], x);
    assert_ne!(after[2], st);

    // The commit: the first `keep` stored tokens of each sequence ([c, keep, n_seqs] view)
    // through the conv, against decoding them alone.
    let kept: Vec<f32> = (0..n_seqs)
        .flat_map(|s| x[s * n_tok * c..][..keep * c].to_vec())
        .collect();
    let (_, want) = run_all(
        &[
            (&[c as i64, (keep * n_seqs) as i64], Data::F32(kept)),
            (&shapes[1][..], Data::F32(w.clone())),
            (&shapes[2][..], Data::F32(st.clone())),
        ],
        |ctx, t| vec![kda_conv(ctx, t[0], t[1], t[2], 0, keep as i64)],
    );
    let (_, got) = run_all(&data(&x), |ctx, t| {
        let nb = t[3].nb();
        let view = ctx.view_3d(
            t[3],
            [c as i64, keep as i64, n_seqs as i64],
            [nb[1], nb[2]],
            0,
        );
        vec![kda_conv(ctx, view, t[1], t[2], 0, keep as i64)]
    });
    assert_eq!(got[2], want[2]);
}

/// A commit's q, k and v convs in one op (#676): each projection's outputs and conv states
/// bit for bit as its own op gives them, from strided views of stored inputs.
#[test]
fn kda_conv3_matches_three_convs() {
    let (c, n_tok, n_seqs, keep) = (5usize, 3usize, 2usize, 2usize);
    let xs: Vec<Vec<f32>> = (0..3).map(|j| noise(c * n_tok * n_seqs, 71 + j)).collect();
    let ws: Vec<Vec<f32>> = (0..3).map(|j| noise(4 * c, 81 + j)).collect();
    let st = noise(3 * 3 * c * n_seqs, 91);
    let x_ne = [c as i64, n_tok as i64, n_seqs as i64];
    let w_ne = [4, 1, c as i64];
    let st_ne = [3, 3 * c as i64, n_seqs as i64];
    let mut inputs: Vec<(&[i64], Data)> = Vec::new();
    for j in 0..3 {
        inputs.push((&x_ne[..], Data::F32(xs[j].clone())));
        inputs.push((&w_ne[..], Data::F32(ws[j].clone())));
    }
    inputs.push((&st_ne[..], Data::F32(st.clone())));
    // The first `keep` stored tokens of each sequence.
    fn kept<'a>(ctx: &'a Context, t: Tensor<'a>, keep: i64) -> Tensor<'a> {
        let (ne, nb) = (t.ne(), t.nb());
        ctx.view_3d(t, [ne[0], keep, ne[2]], [nb[1], nb[2]], 0)
    }
    let n_keep = keep as i64;
    let (want, want_st) = run_all(&inputs, |ctx, t| {
        (0..3)
            .map(|j| {
                let x = kept(ctx, t[2 * j], n_keep);
                kda_conv(ctx, x, t[2 * j + 1], t[6], (j * c) as i64, n_keep)
            })
            .collect()
    });
    let (got, got_st) = run_all(&inputs, |ctx, t| {
        let x = [0, 2, 4].map(|i| kept(ctx, t[i], n_keep));
        vec![kda_conv3(ctx, x, [t[1], t[3], t[5]], t[6], n_keep)]
    });
    assert_eq!(got[0], want.concat());
    assert_eq!(got_st[6], want_st[6]);
    assert_ne!(got_st[6], st);
}

/// A verify group's scan: the decode outputs, the states untouched, each token's decay and
/// beta stored; the commit's scan of the kept tokens over the stored rows (strided views)
/// advances the states exactly as decoding them did.
#[test]
fn kda_scan_verify_and_replay_match_decoding() {
    let (d, h, n_tok, n_seqs, keep) = (8usize, 2usize, 3usize, 2usize, 2usize);
    let n = n_tok * n_seqs;
    let (q, k, v) = (
        noise(d * h * n, 61),
        noise(d * h * n, 62),
        noise(d * h * n, 63),
    );
    let g: Vec<f32> = noise(d * h * n, 64)
        .iter()
        .map(|x| -0.5 * x.abs())
        .collect();
    let beta: Vec<f32> = noise(h * n, 65).iter().map(|x| 0.5 + 0.4 * x).collect();
    let st = noise(d * d * h * n_seqs, 66);
    let (di, hi, ni) = (d as i64, h as i64, n as i64);
    let (ti, si) = (n_tok as i64, n_seqs as i64);
    let shapes: [Vec<i64>; 8] = [
        vec![di, hi, ni],
        vec![di, hi, ni],
        vec![di, hi, ni],
        vec![di, hi, ni],
        vec![hi, ni],
        vec![di, di, hi, si],
        vec![di * hi, ti, si],
        vec![hi, ti, si],
    ];
    let data = |gs: &[f32], bs: &[f32]| {
        let vals = [&q, &k, &v, &g, &beta, &st];
        let mut out: Vec<(&[i64], Data)> = vals
            .iter()
            .zip(&shapes)
            .map(|(x, s)| (&s[..], Data::F32(x.to_vec())))
            .collect();
        out.push((&shapes[6][..], Data::F32(gs.to_vec())));
        out.push((&shapes[7][..], Data::F32(bs.to_vec())));
        out
    };
    let (gz, bz) = (vec![0.0; d * h * n], vec![0.0; h * n]);
    let (o, _) = run_all(&data(&gz, &bz), |ctx, t| {
        vec![kda_scan(ctx, t[0], t[1], t[2], t[3], t[4], t[5], ti)]
    });
    let (ov, after) = run_all(&data(&gz, &bz), |ctx, t| {
        vec![kda_scan_verify(
            ctx,
            t[0],
            t[1],
            t[2],
            t[3],
            t[4],
            t[5],
            ti,
            [t[6], t[7]],
        )]
    });
    assert_eq!(ov[0], o[0]);
    assert_eq!(after[5], st);
    assert_eq!(after[6], g);
    assert_eq!(after[7], beta);

    // Decoding the kept tokens alone (contiguous inputs) ...
    let first = |x: &[f32], w: usize| -> Vec<f32> {
        (0..n_seqs)
            .flat_map(|s| x[s * n_tok * w..][..keep * w].to_vec())
            .collect()
    };
    let nk = (keep * n_seqs) as i64;
    let kept = |x: &[f32]| Data::F32(first(x, d * h));
    let (_, want) = run_all(
        &[
            (&[di, hi, nk], kept(&q)),
            (&[di, hi, nk], kept(&k)),
            (&[di, hi, nk], kept(&v)),
            (&[di, hi, nk], kept(&g)),
            (&[hi, nk], Data::F32(first(&beta, h))),
            (&shapes[5][..], Data::F32(st.clone())),
        ],
        |ctx, t| {
            vec![kda_scan(
                ctx,
                t[0],
                t[1],
                t[2],
                t[3],
                t[4],
                t[5],
                keep as i64,
            )]
        },
    );
    // ... and the commit's scan over the stored rows.
    let (_, got) = run_all(
        &[
            (&[di, hi, nk], kept(&q)),
            (&[di, hi, nk], kept(&k)),
            (&[di, hi, nk], kept(&v)),
            (&shapes[6][..], Data::F32(g.clone())),
            (&shapes[7][..], Data::F32(beta.clone())),
            (&shapes[5][..], Data::F32(st.clone())),
        ],
        |ctx, t| {
            let (gn, bn) = (t[3].nb(), t[4].nb());
            let kk = keep as i64;
            let gv = ctx.view_4d(t[3], [di, hi, kk, si], [d * 4, gn[1], gn[2]], 0);
            let bv = ctx.view_3d(t[4], [hi, kk, si], [bn[1], bn[2]], 0);
            vec![kda_scan(ctx, t[0], t[1], t[2], gv, bv, t[5], kk)]
        },
    );
    assert_eq!(got[5], want[5]);
}

/// The KDA conv and scan nodes carry the GB10 kernel switch ([`set_gb10_linear`]) as their last
/// parameter, which the CUDA launcher reads to take the prompt kernels (#571); the CPU
/// references ignore it.
#[test]
fn kda_nodes_carry_the_gb10_switch() {
    let flags = |on: bool| {
        set_gb10_linear(on);
        let ctx = Context::new(16, 0);
        let (d, h, n) = (8i64, 2i64, 4i64);
        let x = ctx.new_tensor(types::F32, &[5, n]);
        let w = ctx.new_tensor(types::F32, &[4, 1, 5]);
        let cs = ctx.new_tensor(types::F32, &[3, 5, 1]);
        let conv = kda_conv(&ctx, x, w, cs, 0, n);
        let [q, k, v, g] = [0; 4].map(|_| ctx.new_tensor(types::F32, &[d, h, n]));
        let beta = ctx.new_tensor(types::F32, &[h, n]);
        let st = ctx.new_tensor(types::F32, &[d, d, h, 1]);
        let scan = kda_scan(&ctx, q, k, v, g, beta, st, n);
        // SAFETY: both nodes live in `ctx`, alive here.
        let (pc, ps) = unsafe {
            (
                crate::ggml::external_params(&*conv.raw()),
                crate::ggml::external_params(&*scan.raw()),
            )
        };
        (pc[3], ps[2])
    };
    let on = flags(true);
    let off = flags(false);
    assert_eq!(on, (1, 1));
    assert_eq!(off, (0, 0));
}

#[test]
fn f16_rounds_like_ggml_and_linear_h_matches_linear() {
    let x = noise(6 * 3, 41)
        .iter()
        .map(|v| v * 300.0)
        .collect::<Vec<_>>();
    let w = noise(6 * 4, 42);
    let got = run(&[(&[6, 3], Data::F32(x.clone()))], |ctx, t| f16(ctx, t[0]));
    // SAFETY: plain conversions.
    let want: Vec<f32> = x
        .iter()
        .map(|&v| unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(v)) })
        .collect();
    assert_eq!(got, want);
    let inputs = [
        (&[6i64, 4][..], Data::F32(w.clone())),
        (&[6, 3][..], Data::F32(x.clone())),
    ];
    let plain = run(&inputs, |ctx, t| linear(ctx, t[0], t[1], 0.5, None));
    let inputs = [(&[6i64, 4][..], Data::F32(w)), (&[6, 3][..], Data::F32(x))];
    let with_h = run(&inputs, |ctx, t| {
        let out = linear_h(ctx, t[0], t[1], Some(f16(ctx, t[1])), 0.5, None);
        // SAFETY: a node of `ctx`, alive here.
        let p = unsafe { crate::ggml::external_params(&*out.raw()) };
        assert_eq!(p[3], 1);
        out
    });
    assert_eq!(plain, with_h);
}

#[test]
fn gb10_kernels_keep_the_plain_gemv_when_asked() {
    assert!(!gb10_kernels(false, true, 1));
    assert!(!gb10_kernels(false, true, 512));
    assert!(gb10_kernels(true, true, 1));
    assert!(gb10_kernels(true, true, 8));
    assert!(gb10_kernels(true, true, 512));
    assert!(!gb10_kernels(true, false, 1));
    assert!(!gb10_kernels(true, false, 8));
    assert!(gb10_kernels(true, false, 9));
    assert!(gb10_kernels(true, false, 512));
}

#[test]
fn wf16_rounds_q6k_values_like_ggml() {
    let (k, m, h) = (512usize, 3usize, 2usize);
    let w = noise(k * m * h, 43)
        .iter()
        .map(|v| v * 7.0)
        .collect::<Vec<_>>();
    let q = crate::ggml::quantize(types::Q6_K, &w, k);
    let ne = [k as i64, m as i64, h as i64];
    let got = run(&[(&ne, Data::Raw(types::Q6_K, q.clone()))], |ctx, t| {
        wf16(ctx, t[0])
    });
    // SAFETY: plain conversions.
    let want: Vec<f32> = crate::ggml::dequantize(types::Q6_K, &q, k * m * h)
        .iter()
        .map(|&v| unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(v)) })
        .collect();
    assert_eq!(got, want);
}

#[test]
fn q6k_f16_only_for_long_prompt_steps_on_the_gb10_kernels() {
    assert!(q6k_f16_takes(true, true, Q6K_F16_MIN));
    assert!(q6k_f16_takes(true, true, 2048));
    assert!(!q6k_f16_takes(true, true, Q6K_F16_MIN - 1));
    assert!(!q6k_f16_takes(true, true, 1));
    assert!(!q6k_f16_takes(false, true, 2048));
    assert!(!q6k_f16_takes(true, false, 2048));
}

/// A linear node carries its F16 GEMM kernel ([`LinTc`]) as params[4]: 0 (the device's choice)
/// from [`linear`] and [`linear_h`], the explicit one from [`linear_tc_kernel`] (#862).
#[test]
fn linear_nodes_carry_the_tc_kernel() {
    let ctx = Context::new(16, 0);
    let w = ctx.new_tensor(types::F16, &[32, 4]);
    let x = ctx.new_tensor(types::F32, &[32, 3]);
    // SAFETY: the nodes live in `ctx`, alive here.
    let params = |t: Tensor<'_>| unsafe { crate::ggml::external_params(&*t.raw()) };
    assert_eq!(params(linear(&ctx, w, x, 1.0, None))[4], 0);
    assert_eq!(params(linear_h(&ctx, w, x, None, 1.0, None))[4], 0);
    for (kernel, want) in [
        (LinTc::Auto, 0),
        (LinTc::Tc, 1),
        (LinTc::Gb(0), 2),
        (LinTc::Gb(5), 7),
    ] {
        let p = params(linear_tc_kernel(&ctx, w, x, None, 1.0, None, kernel));
        assert_eq!(p[4], want, "{kernel:?}");
    }
}

/// A lone product's `x` goes to F16 first only on a compute capability 12.x GPU, for a product
/// that takes linear_tc, with 1024+ weight rows and rows of a multiple of 512 values (#862).
#[test]
fn lone_products_take_f16_rows_only_on_cc12_linear_tc() {
    // Maya-S: KDA and MLA attn_output, attn_q_a, attn_q_b, the shared expert's down
    for (k, m) in [
        (8192, 4096),
        (16384, 4096),
        (4096, 1536),
        (1536, 16384),
        (2048, 4096),
    ] {
        assert!(xh_one_pays(true, true, k, m), "{k} x {m}");
        assert!(!xh_one_pays(false, true, k, m), "not cc 12.x");
        assert!(!xh_one_pays(true, false, k, m), "not linear_tc");
    }
    // attn_kv_a_mqa: 512 rows
    assert!(!xh_one_pays(true, true, 4096, 512));
    assert!(xh_one_pays(true, true, 4096, XH_ONE_MIN_ROWS));
    assert!(!xh_one_pays(true, true, 4096 + 256, 4096));
}

/// The MLA products hand each other F16 rows only on a compute capability 12.x GPU where they
/// take linear_tc (#862).
#[test]
fn f16_chain_only_on_cc12_linear_tc() {
    assert!(f16_chain_pays(true, true));
    assert!(!f16_chain_pays(false, true), "not cc 12.x");
    assert!(!f16_chain_pays(true, false), "not linear_tc");
    assert!(!f16_chain_pays(false, false));
}

/// [`linear_io`]: F16 outputs are the F32 ones rounded as [`f16`] rounds them, F16 rows give the
/// product of their values, and the node carries both (dst F16, params[5]); one matrix and
/// per-head ones (#862).
#[test]
fn linear_io_f16_outputs_and_rows() {
    for (k, m, h, n) in [(32i64, 6i64, 1i64, 5i64), (32, 4, 3, 2)] {
        let w = noise((k * m * h) as usize, 61);
        let x: Vec<f32> = noise((k * h * n) as usize, 62)
            .iter()
            .map(|v| v * 3.0)
            .collect();
        let xne: Vec<i64> = if h == 1 { vec![k, n] } else { vec![k, h, n] };
        let inputs = [
            (&[k, m, h][..], Data::F32(w.clone())),
            (&xne[..], Data::F32(x.clone())),
        ];
        let plain = run(&inputs, |ctx, t| linear(ctx, t[0], t[1], 0.5, None));
        let out16 = run(&inputs, |ctx, t| {
            let out = linear_io(ctx, t[0], t[1], None, 0.5, None, LinTc::Auto, true);
            assert_eq!(out.ty(), types::F16);
            // SAFETY: a node of `ctx`, alive here.
            assert_eq!(unsafe { crate::ggml::external_params(&*out.raw()) }[5], 1);
            out
        });
        // SAFETY: plain conversions.
        let want: Vec<f32> = plain
            .iter()
            .map(|&v| unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(v)) })
            .collect();
        assert_eq!(out16, want, "h {h}");
        // F16 rows: the product of the rounded values
        let rows16 = run(&inputs, |ctx, t| {
            linear_io(
                ctx,
                t[0],
                f16(ctx, t[1]),
                None,
                0.5,
                None,
                LinTc::Auto,
                false,
            )
        });
        // SAFETY: plain conversions.
        let xr: Vec<f32> = x
            .iter()
            .map(|&v| unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(v)) })
            .collect();
        let inputs = [(&[k, m, h][..], Data::F32(w)), (&xne[..], Data::F32(xr))];
        let want = run(&inputs, |ctx, t| linear(ctx, t[0], t[1], 0.5, None));
        assert_eq!(rows16, want, "h {h}");
    }
    let ctx = Context::new(16, 0);
    let w = ctx.new_tensor(types::F16, &[32, 4]);
    let x = ctx.new_tensor(types::F32, &[32, 3]);
    let out = linear(&ctx, w, x, 1.0, None);
    assert_eq!(out.ty(), types::F32);
    // SAFETY: a node of `ctx`, alive here.
    assert_eq!(unsafe { crate::ggml::external_params(&*out.raw()) }[5], 0);
}

/// On CUDA with the GB10 kernels, linear_tc's F16 handoff gives the bits of the F32 path
/// (#862): F16 outputs equal the F32 ones through [`f16`] (F32 and F16 activations, one matrix
/// and per-head ones), and F16 rows give the outputs of the F32 rows they round.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_linear_f16_handoff_gives_the_same_bits() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    set_gb10_linear(true);
    let f16_raw = |v: &[f32]| -> Vec<u8> {
        v.iter()
            // SAFETY: a plain conversion.
            .flat_map(|&x| unsafe { ffi::ggml_fp32_to_fp16(x) }.to_le_bytes())
            .collect()
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    // (k, m, h, n): q_b-like (4096+ rows, shape 2), ragged tiles, per-head matrices
    for (k, m, h, n) in [
        (512i64, 4224i64, 1i64, 1100i64),
        (512, 600, 1, 700),
        (256, 136, 3, 300),
    ] {
        let wv: Vec<f32> = noise((k * m * h) as usize, 71)
            .iter()
            .map(|v| v * 0.05)
            .collect();
        let xv: Vec<f32> = noise((k * h * n) as usize, 72)
            .iter()
            .map(|v| v * 4.0)
            .collect();
        let xne: Vec<i64> = if h == 1 { vec![k, n] } else { vec![k, h, n] };
        let inputs = [
            (&[k, m, h][..], Data::Raw(types::F16, f16_raw(&wv))),
            (&xne[..], Data::F32(xv.clone())),
        ];
        for xh in [false, true] {
            if xh && h > 1 {
                continue;
            }
            let want = run_on(&mut gpu, &inputs, |ctx, t| {
                let x16 = xh.then(|| f16(ctx, t[1]));
                let y = linear_h(ctx, t[0], t[1], x16, 1.0, None);
                f16(ctx, ctx.reshape_2d(y, y.ne()[0] * y.ne()[1], y.ne()[2]))
            });
            let got = run_on(&mut gpu, &inputs, |ctx, t| {
                let x16 = xh.then(|| f16(ctx, t[1]));
                linear_io(ctx, t[0], t[1], x16, 1.0, None, LinTc::Auto, true)
            });
            assert_eq!(
                bits(&got),
                bits(&want),
                "out F16: k {k} m {m} h {h} n {n} xh {xh}"
            );
        }
        let want = run_on(&mut gpu, &inputs, |ctx, t| {
            linear(ctx, t[0], t[1], 1.0, None)
        });
        let got = run_on(&mut gpu, &inputs, |ctx, t| {
            linear_io(
                ctx,
                t[0],
                f16(ctx, t[1]),
                None,
                1.0,
                None,
                LinTc::Auto,
                false,
            )
        });
        assert_eq!(bits(&got), bits(&want), "F16 rows: k {k} m {m} h {h} n {n}");
    }
    set_gb10_linear(false);
}

/// On CUDA, every shape of linear_tc_gb (and the device's choice) gives linear_tc's bits
/// (#862): F16 and BF16 weights, F32 and F16 (`xh`) activations, rows and tokens that are not
/// whole tiles, a row scale, and per-head matrices.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_linear_tc_gb_shapes_give_linear_tcs_bits() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    set_gb10_linear(true);
    let f16_raw = |v: &[f32]| -> Vec<u8> {
        v.iter()
            // SAFETY: a plain conversion.
            .flat_map(|&x| unsafe { ffi::ggml_fp32_to_fp16(x) }.to_le_bytes())
            .collect()
    };
    let bf16_raw = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|&x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    // (k, m, h, n): a ragged tile edge in rows and tokens; per-head matrices
    for (k, m, h, n) in [(512i64, 600i64, 1i64, 700i64), (256, 136, 3, 300)] {
        let wv: Vec<f32> = noise((k * m * h) as usize, 51)
            .iter()
            .map(|v| v * 0.05)
            .collect();
        let xv = noise((k * h * n) as usize, 52);
        let ws = noise(m as usize, 53);
        let xne: Vec<i64> = if h == 1 { vec![k, n] } else { vec![k, h, n] };
        for (wty, wraw) in [(types::F16, f16_raw(&wv)), (types::BF16, bf16_raw(&wv))] {
            for xh in [false, true] {
                if xh && h > 1 {
                    continue;
                }
                let inputs = [
                    (&[k, m, h][..], Data::Raw(wty, wraw.clone())),
                    (&xne[..], Data::F32(xv.clone())),
                    (&[m][..], Data::F32(ws.clone())),
                ];
                let on = |gpu: &mut Backend, kernel: LinTc| {
                    run_on(gpu, &inputs, |ctx, t| {
                        let xh = xh.then(|| f16(ctx, t[1]));
                        linear_tc_kernel(ctx, t[0], t[1], xh, 0.75, Some(t[2]), kernel)
                    })
                };
                let want = on(&mut gpu, LinTc::Tc);
                let mut kernels: Vec<LinTc> = (0..6).map(LinTc::Gb).collect();
                kernels.push(LinTc::Auto);
                for kernel in kernels {
                    let got = on(&mut gpu, kernel);
                    assert!(
                        got.iter()
                            .zip(&want)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{kernel:?} k {k} m {m} h {h} n {n} {wty:?} xh {xh}"
                    );
                }
            }
        }
    }
    set_gb10_linear(false);
}

/// The CUDA K-quant products and embedding (Q4_K, Q5_K, Q6_K) against the CPU on ggml's
/// dequantization of the same weights (F32 products: ggml's own K-quant dot products round
/// the activations to Q8_K, further from exact than the GPU): the decode GEMVs (1-8 tokens,
/// 16-byte aligned activations or not), the plain GEMM and, with the GB10 kernels, the wide
/// one; one matrix and two.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_kquant_linear_and_embed_match_the_cpu() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let mut cpu = Backend::cpu().unwrap();
    let (k, m) = (2048usize, 37usize);
    for ty in [types::Q4_K, types::Q5_K, types::Q6_K] {
        let w = noise(k * m * 2, 50)
            .iter()
            .map(|v| v * 0.5)
            .collect::<Vec<_>>();
        let q = crate::ggml::quantize(ty, &w, k);
        let one = &q[..q.len() / 2];
        let wf = crate::ggml::dequantize(ty, &q, k * m * 2);
        let wf_one = &wf[..k * m];
        let name = format!("type {ty}");
        for gb10 in [false, true] {
            set_gb10_linear(gb10);
            for n in [1usize, 2, 3, 5, 8, 40, 300] {
                let x = noise(k * 2 * n, 51 + n as u64);
                // The GEMMs (9+ tokens) multiply F16 roundings of the weights and activations.
                let tol = if n <= 8 { 1e-4 } else { 3e-3 };
                let what = format!("{name} gb10 {gb10} n {n}");
                // One matrix.
                let x1 = (&[k as i64, n as i64][..], Data::F32(x[..k * n].to_vec()));
                let ne = [k as i64, m as i64];
                let (got, want) = both(
                    &mut gpu,
                    &mut cpu,
                    [
                        (&ne[..], Data::Raw(ty, one.to_vec())),
                        (&ne[..], Data::F32(wf_one.to_vec())),
                    ],
                    &[x1],
                    |ctx, t| linear(ctx, t[0], t[1], 0.5, None),
                );
                close_in(&what, &got, &want, tol);
                // Rows 4 bytes apart from 16 (the decode steps' generic GEMV).
                let xs = (
                    &[(k + 1) as i64, n as i64][..],
                    Data::F32(x[..(k + 1) * n].to_vec()),
                );
                let (got, want) = both(
                    &mut gpu,
                    &mut cpu,
                    [
                        (&ne[..], Data::Raw(ty, one.to_vec())),
                        (&ne[..], Data::F32(wf_one.to_vec())),
                    ],
                    &[xs],
                    |ctx, t| {
                        let xs = ctx.view_2d(t[1], [k as i64, n as i64], (k + 1) * 4, 0);
                        linear(ctx, t[0], xs, 1.0, None)
                    },
                );
                close_in(&format!("{what} strided"), &got, &want, tol);
                // Two matrices, a row of `x` per matrix and token.
                let ne2 = [k as i64, m as i64, 2];
                let (got, want) = both(
                    &mut gpu,
                    &mut cpu,
                    [
                        (&ne2[..], Data::Raw(ty, q.clone())),
                        (&ne2[..], Data::F32(wf.clone())),
                    ],
                    &[(&[k as i64, 2, n as i64][..], Data::F32(x.clone()))],
                    |ctx, t| linear(ctx, t[0], t[1], 1.0, None),
                );
                close_in(&format!("{what} two"), &got, &want, tol);
            }
        }
        set_gb10_linear(false);
        // The embedding: rows of `k` values, `m` of them, two streams.
        let ne = [k as i64, m as i64];
        let (got, want) = both(
            &mut gpu,
            &mut cpu,
            [
                (&ne[..], Data::Raw(ty, one.to_vec())),
                (&ne[..], Data::F32(wf_one.to_vec())),
            ],
            &[(&[5][..], Data::I32(vec![0, 36, 7, 7, 20]))],
            |ctx, t| embed(ctx, t[0], t[1], 2),
        );
        close_in(&format!("{name} embed"), &got, &want, 1e-6);
    }
}

/// `f` on `a` with the first of `w` and on `b` with the second, the rest of the inputs the same.
#[cfg(feature = "cuda")]
fn both<'s>(
    a: &mut Backend,
    b: &mut Backend,
    w: [(&'s [i64], Data); 2],
    rest: &[(&'s [i64], Data)],
    f: impl for<'a> Fn(&'a Context, &[Tensor<'a>]) -> Tensor<'a>,
) -> (Vec<f32>, Vec<f32>) {
    let [wa, wb] = w;
    let with = |w: (&'s [i64], Data)| -> Vec<(&'s [i64], Data)> {
        let mut v = vec![w];
        v.extend(rest.iter().map(|(ne, d)| (*ne, d.clone())));
        v
    };
    (run_on(a, &with(wa), &f), run_on(b, &with(wb), &f))
}

/// `a` within `tol` of `b` relative to `b`'s largest magnitude (at least 1): rounding errors
/// of a long dot product scale with its terms, not with the sum.
#[cfg(feature = "cuda")]
fn close_in(what: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{what}");
    let scale = b.iter().fold(1f32, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * scale,
            "{what} {i}: {x} vs {y} (scale {scale})"
        );
    }
}

/// linear_set's products and epilogues are linear's, then sigmoid's and kda_gate's, bit for bit.
#[test]
fn linear_set_is_linear_then_sigmoid_and_kda_gate() {
    let (k, m, mb, n, kb) = (64usize, 8usize, 4usize, 2usize, 32usize);
    let q8 = |rows: usize, cols: usize, seed| {
        Data::Raw(
            types::Q8_0,
            crate::ggml::quantize(types::Q8_0, &noise(rows * cols, seed), cols),
        )
    };
    let inputs = [
        (&[k as i64, n as i64][..], Data::F32(noise(k * n, 61))),
        (&[k as i64, kb as i64][..], q8(kb, k, 62)),
        (&[k as i64, mb as i64][..], q8(mb, k, 63)),
        (&[kb as i64, m as i64][..], q8(m, kb, 64)),
        (&[m as i64][..], Data::F32(noise(m, 65))),
        (&[2][..], Data::F32(vec![0.5, -1.5])),
    ];
    let (got, _) = run_all(&inputs, |ctx, t| {
        let a = linear_set(
            ctx,
            &[(t[1], t[0], SetEpi::Plain), (t[2], t[0], SetEpi::Sigmoid)],
        );
        let b = linear_set(ctx, &[(t[3], a[0], SetEpi::KdaGate(t[4], t[5], -5.0))]);
        vec![a[0], a[1], b[0]]
    });
    let (want, _) = run_all(&inputs, |ctx, t| {
        let a0 = linear(ctx, t[1], t[0], 1.0, None);
        let a1 = sigmoid(ctx, linear(ctx, t[2], t[0], 1.0, None));
        let b0 = kda_gate(ctx, linear(ctx, t[3], a0, 1.0, None), t[4], t[5], -5.0);
        vec![a0, a1, b0]
    });
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.len(), w.len());
        assert!(g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    assert!(linear_set_takes(&[], 8) && !linear_set_takes(&[], 9));
}

/// hc_pre_norm is hc_pre (the same y and weights, bit for bit) with rms_norm's output on y.
#[test]
fn hc_pre_norm_is_hc_pre_then_rms_norm() {
    // One node (3 tokens) and the two ops of a longer step (9).
    for n in [3usize, 9] {
        hc_pre_norm_matches(n);
    }
}

fn hc_pre_norm_matches(n: usize) {
    let (e, hc) = (64usize, 4usize);
    let nm = (2 + hc) * hc;
    let inputs = [
        (
            &[e as i64, hc as i64, n as i64][..],
            Data::F32(noise(e * hc * n, 51)),
        ),
        (
            &[(e * hc) as i64, nm as i64][..],
            Data::F32(noise(e * hc * nm, 52)),
        ),
        (&[3][..], Data::F32(vec![0.7, -0.4, 1.3])),
        (&[nm as i64][..], Data::F32(noise(nm, 53))),
        (&[e as i64][..], Data::F32(noise(e, 54))),
    ];
    let (got, _) = run_all(&inputs, |ctx, t| {
        let (p, h) = hc_pre_norm(ctx, t[0], t[1], t[2], t[3], 1e-6, 1e-6, 3, t[4], 1e-5);
        vec![p.y, p.pre, p.post, p.comb, h]
    });
    let (want, _) = run_all(&inputs, |ctx, t| {
        let p = hc_pre(ctx, t[0], t[1], t[2], t[3], 1e-6, 1e-6, 3);
        let h = rms_norm(ctx, p.y, Some(t[4]), 1e-5);
        vec![p.y, p.pre, p.post, p.comb, h]
    });
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.len(), w.len());
        assert!(g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    assert_eq!(got[4].len(), e * n);
}

/// On CUDA, hc_pre_norm's one launch gives hc_pre's and rms_norm's kernels' bits, at GLM's
/// size (4096 x 4 streams, F32 weights) for the decode steps' token counts. The other widths
/// take the other kernels (#831): hc_finish_kernel past 4096 elements, and below 4096 the
/// mix split's 16 loads ahead instead of 64.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_pre_norm_is_hc_pre_then_rms_norm() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let hc = 4usize;
    let nm = (2 + hc) * hc;
    for (e, n) in [4096usize, 5120, 1056]
        .into_iter()
        .flat_map(|e| [1usize, 2, 3, 4, 8, 9].map(|n| (e, n)))
    {
        let inputs = [
            (
                &[e as i64, hc as i64, n as i64][..],
                Data::F32(noise(e * hc * n, 71 + n as u64)),
            ),
            (
                &[(e * hc) as i64, nm as i64][..],
                Data::F32(noise(e * hc * nm, 72).iter().map(|v| v * 0.02).collect()),
            ),
            (&[3][..], Data::F32(vec![0.7, -0.4, 1.3])),
            (&[nm as i64][..], Data::F32(noise(nm, 73))),
            (&[e as i64][..], Data::F32(noise(e, 74))),
        ];
        let (got, _) = run_all_on(&mut gpu, &inputs, |ctx, t| {
            let (p, h) = hc_pre_norm(ctx, t[0], t[1], t[2], t[3], 1e-6, 1e-6, 20, t[4], 1e-5);
            vec![p.y, p.pre, p.post, p.comb, h]
        });
        let (want, _) = run_all_on(&mut gpu, &inputs, |ctx, t| {
            let p = hc_pre(ctx, t[0], t[1], t[2], t[3], 1e-6, 1e-6, 20);
            let h = rms_norm(ctx, p.y, Some(t[4]), 1e-5);
            vec![p.y, p.pre, p.post, p.comb, h]
        });
        let mut bad = Vec::new();
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.len(), w.len(), "e {e} n {n} output {k}");
            let diff: Vec<usize> = (0..g.len())
                .filter(|&i| g[i].to_bits() != w[i].to_bits())
                .collect();
            if let Some(&i) = diff.first() {
                bad.push(format!(
                    "e {e} n {n} output {k}: {} of {} differ, first {i}: {} vs {}",
                    diff.len(),
                    g.len(),
                    g[i],
                    w[i]
                ));
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }
}

/// On CUDA, hc_pre's tiled mix (prompt steps of 64+ tokens with the GB10 kernels, also on the
/// V100) gives hc_mix_kernel's bits at GLM's size (4096 x 4 streams, F32 weights), with a short
/// last block of tokens too (#1144).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_hc_pre_tiled_mix_is_hc_mix() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let (e, hc) = (4096usize, 4usize);
    let nm = (2 + hc) * hc;
    for n in [64usize, 130, 515] {
        let inputs = [
            (
                &[e as i64, hc as i64, n as i64][..],
                Data::F32(noise(e * hc * n, 91 + n as u64)),
            ),
            (
                &[(e * hc) as i64, nm as i64][..],
                Data::F32(noise(e * hc * nm, 92).iter().map(|v| v * 0.02).collect()),
            ),
            (&[3][..], Data::F32(vec![0.7, -0.4, 1.3])),
            (&[nm as i64][..], Data::F32(noise(nm, 93))),
        ];
        set_gb10_linear(true);
        let (got, _) = run_all_on(&mut gpu, &inputs, hc_pre_outs);
        set_gb10_linear(false);
        let (want, _) = run_all_on(&mut gpu, &inputs, hc_pre_outs);
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.len(), w.len(), "n {n} output {k}");
            let diff = (0..g.len())
                .filter(|&i| g[i].to_bits() != w[i].to_bits())
                .count();
            assert_eq!(
                diff,
                0,
                "n {n} output {k}: {diff} of {} values differ",
                g.len()
            );
        }
    }
}

/// hc_pre's four outputs, as cuda_hc_pre_tiled_mix_is_hc_mix compares them.
#[cfg(feature = "cuda")]
fn hc_pre_outs<'a>(ctx: &'a Context, t: &[Tensor<'a>]) -> Vec<Tensor<'a>> {
    let p = hc_pre(ctx, t[0], t[1], t[2], t[3], 1e-6, 1e-6, 20);
    vec![p.y, p.pre, p.post, p.comb]
}

/// argmax_wide's row is argmax's (the CPU runs the same search).
#[test]
fn argmax_wide_is_argmax() {
    let (v, n) = (300usize, 3usize);
    let mut x = noise(v * n, 81);
    x[v + 7] = 2.0;
    x[v + 250] = 2.0;
    let inputs = [(&[v as i64, n as i64][..], Data::F32(x))];
    let got = run(&inputs, |ctx, t| argmax_wide(ctx, t[0]));
    let want = run(&inputs, |ctx, t| argmax(ctx, t[0]));
    assert_eq!(got.len(), n);
    assert_eq!(got, want);
    assert_eq!(got[1], 7.0);
}

/// On CUDA, argmax_wide's chunked search finds argmax's index: the first of the largest, across
/// chunks too, in a column of -inf, at either end and in a vocabulary the chunks do not divide.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_argmax_wide_is_argmax() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for v in [154_880usize, 1001, 37] {
        for n in [1usize, 2, 3, 4] {
            let mut x: Vec<f32> = noise(v * n, 82 + v as u64 + n as u64)
                .iter()
                .map(|e| e * 20.0)
                .collect();
            let col = |t: usize| t * v..(t + 1) * v;
            // Column 0: a tie between two chunks, the first wins.
            x[v / 2 + 1] = 30.0;
            x[v - 2] = 30.0;
            if n > 1 {
                // The largest entry last.
                x[col(1).end - 1] = 31.0;
            }
            if n > 2 {
                x[col(2)].fill(f32::NEG_INFINITY);
            }
            if n > 3 {
                // The largest entry first, tied at the end.
                x[col(3).start] = 32.0;
                x[col(3).end - 1] = 32.0;
            }
            let inputs = [(&[v as i64, n as i64][..], Data::F32(x.clone()))];
            let got = run_on(&mut gpu, &inputs, |ctx, t| argmax_wide(ctx, t[0]));
            let old = run_on(&mut gpu, &inputs, |ctx, t| argmax(ctx, t[0]));
            let cpu = run(&inputs, |ctx, t| argmax(ctx, t[0]));
            assert_eq!(got, old, "v {v} n {n}");
            assert_eq!(got, cpu, "v {v} n {n}");
        }
    }
}

/// On CUDA, linear_mv_f32 (F32 weights, 1-8 tokens) gives linear_mv's bits: each lane's blocks
/// in order, a 16-term FMA chain a half block added to its sum, then the warp's butterfly sum,
/// emulated here; at the routers' size and with rows a whole round of blocks does not cover.
/// Up to 4096 rows take linear_mv_f32_split (a block a row, #831), the last case the warp a row.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_linear_f32_sums_like_linear_mv() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (k, m) in [
        (4096usize, 37usize),
        (4096, 288),
        (4096 + 32 * 7, 37),
        (32 * 3, 37),
        (32 * 3, 4100),
    ] {
        let w = noise(k * m, 90 + k as u64);
        for n in [1usize, 2, 3, 5, 8] {
            let x = noise(k * n, 91 + n as u64);
            let got = run_on(
                &mut gpu,
                &[
                    (&[k as i64, m as i64][..], Data::F32(w.clone())),
                    (&[k as i64, n as i64][..], Data::F32(x.clone())),
                ],
                |ctx, t| linear(ctx, t[0], t[1], 0.5, None),
            );
            let want: Vec<f32> = (0..n)
                .flat_map(|t| (0..m).map(move |r| (t, r)))
                .map(|(t, r)| {
                    let (w, x) = (&w[r * k..][..k], &x[t * k..][..k]);
                    let mut lanes = [0f32; 32];
                    for (lane, acc) in lanes.iter_mut().enumerate() {
                        for b in (lane..k / 32).step_by(32) {
                            for h in 0..2 {
                                let o = b * 32 + 16 * h;
                                let mut s = 0f32;
                                for j in 0..16 {
                                    s = w[o + j].mul_add(x[o + j], s);
                                }
                                *acc += s;
                            }
                        }
                    }
                    for o in [16, 8, 4, 2, 1] {
                        let prev = lanes;
                        for (l, a) in lanes.iter_mut().enumerate() {
                            *a = prev[l] + prev[l ^ o];
                        }
                    }
                    lanes[0] * 0.5
                })
                .collect();
            assert_eq!(got.len(), want.len());
            let diff: Vec<usize> = (0..got.len())
                .filter(|&i| got[i].to_bits() != want[i].to_bits())
                .collect();
            assert!(
                diff.is_empty(),
                "k {k} m {m} n {n}: {} of {} differ, first {}: {} vs {}",
                diff.len(),
                got.len(),
                diff[0],
                got[diff[0]],
                want[diff[0]]
            );
        }
    }
}

/// glu_mv's gate and up products and their GLU, on `backend`, against linear, linear and glu
/// on it: F32 outputs of both.
fn glu_mv_and_parts(
    backend: &mut Backend,
    k: usize,
    m: usize,
    n: usize,
    limit: f32,
) -> (Vec<f32>, Vec<f32>) {
    let q6 = |seed| {
        let w: Vec<f32> = noise(k * m, seed).iter().map(|v| v * 0.5).collect();
        Data::Raw(types::Q6_K, crate::ggml::quantize(types::Q6_K, &w, k))
    };
    let inputs = [
        (&[k as i64, m as i64][..], q6(101)),
        (&[k as i64, m as i64][..], q6(102)),
        (
            &[k as i64, n as i64][..],
            Data::F32(noise(k * n, 103 + n as u64)),
        ),
    ];
    let got = run_on(backend, &inputs, |ctx, t| {
        glu_mv(ctx, t[0], t[1], t[2], limit)
    });
    let want = run_on(backend, &inputs, |ctx, t| {
        let g = linear(ctx, t[0], t[2], 1.0, None);
        let u = linear(ctx, t[1], t[2], 1.0, None);
        glu(ctx, g, u, limit)
    });
    (got, want)
}

/// glu_mv is linear twice and glu, bit for bit (the CPU references), with and without GLM's
/// limit.
#[test]
fn glu_mv_is_linear_twice_then_glu() {
    let mut cpu = Backend::cpu().unwrap();
    for (n, limit) in [(1usize, 0.0f32), (3, 7.0), (8, 0.5)] {
        let (got, want) = glu_mv_and_parts(&mut cpu, 512, 9, n, limit);
        assert_eq!(got.len(), 9 * n);
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "n {n} limit {limit}"
        );
    }
}

/// On CUDA, glu_mv's one launch gives linear_mv_q6k's and glu_kernel's bits: the shared
/// expert's size (4096 x 2048) and an odd row count, the decode steps' token counts, with and
/// without the limit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_glu_mv_is_linear_twice_then_glu() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    set_gb10_linear(false);
    let mut bad = Vec::new();
    for (k, m) in [(4096usize, 2048usize), (512, 37)] {
        for n in [1usize, 2, 3, 4, 5, 8] {
            for limit in [0.0f32, 0.3] {
                let (got, want) = glu_mv_and_parts(&mut gpu, k, m, n, limit);
                assert_eq!(got.len(), m * n);
                let diff: Vec<usize> = (0..got.len())
                    .filter(|&i| got[i].to_bits() != want[i].to_bits())
                    .collect();
                if let Some(&i) = diff.first() {
                    bad.push(format!(
                        "k {k} m {m} n {n} limit {limit}: {} of {} differ, first {i}: {} vs {}",
                        diff.len(),
                        got.len(),
                        got[i],
                        want[i]
                    ));
                }
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// MLA attention nodes carry the prompt kernel as their sixth parameter (#862): the launcher's
/// choice for the model's [`mla_attn`], the one asked for by [`mla_attn_prompt`], which also
/// turns the GB10 kernels on and the tensor-core kernel on for `Tc` only; the CPU reference
/// ignores it.
#[test]
fn mla_nodes_carry_the_prompt_kernel() {
    let ctx = Context::new(16, 0);
    let q = ctx.new_tensor(types::F32, &[8, 2, 4]);
    let cache = ctx.new_tensor(types::F16, &[8, 64]);
    let table = ctx.new_tensor(types::I32, &[1, 1]);
    let sel = ctx.new_tensor(types::I32, &[3, 4]);
    let after = ctx.new_tensor(types::F32, &[1]);
    let srcs = [q, cache, table, sel, after];
    // SAFETY: the nodes live in `ctx`, alive here.
    let params = |t: Tensor<'_>| unsafe { crate::ggml::external_params(&*t.raw()) };
    assert_eq!(
        params(mla_attn(&ctx, q, cache, table, sel, 4, 0.5, after))[5],
        0
    );
    for (kernel, want) in [
        (MlaKernel::Auto, 0),
        (MlaKernel::Tiled, 1),
        (MlaKernel::Gb, 2),
        (MlaKernel::Tc, 3),
    ] {
        let p = params(mla_attn_prompt(&ctx, srcs, 4, 0.5, kernel));
        let tc = (kernel == MlaKernel::Tc) as i32;
        assert_eq!((p[3], p[4], p[5]), (1, tc, want), "{kernel:?}");
    }
}

/// On CUDA, mla_gb_kernel (the GB10's prompt kernel, taken here on any sm_80+) gives
/// mla_tiled_kernel's bits (#862), at selections of every length around the chunk of 32 and up
/// to the 2,051 a GLM row holds, empty ones too, through a permuted block table; both near the
/// CPU reference.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_mla_gb_matches_tiled() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let (d, h, n_tok, n_seqs) = (512usize, 32usize, 64usize, 2usize);
    let n = n_tok * n_seqs;
    let block = crate::model::core::KV_BLOCK;
    let (width, cap) = (40usize, 2051usize);
    let cells = width * n_seqs * block;
    let counts = [
        0usize, 1, 2, 31, 32, 33, 63, 64, 65, 96, 100, 2051, 2050, 1000,
    ];
    let pick = noise(n * cap, 91);
    let mut sel = Vec::with_capacity(n * (1 + cap));
    for t in 0..n {
        let c = counts[t % counts.len()];
        sel.push(c as i32);
        sel.extend((0..cap).map(|j| {
            if j < c {
                ((pick[t * cap + j].abs() * (width * block) as f32) as i32)
                    .min((width * block) as i32 - 1)
            } else {
                -1
            }
        }));
    }
    // Sequence s's blocks in reverse, after the other sequences'.
    let table: Vec<i32> = (0..n_seqs)
        .flat_map(|s| (0..width).rev().map(move |b| (s * width + b) as i32))
        .collect();
    let latents: Vec<f32> = noise(cells * d, 92).iter().map(|x| x * 2.0).collect();
    let inputs = [
        (
            &[d as i64, h as i64, n as i64][..],
            Data::F32(noise(d * h * n, 93)),
        ),
        (
            &[d as i64, cells as i64][..],
            Data::Raw(types::F16, super::bench::f16_bytes(&latents)),
        ),
        (&[width as i64, n_seqs as i64][..], Data::I32(table)),
        (&[1 + cap as i64, n as i64][..], Data::I32(sel)),
        (&[1][..], Data::F32(vec![0.0])),
    ];
    let scale = 1.0 / 192f32.sqrt();
    let on = |gpu: &mut Backend, kernel: MlaKernel| {
        run_on(gpu, &inputs, |ctx, t| {
            mla_attn_prompt(
                ctx,
                [t[0], t[1], t[2], t[3], t[4]],
                n_tok as i64,
                scale,
                kernel,
            )
        })
    };
    let tiled = on(&mut gpu, MlaKernel::Tiled);
    let gb = on(&mut gpu, MlaKernel::Gb);
    let cpu = run(&inputs, |ctx, t| {
        mla_attn_prompt(
            ctx,
            [t[0], t[1], t[2], t[3], t[4]],
            n_tok as i64,
            scale,
            MlaKernel::Auto,
        )
    });
    let diff: Vec<usize> = (0..gb.len())
        .filter(|&i| gb[i].to_bits() != tiled[i].to_bits())
        .collect();
    assert!(
        diff.is_empty(),
        "{} of {} differ, first at token {}: {} vs {}",
        diff.len(),
        gb.len(),
        diff[0] / (d * h),
        gb[diff[0]],
        tiled[diff[0]]
    );
    close(&tiled, &cpu, 1e-4);
}

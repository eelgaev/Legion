//! The DeepSeek-V4.1 ops' CPU references on small inputs against direct formulas; with
//! `--features cuda` (ignored tests, a GPU) the CUDA kernels against the same formulas and
//! against the CPU references at the real model's sizes.

use std::cell::Cell;

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

#[cfg(feature = "cuda")]
mod speed;

thread_local! {
    /// Whether [`run_all`] runs its graph on the GPU (the `cuda_*` tests).
    static ON_GPU: Cell<bool> = const { Cell::new(false) };
}

/// The backend [`run_all`] runs on: the CPU, or the first GPU while [`ON_GPU`] is set.
fn backend() -> Backend {
    #[cfg(feature = "cuda")]
    if ON_GPU.get() {
        use crate::ggml::{DeviceKind, device_by_kind};
        return device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU device")
            .init()
            .unwrap();
    }
    let mut b = Backend::cpu().unwrap();
    b.set_threads(3);
    b
}

/// The relative tolerance of [`close`] on the GPU at least: its sums run in another order and
/// its `expf` / `log1pf` are the CUDA library's.
const GPU_TOL: f32 = 2e-5;

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

enum Data {
    F32(Vec<f32>),
    F16(Vec<f32>),
    I32(Vec<i32>),
}

/// Builds `f` over input tensors of the given shapes and values, runs it on the CPU and
/// returns its output (I32 converted) and every input's values afterwards (F16 as F32).
fn run_all(
    inputs: &[(&[i64], Data)],
    f: impl for<'a> Fn(&'a Context, &[Tensor<'a>]) -> Tensor<'a>,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut backend = backend();
    let ctx = Context::new(32, 16);
    let ts: Vec<Tensor<'_>> = inputs
        .iter()
        .map(|(ne, d)| {
            let ty = match d {
                Data::F32(_) => types::F32,
                Data::F16(_) => types::F16,
                Data::I32(_) => types::I32,
            };
            ctx.new_tensor(ty, ne).as_input()
        })
        .collect();
    let out = f(&ctx, &ts).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(out);
    // Inputs an op leaves out get memory too.
    for &t in &ts {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    for (t, (_, d)) in ts.iter().zip(inputs) {
        match d {
            Data::F32(v) => t.set_f32(v),
            Data::I32(v) => t.set_i32(v),
            Data::F16(v) => {
                let b: Vec<u8> = v
                    .iter()
                    .flat_map(|x| f32_to_f16(*x).to_ne_bytes())
                    .collect();
                t.set_bytes(&b);
            }
        }
    }
    backend.compute(&graph).unwrap();
    let res = if out.ty() == types::I32 {
        out.to_i32().into_iter().map(|x| x as f32).collect()
    } else {
        out.to_f32()
    };
    let after = ts
        .iter()
        .map(|t| match t.ty() {
            types::F32 => t.to_f32(),
            types::I32 => t.to_i32().into_iter().map(|x| x as f32).collect(),
            _ => t
                .bytes()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16_to_f32(u16::from_ne_bytes(*b)))
                .collect(),
        })
        .collect();
    (res, after)
}

fn run(
    inputs: &[(&[i64], Data)],
    f: impl for<'a> Fn(&'a Context, &[Tensor<'a>]) -> Tensor<'a>,
) -> Vec<f32> {
    run_all(inputs, f).0
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    let tol = if ON_GPU.get() { tol.max(GPU_TOL) } else { tol };
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{i}: {x} vs {y}");
    }
}

/// Values as the F16 cache keeps them.
fn f16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f16_to_f32(f32_to_f16(x))).collect()
}

/// A block table of one sequence whose blocks are cells in reverse order (`n_blocks` of them),
/// and the cell of position `p`.
fn table(n_blocks: usize) -> Vec<i32> {
    (0..n_blocks as i32).rev().collect()
}

fn cell_of(table: &[i32], p: usize) -> usize {
    let b = crate::model::core::KV_BLOCK;
    table[p / b] as usize * b + p % b
}

#[test]
fn rope_rotates_the_tail_pairs_and_inverts() {
    let (d, h, n, n_rot) = (8usize, 2usize, 3usize, 4i64);
    let x = noise(d * h * n, 1);
    let pos = vec![0, 5, 9];
    for r in [
        Rope::plain(n_rot, 10_000.0),
        Rope::yarn(n_rot, 160_000.0, 4.0, 32.0, 32.0, 1.0),
    ] {
        let shift = -1;
        let got = run(
            &[
                (&[d as i64, h as i64, n as i64], Data::F32(x.clone())),
                (&[n as i64], Data::I32(pos.clone())),
            ],
            |ctx, t| rope(ctx, t[0], t[1], &r, shift, false),
        );
        let mut want = x.clone();
        for (t, &pt) in pos.iter().enumerate() {
            let p = (pt + shift) as f32;
            for hd in 0..h {
                let base = (t * h + hd) * d + d - n_rot as usize;
                for i in 0..n_rot as usize / 2 {
                    let (s, c) = (p * r.freq(i)).sin_cos();
                    let (a, b) = (x[base + 2 * i], x[base + 2 * i + 1]);
                    want[base + 2 * i] = a * c - b * s;
                    want[base + 2 * i + 1] = a * s + b * c;
                }
            }
        }
        close(&got, &want, 1e-6);
        let back = run(
            &[
                (&[d as i64, h as i64, n as i64], Data::F32(got)),
                (&[n as i64], Data::I32(pos.clone())),
            ],
            |ctx, t| rope(ctx, t[0], t[1], &r, shift, true),
        );
        close(&back, &x, 1e-5);
    }
}

#[test]
fn yarn_divides_the_slow_dimensions() {
    let r = Rope::yarn(64, 160_000.0, 16.0, 65_536.0, 32.0, 1.0);
    let plain = Rope::plain(64, 160_000.0);
    // Below the ramp: as RoPE; past it: divided by the factor.
    assert_eq!(r.freq(0), plain.freq(0));
    assert!((r.freq(31) - plain.freq(31) / 16.0).abs() <= 1e-6 * plain.freq(31));
    assert!(r.low < r.high);
}

#[test]
fn collapse_weighs_the_streams() {
    let (e, hc, n) = (5usize, 3usize, 4usize);
    let res = noise(e * hc * n, 2);
    let pre = noise(hc * n, 3);
    let rows = vec![3, 1];
    let got = run(
        &[
            (&[e as i64, hc as i64, n as i64], Data::F32(res.clone())),
            (&[hc as i64, n as i64], Data::F32(pre.clone())),
            (&[2], Data::I32(rows.clone())),
        ],
        |ctx, t| collapse(ctx, t[0], Some(t[1]), Some(t[2])),
    );
    let mut want = Vec::new();
    for &t in &rows {
        let t = t as usize;
        for i in 0..e {
            want.push(
                (0..hc)
                    .map(|s| pre[t * hc + s] * res[(t * hc + s) * e + i])
                    .sum(),
            );
        }
    }
    close(&got, &want, 1e-6);
    let first = run(
        &[(&[e as i64, hc as i64, n as i64], Data::F32(res.clone()))],
        |ctx, t| collapse(ctx, t[0], None, None),
    );
    let want: Vec<f32> = (0..n)
        .flat_map(|t| res[t * hc * e..t * hc * e + e].to_vec())
        .collect();
    assert_eq!(first, want);
}

#[test]
fn engram_gates_the_value_into_each_stream() {
    let (e, hc, n) = (8usize, 2usize, 3usize);
    let eps = 1e-6;
    let res = noise(e * hc * n, 4);
    let kv = noise((hc + 1) * e * n, 5);
    let q = noise(e * hc, 6);
    let k = noise(e * hc, 7);
    let got = run(
        &[
            (&[e as i64, hc as i64, n as i64], Data::F32(res.clone())),
            (&[((hc + 1) * e) as i64, n as i64], Data::F32(kv.clone())),
            (&[e as i64, hc as i64], Data::F32(q.clone())),
            (&[e as i64, hc as i64], Data::F32(k.clone())),
        ],
        |ctx, t| engram(ctx, t[0], t[1], t[2], t[3], eps),
    );
    let rms = |v: &[f32]| {
        let ms = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64;
        v.iter()
            .map(|x| (*x as f64 / (ms + eps as f64).sqrt()) as f32)
            .collect::<Vec<_>>()
    };
    let mut want = Vec::new();
    for t in 0..n {
        let row = &kv[t * (hc + 1) * e..(t + 1) * (hc + 1) * e];
        for h in 0..hc {
            let x = &res[(t * hc + h) * e..(t * hc + h + 1) * e];
            let (xn, kn) = (rms(x), rms(&row[h * e..(h + 1) * e]));
            let s: f32 = (0..e)
                .map(|i| xn[i] * q[h * e + i] * kn[i] * k[h * e + i])
                .sum::<f32>()
                / (e as f32).sqrt();
            let g = s.abs().max(1e-6).sqrt() * s.signum();
            let gate = 1.0 / (1.0 + (-g).exp());
            want.extend((0..e).map(|i| x[i] + gate * row[hc * e + i]));
        }
    }
    close(&got, &want, 1e-5);
}

#[test]
fn cache_write_keeps_group_ends_only() {
    let (w, row, cells, n) = (3usize, 8usize, 128usize, 5usize);
    let x = noise(w * n, 8);
    let pos: Vec<i32> = (10..15).collect();
    let tab = table(2);
    let cl: Vec<i32> = pos
        .iter()
        .map(|&p| cell_of(&tab, p as usize) as i32)
        .collect();
    let (_, after) = run_all(
        &[
            (
                &[row as i64, cells as i64],
                Data::F16(vec![0.0; row * cells]),
            ),
            (&[w as i64, n as i64], Data::F32(x.clone())),
            (&[n as i64], Data::I32(cl.clone())),
            (&[n as i64], Data::I32(pos.clone())),
        ],
        |ctx, t| cache_write(ctx, t[0], t[1], t[2], t[3], 2, 2, &[]),
    );
    let cache = &after[0];
    let mut want = vec![0.0; row * cells];
    for (t, &p) in pos.iter().enumerate() {
        if (p + 1) % 2 == 0 {
            let c = cl[t] as usize;
            want[c * row + 2..c * row + 2 + w].copy_from_slice(&f16(&x[t * w..(t + 1) * w]));
        }
    }
    assert_eq!(cache, &want);
}

#[test]
fn pool_softmax_gates_each_channel() {
    let (d, cells, ratio) = (4usize, 128usize, 2usize);
    let v = f16(&noise(2 * d * cells, 9));
    let tab = table(2);
    let pos = vec![6, 7];
    let got = run(
        &[
            (&[2 * d as i64, cells as i64], Data::F16(v.clone())),
            (&[2, 1], Data::I32(tab.clone())),
            (&[2], Data::I32(pos.clone())),
            (&[1], Data::I32(vec![0])),
        ],
        |ctx, t| pool(ctx, t[0], t[1], t[2], 2, ratio, d as i64, t[3]),
    );
    let row = |p: usize| &v[cell_of(&tab, p) * 2 * d..][..2 * d];
    let mut want = vec![0.0; d * 2];
    for c in 0..d {
        let (a, b) = (row(6), row(7));
        let m = a[d + c].max(b[d + c]);
        let (ea, eb) = ((a[d + c] - m).exp(), (b[d + c] - m).exp());
        want[d + c] = (a[c] * ea + b[c] * eb) / (ea + eb);
    }
    close(&got, &want, 1e-6);
}

#[test]
fn index_scores_are_relu_dots_over_visible_groups() {
    let (di, ih, cells, ratio, koff) = (4usize, 2usize, 128usize, 2usize, 3usize);
    let row = koff + di;
    let kc = f16(&noise(row * cells, 10));
    let tab = table(2);
    let pos = vec![4, 9];
    let (ng, nc) = (2usize, 6usize);
    let iq = noise(di * ih * ng, 11);
    let w = noise(ih * ng, 12);
    let mask: Vec<f32> = (0..nc * ng)
        .map(|i| if i % 3 == 0 { f32::NEG_INFINITY } else { 0.5 })
        .collect();
    for masked in [false, true] {
        let got = run(
            &[
                (&[di as i64, ih as i64, ng as i64], Data::F32(iq.clone())),
                (&[ih as i64, ng as i64], Data::F32(w.clone())),
                (&[row as i64, cells as i64], Data::F16(kc.clone())),
                (&[2, 1], Data::I32(tab.clone())),
                (&[ng as i64], Data::I32(pos.clone())),
                (&[nc as i64, ng as i64], Data::F32(mask.clone())),
                (&[1], Data::I32(vec![0])),
            ],
            |ctx, t| {
                let m = masked.then_some(t[5]);
                index_scores(
                    ctx,
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    t[4],
                    m,
                    2,
                    ratio,
                    koff as i64,
                    nc as i64,
                    t[6],
                )
            },
        );
        let mut want = vec![f32::NEG_INFINITY; nc * ng];
        for t in 0..ng {
            for j in 0..(pos[t] as usize + 1) / ratio {
                let c = cell_of(&tab, j * ratio + ratio - 1);
                let key = &kc[c * row + koff..][..di];
                let mut s = 0.0;
                for h in 0..ih {
                    let q = &iq[(t * ih + h) * di..][..di];
                    let dot: f32 = q.iter().zip(key).map(|(a, b)| a * b).sum();
                    s += dot.max(0.0) * w[t * ih + h];
                }
                if masked {
                    s += mask[t * nc + j];
                }
                want[t * nc + j] = s;
            }
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                g == w || (g - w).abs() <= 1e-5 * (1.0 + w.abs()),
                "{masked} {i}: {g} vs {w}"
            );
        }
    }
}

#[test]
fn cand_mask_keeps_the_best_blocks_and_the_newest() {
    // One query at position 9 (ratio 1: 10 visible), blocks of 2, keep 2.
    let nc = 12;
    let mut score = vec![f32::NEG_INFINITY; nc];
    for (j, s) in score.iter_mut().enumerate().take(10) {
        *s = [0.1, 0.2, 5.0, 0.3, 0.0, 0.0, 4.0, 0.1, 0.0, 0.0][j];
    }
    let got = run(
        &[
            (&[nc as i64, 1], Data::F32(score)),
            (&[1], Data::I32(vec![9])),
        ],
        |ctx, t| cand_mask(ctx, t[0], t[1], 1, 2, 2),
    );
    // The newest block (8, 9) pinned, then the best other: block 1 (5.0).
    let kept: Vec<usize> = (0..nc).filter(|&j| got[j] == 0.0).collect();
    assert_eq!(kept, [2, 3, 8, 9]);
    assert!(got.iter().all(|&v| v == 0.0 || v == f32::NEG_INFINITY));
}

#[test]
fn topk_takes_the_best_finite_in_order() {
    let score = vec![
        0.5,
        f32::NEG_INFINITY,
        0.9,
        0.5,
        -1.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        2.0,
        f32::NEG_INFINITY,
    ];
    let got = run(&[(&[5, 2], Data::F32(score))], |ctx, t| topk(ctx, t[0], 3));
    // Row 0: 0.9, then the tie 0.5 / 0.5 to the lower position. Row 1: one finite score.
    assert_eq!(got, [0.0, 2.0, 3.0, 3.0, -1.0, -1.0]);
}

#[test]
fn attn_softmaxes_window_and_selected_with_a_sink() {
    let (d, nh, cells, window, ratio, coff) = (4usize, 2usize, 128usize, 3usize, 2usize, 4usize);
    let row = 2 * d;
    let kc = f16(&noise(row * cells, 13));
    let tab = table(2);
    let pos = vec![1, 8];
    let ng = pos.len();
    let q = noise(d * nh * ng, 14);
    let sinks = vec![0.3, -0.2];
    let sel = vec![0, 2, -1, 1, 3, -1];
    let scale = 0.7;
    let got = run(
        &[
            (&[d as i64, nh as i64, ng as i64], Data::F32(q.clone())),
            (&[row as i64, cells as i64], Data::F16(kc.clone())),
            (&[2, 1], Data::I32(tab.clone())),
            (&[ng as i64], Data::I32(pos.clone())),
            (&[nh as i64], Data::F32(sinks.clone())),
            (&[3, ng as i64], Data::I32(sel.clone())),
        ],
        |ctx, t| {
            let comp = Compressed {
                kcache: t[1],
                sel: t[5],
                ratio,
                off: coff as i64,
            };
            attn(
                ctx,
                t[0],
                t[1],
                t[2],
                None,
                t[3],
                t[4],
                Some(comp),
                2,
                scale,
                window,
                &[],
            )
        },
    );
    let mut want = Vec::new();
    for t in 0..ng {
        let p = pos[t] as usize;
        let mut keys: Vec<&[f32]> = (p.saturating_sub(window - 1)..=p)
            .map(|w| &kc[cell_of(&tab, w) * row..][..d])
            .collect();
        for &j in &sel[t * 3..t * 3 + 3] {
            if j >= 0 {
                let c = cell_of(&tab, j as usize * ratio + ratio - 1);
                keys.push(&kc[c * row + coff..][..d]);
            }
        }
        for h in 0..nh {
            let qh = &q[(t * nh + h) * d..][..d];
            let s: Vec<f64> = keys
                .iter()
                .map(|k| scale as f64 * qh.iter().zip(*k).map(|(a, b)| (a * b) as f64).sum::<f64>())
                .collect();
            let z: f64 = s.iter().map(|x| x.exp()).sum::<f64>() + (sinks[h] as f64).exp();
            for i in 0..d {
                let o: f64 = keys
                    .iter()
                    .zip(&s)
                    .map(|(k, x)| k[i] as f64 * x.exp() / z)
                    .sum();
                want.push(o as f32);
            }
        }
    }
    close(&got, &want, 1e-5);
}

/// A window read through a ring (a layer whose cache keeps its last positions only): the same
/// output bits as through the whole cache, the compressed keys still through the block table.
/// The queries' window wraps around the ring; every ring cell it does not read is noise.
#[test]
fn attn_reads_the_window_through_a_ring() {
    ring_reads_like_the_whole_cache(4, 2, 3, &[200, 201]);
}

/// [`attn_reads_the_window_through_a_ring`] for heads of `d`, `nh` heads, a window of
/// `window` and one sequence's queries at `pos` (consecutive), ratio-2 picks.
fn ring_reads_like_the_whole_cache(d: usize, nh: usize, window: usize, pos: &[i32]) {
    let kb = crate::model::core::KV_BLOCK;
    let (ratio, coff, ng) = (2usize, d, pos.len());
    let last = *pos.last().unwrap() as usize;
    let n_blocks = (last + 1).div_ceil(kb);
    let (row, cells) = (2 * d, n_blocks * kb);
    let kc = f16(&noise(row * cells, 16));
    let tab = table(n_blocks);
    // The fewest entries the op takes, blocks of a pool two blocks larger, in reverse order.
    let entries = (window + ng).div_ceil(kb) + 1;
    let ring_tab: Vec<i32> = (0..entries).map(|i| (entries + 1 - i) as i32).collect();
    let ring_cells = (entries + 2) * kb;
    let ring_cell = |p: usize| ring_tab[(p / kb) % entries] as usize * kb + p % kb;
    let mut kr = f16(&noise(d * ring_cells, 17));
    for w in (pos[0] as usize + 1).saturating_sub(window)..=last {
        let c = cell_of(&tab, w);
        kr[ring_cell(w) * d..][..d].copy_from_slice(&kc[c * row..][..d]);
    }
    let q = noise(d * nh * ng, 18);
    let sinks = noise(nh, 19);
    let sel: Vec<i32> = (0..3 * ng).map(|i| [3, 40, -1, 0, 99, 7][i % 6]).collect();
    let attend = |ring: bool| {
        let own = if ring {
            (&[d as i64, ring_cells as i64][..], Data::F16(kr.clone()))
        } else {
            (&[row as i64, cells as i64][..], Data::F16(kc.clone()))
        };
        run(
            &[
                (&[d as i64, nh as i64, ng as i64], Data::F32(q.clone())),
                own,
                (&[n_blocks as i64, 1], Data::I32(tab.clone())),
                (&[ng as i64], Data::I32(pos.to_vec())),
                (&[nh as i64], Data::F32(sinks.clone())),
                (&[3, ng as i64], Data::I32(sel.clone())),
                (&[row as i64, cells as i64], Data::F16(kc.clone())),
                (&[entries as i64, 1], Data::I32(ring_tab.clone())),
            ],
            |ctx, t| {
                let comp = Compressed {
                    kcache: t[6],
                    sel: t[5],
                    ratio,
                    off: coff as i64,
                };
                let ring = ring.then_some((t[7], entries));
                attn(
                    ctx,
                    t[0],
                    t[1],
                    t[2],
                    ring,
                    t[3],
                    t[4],
                    Some(comp),
                    ng as i64,
                    1.0 / (d as f32).sqrt(),
                    window,
                    &[],
                )
            },
        )
    };
    let whole = attend(false);
    let ringed = attend(true);
    assert!(whole.iter().all(|v| v.is_finite()));
    assert_eq!(
        whole.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ringed.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn router_picks_by_biased_score_and_weighs_unbiased() {
    let (ne, n, k) = (6usize, 2usize, 2usize);
    let logits = noise(ne * n, 15)
        .iter()
        .map(|x| 4.0 * x)
        .collect::<Vec<_>>();
    let bias = vec![0.0, 3.0, 0.0, 0.0, 0.0, 0.0];
    let (ln, bn) = ([ne as i64, n as i64], [ne as i64]);
    let inputs = [
        (&ln[..], Data::F32(logits.clone())),
        (&bn[..], Data::F32(bias.clone())),
    ];
    let ids = run(&inputs, |ctx, t| router_ids(ctx, t[0], t[1], k as i64));
    let sp = |x: f32| (x.exp().ln_1p()).sqrt();
    for t in 0..n {
        let l = &logits[t * ne..(t + 1) * ne];
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| (sp(l[b]) + bias[b]).total_cmp(&(sp(l[a]) + bias[a])));
        assert_eq!(ids[t * k], order[0] as f32);
        assert_eq!(ids[t * k + 1], order[1] as f32);
        // The bias decides: expert 1 is always in.
        assert!(ids[t * k..t * k + k].contains(&1.0));
    }
    let ids_i: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
    let w = run(
        &[
            (&[ne as i64, n as i64], Data::F32(logits.clone())),
            (&[k as i64, n as i64], Data::I32(ids_i.clone())),
        ],
        |ctx, t| router_w(ctx, t[0], t[1], true, 1.5),
    );
    for t in 0..n {
        let s: Vec<f32> = (0..k)
            .map(|i| sp(logits[t * ne + ids_i[t * k + i] as usize]))
            .collect();
        let sum: f32 = s.iter().sum();
        let want: Vec<f32> = s.iter().map(|x| x / sum * 1.5).collect();
        close(&w[t * k..(t + 1) * k], &want, 1e-6);
    }
}

/// Every op's formula test above with its graph on the GPU: the CUDA kernels against the same
/// formulas (to [`GPU_TOL`] where the CPU's are tighter).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_ops_match_the_formulas() {
    ON_GPU.set(true);
    rope_rotates_the_tail_pairs_and_inverts();
    collapse_weighs_the_streams();
    engram_gates_the_value_into_each_stream();
    cache_write_keeps_group_ends_only();
    pool_softmax_gates_each_channel();
    index_scores_are_relu_dots_over_visible_groups();
    cand_mask_keeps_the_best_blocks_and_the_newest();
    topk_takes_the_best_finite_in_order();
    attn_softmaxes_window_and_selected_with_a_sink();
    attn_reads_the_window_through_a_ring();
    router_picks_by_biased_score_and_weighs_unbiased();
    ON_GPU.set(false);
}

/// `f` over `inputs` on the CPU and then on the GPU: `[cpu, gpu]` outputs and inputs after.
#[cfg(feature = "cuda")]
fn cpu_and_gpu(
    inputs: &[(&[i64], Data)],
    f: impl for<'a> Fn(&'a Context, &[Tensor<'a>]) -> Tensor<'a>,
) -> [(Vec<f32>, Vec<Vec<f32>>); 2] {
    let cpu = run_all(inputs, &f);
    ON_GPU.set(true);
    let gpu = run_all(inputs, &f);
    ON_GPU.set(false);
    [cpu, gpu]
}

/// The real model's sizes (DeepSeek-V4.1-Flash): 64 heads of 512, YaRN on the last 64, the
/// indexer's 32 heads of 128, ratio-4 compression with 512 selected positions and blocks of 8,
/// 384 experts choosing 6, 5120-wide streams; three queries of one sequence at positions far
/// past the 128-position window.
#[cfg(feature = "cuda")]
mod real {
    pub const D: usize = 512;
    pub const NH: usize = 64;
    pub const ROT: i64 = 64;
    pub const IDX_D: usize = 128;
    pub const IDX_H: usize = 32;
    pub const RATIO: usize = 4;
    pub const TOP_K: usize = 512;
    pub const WINDOW: usize = 128;
    pub const E: usize = 5120;
    pub const HC: usize = 4;
    pub const N_EXPERT: usize = 384;
    pub const N_USED: usize = 6;
    pub const POS: [i32; 3] = [203, 511, 1279];
    /// Cache blocks of the sequence: past the last position.
    pub const BLOCKS: usize = 21;
}

/// The CUDA kernels against the CPU references at [`real`] sizes, with random inputs.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_ops_match_the_cpu_at_real_sizes() {
    use real::*;
    let kb = crate::model::core::KV_BLOCK;
    let cells = BLOCKS * kb;
    let ng = POS.len();
    let pos = POS.to_vec();
    let tab = table(BLOCKS);
    let tab_ne = [BLOCKS as i64, 1];
    let tol = 1e-4;

    // RoPE: YaRN as the compressed layers have it, positions up to a million.
    let r = Rope::yarn(ROT, 160_000.0, 16.0, 65_536.0, 32.0, 1.0);
    let x = noise(D * NH * ng, 20);
    for (p, inverse) in [(pos.clone(), false), (vec![5, 70_000, 1_000_000], true)] {
        let [c, g] = cpu_and_gpu(
            &[
                (&[D as i64, NH as i64, ng as i64], Data::F32(x.clone())),
                (&[ng as i64], Data::I32(p)),
            ],
            |ctx, t| rope(ctx, t[0], t[1], &r, -1, inverse),
        );
        close(&g.0, &c.0, tol);
    }

    // Collapse and Engram over 5120-wide streams.
    let res = noise(E * HC * ng, 21);
    let pre = noise(HC * ng, 22);
    let [c, g] = cpu_and_gpu(
        &[
            (&[E as i64, HC as i64, ng as i64], Data::F32(res.clone())),
            (&[HC as i64, ng as i64], Data::F32(pre)),
            (&[2], Data::I32(vec![2, 0])),
        ],
        |ctx, t| collapse(ctx, t[0], Some(t[1]), Some(t[2])),
    );
    close(&g.0, &c.0, tol);
    let [c, g] = cpu_and_gpu(
        &[
            (&[E as i64, HC as i64, ng as i64], Data::F32(res)),
            (
                &[((HC + 1) * E) as i64, ng as i64],
                Data::F32(noise((HC + 1) * E * ng, 23)),
            ),
            (&[E as i64, HC as i64], Data::F32(noise(E * HC, 24))),
            (&[E as i64, HC as i64], Data::F32(noise(E * HC, 25))),
        ],
        |ctx, t| engram(ctx, t[0], t[1], t[2], t[3], 1e-6),
    );
    close(&g.0, &c.0, tol);

    // A latent write at an offset (ratio 4: the group ends only), bit for bit.
    let cl: Vec<i32> = pos
        .iter()
        .map(|&p| cell_of(&tab, p as usize) as i32)
        .collect();
    let [c, g] = cpu_and_gpu(
        &[
            (
                &[2 * D as i64, cells as i64],
                Data::F16(vec![0.0; 2 * D * cells]),
            ),
            (&[D as i64, ng as i64], Data::F32(noise(D * ng, 26))),
            (&[ng as i64], Data::I32(cl)),
            (&[ng as i64], Data::I32(pos.clone())),
        ],
        |ctx, t| cache_write(ctx, t[0], t[1], t[2], t[3], D as i64, RATIO, &[]),
    );
    assert_eq!(g.1[0], c.1[0], "cache_write");

    // The compressor's pooling over raw | gate rows.
    let [c, g] = cpu_and_gpu(
        &[
            (
                &[2 * D as i64, cells as i64],
                Data::F16(noise(2 * D * cells, 27)),
            ),
            (&tab_ne, Data::I32(tab.clone())),
            (&[ng as i64], Data::I32(pos.clone())),
            (&[1], Data::I32(vec![0])),
        ],
        |ctx, t| pool(ctx, t[0], t[1], t[2], ng as i64, RATIO, D as i64, t[3]),
    );
    close(&g.0, &c.0, tol);

    // The indexer's scores, with a mask and without; then the candidate blocks and the top-k
    // of them, bit for bit.
    let (koff, row) = (64usize, 64 + IDX_D);
    let nc = cells / RATIO;
    let mask: Vec<f32> = noise(nc * ng, 28)
        .iter()
        .map(|&v| if v < -0.8 { f32::NEG_INFINITY } else { v })
        .collect();
    let mut scores = Vec::new();
    for masked in [false, true] {
        let [c, g] = cpu_and_gpu(
            &[
                (
                    &[IDX_D as i64, IDX_H as i64, ng as i64],
                    Data::F32(noise(IDX_D * IDX_H * ng, 29)),
                ),
                (&[IDX_H as i64, ng as i64], Data::F32(noise(IDX_H * ng, 30))),
                (
                    &[row as i64, cells as i64],
                    Data::F16(noise(row * cells, 31)),
                ),
                (&tab_ne, Data::I32(tab.clone())),
                (&[ng as i64], Data::I32(pos.clone())),
                (&[nc as i64, ng as i64], Data::F32(mask.clone())),
                (&[1], Data::I32(vec![0])),
            ],
            |ctx, t| {
                let m = masked.then_some(t[5]);
                index_scores(
                    ctx,
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    t[4],
                    m,
                    ng as i64,
                    RATIO,
                    koff as i64,
                    nc as i64,
                    t[6],
                )
            },
        );
        for (i, (a, b)) in g.0.iter().zip(&c.0).enumerate() {
            assert!(
                a == b || (a - b).abs() <= tol * (1.0 + b.abs()),
                "index_scores {masked} {i}: {a} vs {b}"
            );
        }
        scores = c.0;
    }
    let score_ne = [nc as i64, ng as i64];
    let score_in = || (&score_ne[..], Data::F32(scores.clone()));
    let [c, g] = cpu_and_gpu(
        &[score_in(), (&[ng as i64], Data::I32(pos.clone()))],
        |ctx, t| cand_mask(ctx, t[0], t[1], RATIO, 8, 4),
    );
    assert_eq!(g.0, c.0, "cand_mask");
    let [c, g] = cpu_and_gpu(&[score_in()], |ctx, t| topk(ctx, t[0], TOP_K));
    assert_eq!(g.0, c.0, "topk");

    // The sparse attention: the window, the selection above and the sinks.
    let sel: Vec<i32> = c.0.iter().map(|&v| v as i32).collect();
    let [c, g] = cpu_and_gpu(
        &[
            (
                &[D as i64, NH as i64, ng as i64],
                Data::F32(noise(D * NH * ng, 32)),
            ),
            (
                &[2 * D as i64, cells as i64],
                Data::F16(noise(2 * D * cells, 33)),
            ),
            (&tab_ne, Data::I32(tab.clone())),
            (&[ng as i64], Data::I32(pos.clone())),
            (&[NH as i64], Data::F32(noise(NH, 34))),
            (&[TOP_K as i64, ng as i64], Data::I32(sel)),
        ],
        |ctx, t| {
            let comp = Compressed {
                kcache: t[1],
                sel: t[5],
                ratio: RATIO,
                off: D as i64,
            };
            attn(
                ctx,
                t[0],
                t[1],
                t[2],
                None,
                t[3],
                t[4],
                Some(comp),
                ng as i64,
                1.0 / (D as f32).sqrt(),
                WINDOW,
                &[],
            )
        },
    );
    close(&g.0, &c.0, tol);
    // Through a ring, in each attention kernel: the scored one (3 queries), the tiled ones (an
    // odd head count; 8 queries).
    ON_GPU.set(true);
    let three: Vec<i32> = (1277..1280).collect();
    let eight: Vec<i32> = (1272..1280).collect();
    ring_reads_like_the_whole_cache(D, 4, WINDOW, &three);
    ring_reads_like_the_whole_cache(D, 3, WINDOW, &three);
    ring_reads_like_the_whole_cache(D, 8, WINDOW, &eight);
    ON_GPU.set(false);

    // The router: the same choices, the same weights to rounding.
    let n = 5usize;
    let logits: Vec<f32> = noise(N_EXPERT * n, 35).iter().map(|x| 4.0 * x).collect();
    let [c, g] = cpu_and_gpu(
        &[
            (&[N_EXPERT as i64, n as i64], Data::F32(logits.clone())),
            (&[N_EXPERT as i64], Data::F32(noise(N_EXPERT, 36))),
        ],
        |ctx, t| router_ids(ctx, t[0], t[1], N_USED as i64),
    );
    assert_eq!(g.0, c.0, "router_ids");
    let ids: Vec<i32> = c.0.iter().map(|&v| v as i32).collect();
    let [c, g] = cpu_and_gpu(
        &[
            (&[N_EXPERT as i64, n as i64], Data::F32(logits)),
            (&[N_USED as i64, n as i64], Data::I32(ids)),
        ],
        |ctx, t| router_w(ctx, t[0], t[1], true, 1.5),
    );
    close(&g.0, &c.0, tol);
}

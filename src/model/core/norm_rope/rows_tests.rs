//! qwen4exp's prompt-step rows in one launch each (#1183): [`make_rows_fold`] and
//! [`make_idx_append`], made by `node_fuse`'s rewrite after allocation, against ggml's chains
//! (the K `cont`, `rms_norm`, `mul`, `rope_multi` and `set_rows`; the V `cont` and `set_rows`;
//! the indexer query's `rms_norm`, `mul` and `rope_multi`; the indexer rows' `fill`, `concat`
//! and `set_rows`): on the CPU (the reference implementations, within rounding), on CUDA
//! (ignored tests) bit for bit, and where the rewrite leaves the chains.

use super::tests::{EPS, as_bytes, cells, chain_rope, fill_f16, noise, positions, rel, rope};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::node_fuse::rewrite_graph;

/// One attention layer's step: K / V heads of `d` (`kv` of them) read from a joined
/// projection's rows, RoPE of `n_dims`, an indexer of `hi` query heads of `di`, `n` tokens at
/// positions from `base`.
#[derive(Clone, Copy)]
struct Step {
    d: i64,
    kv: i64,
    di: i64,
    hi: i64,
    n: i64,
    n_dims: i32,
    imrope: bool,
    base: i32,
}

/// qwen4exp's shapes (K and V 2 x 256, RoPE of 64, indexer 4 x 128).
const QWEN4: Step = Step {
    d: 256,
    kv: 2,
    di: 128,
    hi: 4,
    n: 777,
    n_dims: 64,
    imrope: true,
    base: 0,
};

/// How a run makes the rows.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// ggml's chains.
    Chain,
    /// `node_fuse`'s rewrite.
    Rewrite,
    /// `node_fuse`'s rewrite with every chain's intermediate a graph output too (read
    /// elsewhere), so nothing may fold.
    Kept,
    /// The ops made by hand on the chains' last nodes (the chains run too, unread).
    Direct,
}

/// A run's K, V and indexer caches (F16 bytes), indexer queries (F32 bytes) and how many of
/// each the rewrite folded (K, V, query, indexer rows).
struct Out {
    k: Vec<u8>,
    v: Vec<u8>,
    idx: Vec<u8>,
    iq: Vec<u8>,
    folded: [usize; 4],
}

const CELLS: i64 = 1000;

fn run(s: &Step, backend: &mut Backend, mode: Mode) -> Out {
    let ctx = Context::new(128, 128);
    // Inputs, weights and caches are kept for the whole graph.
    let kept = |ty, ne: &[i64]| ctx.new_tensor(ty, ne).as_input().as_output();
    let (d, kv, n) = (s.d, s.kv, s.n);
    // The joined projection's rows: a query part, then K, then V, per token.
    let qw = 3 * d;
    let qkv = kept(types::F32, &[qw + 2 * d * kv, n]);
    let nb = [d as usize * 4, qkv.nb()[1]];
    let k = ctx.view_3d(qkv, [d, kv, n], nb, qw as usize * 4);
    let v = ctx.view_3d(qkv, [d, kv, n], nb, (qw + d * kv) as usize * 4);
    let w = kept(types::F32, &[d]);
    let pos = kept(types::I32, &[4 * n]);
    let cl = kept(types::I32, &[n]);
    let kc = kept(types::F16, &[d * kv, CELLS]);
    let vc = kept(types::F16, &[d * kv, CELLS]);
    let ic = kept(types::F16, &[2 * s.di, CELLS]);
    let r = rope(s.imrope, s.n_dims);
    let kcont = ctx.cont(k);
    let kmul = ctx.mul(ctx.rms_norm(kcont, EPS), w);
    let kr = chain_rope(&ctx, kmul, pos, &r).named(K_ROPE);
    let vr = ctx.cont(v).named(V_CONT);
    let kset = ctx.set_rows(kc, ctx.reshape_2d(kr, d * kv, n), cl);
    let vset = ctx.set_rows(vc, ctx.reshape_2d(vr, d * kv, n), cl);
    // The indexer: its raw keys and a zero pool-key half into its cache, its query normed and
    // rotated.
    let kraw = kept(types::F32, &[s.di, n]);
    let zeros = ctx.fill(ctx.new_tensor(types::F32, &[s.di, n]), 0.0);
    let rows = ctx.concat(kraw, zeros, 0).named(IDX_ROWS);
    let iset = ctx.set_rows(ic, rows, cl);
    let qi = kept(types::F32, &[s.di * s.hi, n]);
    let wi = kept(types::F32, &[s.di]);
    let imul = ctx.mul(ctx.rms_norm(ctx.reshape_3d(qi, s.di, s.hi, n), EPS), wi);
    // The indexer's RoPE, of at most its head's columns.
    let ri = rope(s.imrope, s.n_dims.min(s.di as i32));
    let iq = chain_rope(&ctx, imul, pos, &ri).named(IQ_ROPE).as_output();
    if mode == Mode::Kept {
        for t in [kr, vr, rows, imul] {
            t.as_output();
        }
    }
    let mut graph = ctx.new_graph(128);
    for t in [kset, vset, iset, iq] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let folded = match mode {
        Mode::Chain => [0; 4],
        Mode::Rewrite | Mode::Kept => {
            // SAFETY: an allocated graph that has not run.
            let c = unsafe { rewrite_graph(graph.raw(), true) };
            [c.k_rows, c.v_rows, c.iq_rope, c.idx_rows]
        }
        Mode::Direct => {
            // SAFETY: the chains' last nodes, allocated, of the shapes the ops take.
            unsafe {
                let norm = |w: Tensor<'_>, r: &Rope| RowsNorm {
                    w: w.raw(),
                    pos: pos.raw(),
                    eps: EPS,
                    n_dims: r.n_dims,
                    sections: r.sections,
                    imrope: s.imrope,
                    freq_base: r.freq_base,
                    freq_scale: r.freq_scale,
                    attn_factor: r.attn_factor,
                };
                let f = RowsFold {
                    x: k.raw(),
                    norm: Some(norm(w, &r)),
                    cache: Some((kc.raw(), cl.raw())),
                };
                make_rows_fold(kset.raw(), &f);
                let f = RowsFold {
                    x: v.raw(),
                    norm: None,
                    cache: Some((vc.raw(), cl.raw())),
                };
                make_rows_fold(vset.raw(), &f);
                let x = (*imul.raw()).src[0];
                let f = RowsFold {
                    x: (*x).src[0],
                    norm: Some(norm(wi, &ri)),
                    cache: None,
                };
                make_rows_fold(iq.raw(), &f);
                make_idx_append(iset.raw(), kraw.raw(), ic.raw(), cl.raw());
                assert!(is_rows_fold(kset.raw()) && is_idx_append(iset.raw()));
            }
            [1; 4]
        }
    };

    // Heads of different sizes, so a wrong norm shows.
    let rows: Vec<f32> = noise(qkv.elements(), 41)
        .iter()
        .enumerate()
        .map(|(i, &x)| x * (1.0 + (i as i64 / d % 5) as f32))
        .collect();
    qkv.set_f32(&rows);
    let ones = |n: i64, seed| -> Vec<f32> {
        noise(n as usize, seed)
            .iter()
            .map(|v| 1.0 + 0.5 * v)
            .collect()
    };
    w.set_f32(&ones(d, 42));
    wi.set_f32(&ones(s.di, 43));
    kraw.set_f32(&noise(kraw.elements(), 44));
    qi.set_f32(&noise(qi.elements(), 45));
    pos.set_bytes(as_bytes(&positions(n as usize, s.base)));
    cl.set_bytes(as_bytes(&cells(n, CELLS)));
    for (i, c) in [kc, vc, ic].into_iter().enumerate() {
        fill_f16(c, 50 + i as u64);
    }
    backend.compute(&graph).unwrap();
    Out {
        k: kc.bytes(),
        v: vc.bytes(),
        idx: ic.bytes(),
        iq: iq.bytes(),
        folded,
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|&x| f32::from_le_bytes(x))
        .collect()
}

/// Reads F16 bytes as F32.
fn halves(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|&x| unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes(x)) })
        .collect()
}

/// The CPU references (made by hand: the rewrite leaves host caches to ggml) against ggml's
/// chains, within rounding; the copies exact.
#[test]
fn cpu_rows_fold_matches_ggml() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for s in [
        Step { n: 70, ..QWEN4 },
        Step {
            d: 128,
            kv: 4,
            n: 66,
            n_dims: 128,
            imrope: false,
            base: 199_999,
            ..QWEN4
        },
    ] {
        let want = run(&s, &mut cpu, Mode::Chain);
        let got = run(&s, &mut cpu, Mode::Direct);
        let e = rel(&halves(&got.k), &halves(&want.k));
        assert!(e < 2e-3, "K {e:e}");
        assert_eq!(got.v, want.v);
        assert_eq!(got.idx, want.idx);
        let e = rel(&f32s(&got.iq), &f32s(&want.iq));
        assert!(e < 2e-3, "query {e:e}");
    }
}

/// On the CPU the rewrite folds no cache write (host caches keep ggml's nodes, as the direct
/// writes do), and the rows stay ggml's within rounding.
#[test]
fn cpu_rewrite_keeps_host_cache_writes() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let s = Step { n: 70, ..QWEN4 };
    let want = run(&s, &mut cpu, Mode::Chain);
    let got = run(&s, &mut cpu, Mode::Rewrite);
    let [k, v, _, idx] = got.folded;
    assert_eq!([k, v, idx], [0; 3]);
    assert_eq!((got.k, got.v, got.idx), (want.k, want.v, want.idx));
    let e = rel(&f32s(&got.iq), &f32s(&want.iq));
    assert!(e < 2e-3, "query {e:e}");
}

/// Kernel shapes: rows of 128 or 256 columns, a rotated half a whole number of warps.
#[test]
fn rows_fold_shapes() {
    for (d, r, ok) in [
        (256, Some(64), true),
        (256, Some(128), true),
        (256, Some(256), true),
        (256, Some(192), false),
        (256, Some(32), false),
        (128, Some(64), true),
        (128, Some(128), true),
        (128, Some(256), false),
        (128, None, true),
        (256, None, true),
        (64, None, false),
        (512, Some(64), false),
    ] {
        assert_eq!(rows_fold_fits(d, r), ok, "{d} columns, RoPE {r:?}");
    }
}

/// Whether `LEGION_DEBUG_ROWS_FOLD=0` keeps the chains.
#[cfg(feature = "cuda")]
fn fold_off() -> bool {
    std::env::var_os("LEGION_DEBUG_ROWS_FOLD").is_some_and(|v| v == "0")
}

/// On a GPU, the rewrite's ops against ggml-cuda's chains bit for bit: qwen4exp's shapes and
/// others (each kernel's rotated width), positions from 0 and from 199,999, IMROPE and MROPE;
/// prompt steps fold, decode steps and chains read elsewhere do not.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rows_fold_is_bit_identical_to_ggml() {
    let mut gpu = super::tests::gpu();
    let shapes = [
        QWEN4,
        Step {
            base: 199_999,
            imrope: false,
            n: 1031,
            ..QWEN4
        },
        Step {
            n_dims: 128,
            n: 65,
            ..QWEN4
        },
        Step {
            n_dims: 256,
            base: 77,
            ..QWEN4
        },
        Step {
            d: 128,
            kv: 4,
            n_dims: 128,
            di: 128,
            hi: 8,
            n: 300,
            ..QWEN4
        },
        Step {
            n_dims: 128,
            hi: 2,
            n: 300,
            ..QWEN4
        },
    ];
    for s in shapes {
        let want = run(&s, &mut gpu, Mode::Chain);
        let what = format!(
            "d {}, rope {}, {} tokens from {}, imrope {}",
            s.d, s.n_dims, s.n, s.base, s.imrope
        );
        for mode in [Mode::Rewrite, Mode::Kept, Mode::Direct] {
            let got = run(&s, &mut gpu, mode);
            let folded = match mode {
                Mode::Rewrite if !fold_off() => [1; 4],
                Mode::Direct => [1; 4],
                _ => [0; 4],
            };
            assert_eq!(got.folded, folded, "{what}: folded");
            assert!(got.k == want.k, "{what}: K cache");
            assert!(got.v == want.v, "{what}: V cache");
            assert!(got.idx == want.idx, "{what}: indexer cache");
            assert!(got.iq == want.iq, "{what}: indexer query");
        }
    }
    // A decode-sized step keeps the chains.
    let s = Step { n: 64, ..QWEN4 };
    let got = run(&s, &mut gpu, Mode::Rewrite);
    assert_eq!(got.folded, [0; 4], "a decode step folded");
    let want = run(&s, &mut gpu, Mode::Chain);
    assert!(got.k == want.k && got.v == want.v && got.idx == want.idx && got.iq == want.iq);
}

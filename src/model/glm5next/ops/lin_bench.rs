//! GLM's F16 prompt GEMMs timed at Maya-S's dense shapes (`legion lin-bench`, #862): the
//! products whose Q6_K weights [`wf16`] turns into F16 for linear_tc, on each [`LinTc`] kernel.

use super::*;
use crate::ggml::{Backend, BufferType, ComputeAllocator, Graph};

/// One dense product of Maya-S: `name`, `k` inputs, `m` rows, `h` matrices (per-head
/// up-projections), and whether its input comes as F16 rows (`xh`, shared by 2+ products).
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub name: &'static str,
    pub k: i64,
    pub m: i64,
    pub h: i64,
    pub xh: bool,
}

/// Maya-S's dense products that take linear_tc at prompt steps (the per-layer counts are in
/// the names' comments).
pub const SHAPES: [Shape; 12] = [
    // KDA layers (34): attn_q / attn_k / attn_v share x
    Shape {
        name: "kda.qkv",
        k: 4096,
        m: 8192,
        h: 1,
        xh: true,
    },
    // KDA attn_output
    Shape {
        name: "kda.o",
        k: 8192,
        m: 4096,
        h: 1,
        xh: false,
    },
    // MLA layers (11): attn_output, attn_q_a, attn_q_b, attn_kv_a_mqa
    Shape {
        name: "mla.o",
        k: 16384,
        m: 4096,
        h: 1,
        xh: false,
    },
    Shape {
        name: "mla.q_a",
        k: 4096,
        m: 1536,
        h: 1,
        xh: false,
    },
    Shape {
        name: "mla.q_b",
        k: 1536,
        m: 16384,
        h: 1,
        xh: false,
    },
    Shape {
        name: "mla.kv_a",
        k: 4096,
        m: 512,
        h: 1,
        xh: false,
    },
    // MLA per-head up-projections (64 heads)
    Shape {
        name: "mla.k_b",
        k: 256,
        m: 512,
        h: 64,
        xh: false,
    },
    Shape {
        name: "mla.v_b",
        k: 512,
        m: 256,
        h: 64,
        xh: false,
    },
    // shared experts (43 MoE layers): gate / up share x, then down
    Shape {
        name: "shexp.gu",
        k: 4096,
        m: 2048,
        h: 1,
        xh: true,
    },
    Shape {
        name: "shexp.down",
        k: 2048,
        m: 4096,
        h: 1,
        xh: false,
    },
    // the leading dense FFNs (3): gate / up share x
    Shape {
        name: "ffn.gu",
        k: 4096,
        m: 12288,
        h: 1,
        xh: true,
    },
    Shape {
        name: "ffn.down",
        k: 12288,
        m: 4096,
        h: 1,
        xh: false,
    },
];

/// Deterministic values in [-1, 1).
pub fn noise(n: usize, seed: u64) -> Vec<f32> {
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

/// `v` as F16 bytes (ggml's rounding, round to nearest even as the kernels' `__float2half_rn`).
pub fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut h = vec![0u16; v.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::ggml_fp32_to_fp16_row(v.as_ptr(), h.as_mut_ptr(), v.len() as i64) };
    h.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// FNV-1a over the bits of `v`.
pub fn bits_hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// One kernel's time in [`gemm_ms`].
pub struct Timing {
    pub kernel: LinTc,
    /// Milliseconds per call: the best round's mean over its `reps` calls.
    pub ms: f64,
    /// [`bits_hash`] of the output: equal for kernels that agree bit for bit.
    pub bits: u64,
}

/// Times `shape` at `n_tok` tokens on `backend` for each kernel of `kernels`: F16 weights of
/// Q6_K-like magnitude, F32 activations (and their F16 rows for an `xh` shape). `convert`: the
/// F16 rows of a one-matrix shape made by [`f16`] in each call's graph (timed with the product).
/// Each round runs every kernel in turn, one warm-up call and `reps` timed ones.
pub fn gemm_ms(
    backend: &mut Backend,
    shape: Shape,
    n_tok: usize,
    reps: usize,
    rounds: usize,
    kernels: &[LinTc],
    convert: bool,
) -> Vec<Timing> {
    let Shape { k, m, h, xh, .. } = shape;
    let convert = convert && !xh && h == 1;
    let n = n_tok as i64;
    let data = Context::new(8, 0);
    let w = data.new_tensor(types::F16, &[k, m, h]);
    let x = if h == 1 {
        data.new_tensor(types::F32, &[k, n])
    } else {
        data.new_tensor(types::F32, &[k, h, n])
    };
    let xf = xh.then(|| data.new_tensor(types::F16, &[k, n]));
    data.alloc_tensors(backend.buffer_type()).unwrap();
    let wv: Vec<f32> = noise(w.elements(), 7).iter().map(|v| v * 0.05).collect();
    w.set_bytes(&f16_bytes(&wv));
    let xv = noise(x.elements(), 8);
    x.set_f32(&xv);
    if let Some(xf) = xf {
        xf.set_bytes(&f16_bytes(&xv));
    }
    let ctxs: Vec<Context> = kernels.iter().map(|_| Context::new(4, 16)).collect();
    let outs: Vec<(Tensor<'_>, Graph<'_>)> = ctxs
        .iter()
        .zip(kernels)
        .map(|(ctx, &kernel)| {
            let xf = if convert { Some(f16(ctx, x)) } else { xf };
            let out = linear_tc_kernel(ctx, w, x, xf, 1.0, None, kernel).as_output();
            let mut graph = ctx.new_graph(16);
            graph.expand(out);
            ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                .unwrap();
            (out, graph)
        })
        .collect();
    let mut timings: Vec<Timing> = kernels
        .iter()
        .map(|&kernel| Timing {
            kernel,
            ms: f64::INFINITY,
            bits: 0,
        })
        .collect();
    for _ in 0..rounds.max(1) {
        for ((out, graph), timing) in outs.iter().zip(timings.iter_mut()) {
            backend.compute(graph).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                backend.compute(graph).unwrap();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64;
            timing.ms = timing.ms.min(ms);
            timing.bits = bits_hash(&out.to_f32());
        }
    }
    timings
}

/// One product of GLM's MLA F16 chain ([`chain_ms`], #862): its time as main runs it (`base`) and
/// with the F16 handoff (`chain`, one per kernel), and how many output values differ in bits.
pub struct ChainRow {
    pub name: &'static str,
    pub base_ms: f64,
    /// (kernel, ms per call, differing output values).
    pub chain: Vec<(LinTc, f64, usize)>,
}

/// Each graph's milliseconds per call: per round every graph in turn, one warm-up call and
/// `reps` timed ones; the best round counts.
fn time_graphs(
    backend: &mut Backend,
    graphs: &[&Graph<'_>],
    reps: usize,
    rounds: usize,
) -> Vec<f64> {
    let mut best = vec![f64::INFINITY; graphs.len()];
    for _ in 0..rounds.max(1) {
        for (graph, best) in graphs.iter().zip(best.iter_mut()) {
            backend.compute(graph).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                backend.compute(graph).unwrap();
            }
            *best = best.min(t0.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64);
        }
    }
    best
}

/// Values of two equally long byte runs of `size`-byte elements whose bits differ.
fn differing(a: &[u8], b: &[u8], size: usize) -> usize {
    assert_eq!(a.len(), b.len());
    a.chunks(size)
        .zip(b.chunks(size))
        .filter(|(x, y)| x != y)
        .count()
}

/// Maya-S's MLA prompt products at `n_tok` tokens on `backend`, as main runs them and with the
/// F16 handoff of [`super::f16_chain_takes`] (F16 weights as `wf16` gives them, random inputs):
/// - `q_b`: attn_q_b on F16 rows (`xh`), F32 outputs vs F16 ones;
/// - `k_b`: attn_k_b (64 heads) on F32 rows vs the same rows in F16 (on each of `kernels`);
/// - `v_b+cvt`: attn_v_b with F32 outputs and their [`f16`] copy (attn_output's `xh`) vs F16
///   outputs.
///
/// The F16 outputs must equal the F32 ones rounded by [`f16`], and attn_k_b's outputs those of
/// the F32 rows: the differing values are counted.
pub fn chain_ms(
    backend: &mut Backend,
    n_tok: usize,
    reps: usize,
    rounds: usize,
    kernels: &[LinTc],
) -> Vec<ChainRow> {
    let n = n_tok as i64;
    let (dq, dh, dl, nh) = (1536i64, 256i64, 512i64, 64i64);
    let data = Context::new(16, 0);
    let wqb = data.new_tensor(types::F16, &[dq, dh * nh]);
    let wkb = data.new_tensor(types::F16, &[dh, dl, nh]);
    let wvb = data.new_tensor(types::F16, &[dl, dh, nh]);
    let qr = data.new_tensor(types::F32, &[dq, n]);
    let qrh = data.new_tensor(types::F16, &[dq, n]);
    let q32 = data.new_tensor(types::F32, &[dh, nh, n]);
    let q16 = data.new_tensor(types::F16, &[dh, nh, n]);
    let o = data.new_tensor(types::F32, &[dl, nh, n]);
    data.alloc_tensors(backend.buffer_type()).unwrap();
    for (i, w) in [wqb, wkb, wvb].into_iter().enumerate() {
        let v: Vec<f32> = noise(w.elements(), 11 + i as u64)
            .iter()
            .map(|v| v * 0.05)
            .collect();
        w.set_bytes(&f16_bytes(&v));
    }
    let v = noise(qr.elements(), 21);
    qr.set_f32(&v);
    qrh.set_bytes(&f16_bytes(&v));
    let v: Vec<f32> = noise(q32.elements(), 22).iter().map(|v| v * 4.0).collect();
    q32.set_f32(&v);
    q16.set_bytes(&f16_bytes(&v));
    o.set_f32(&noise(o.elements(), 23));

    let bt = backend.buffer_type();
    fn graph_of<'a>(ctx: &'a Context, out: Tensor<'a>, bt: BufferType) -> Graph<'a> {
        let mut graph = ctx.new_graph(16);
        graph.expand(out);
        ctx.alloc_graph(&graph, ComputeAllocator::new(bt)).unwrap();
        graph
    }
    let mut rows = Vec::new();

    // attn_q_b
    let (cb, cc) = (Context::new(8, 16), Context::new(8, 16));
    let base = linear_io(&cb, wqb, qr, Some(qrh), 1.0, None, LinTc::Auto, false).as_output();
    let chain = linear_io(&cc, wqb, qr, Some(qrh), 1.0, None, LinTc::Auto, true).as_output();
    let (gb, gc) = (graph_of(&cb, base, bt), graph_of(&cc, chain, bt));
    let ms = time_graphs(backend, &[&gb, &gc], reps, rounds);
    let d = differing(&f16_bytes(&base.to_f32()), &chain.bytes(), 2);
    rows.push(ChainRow {
        name: "q_b",
        base_ms: ms[0],
        chain: vec![(LinTc::Auto, ms[1], d)],
    });

    // attn_k_b on each kernel
    let cb = Context::new(8, 16);
    let base = linear_io(&cb, wkb, q32, None, 1.0, None, LinTc::Auto, false).as_output();
    let gb = graph_of(&cb, base, bt);
    let ctxs: Vec<Context> = kernels.iter().map(|_| Context::new(8, 16)).collect();
    let outs: Vec<(Tensor<'_>, Graph<'_>)> = ctxs
        .iter()
        .zip(kernels)
        .map(|(ctx, &k)| {
            let out = linear_io(ctx, wkb, q16, None, 1.0, None, k, false).as_output();
            (out, graph_of(ctx, out, bt))
        })
        .collect();
    let mut graphs = vec![&gb];
    graphs.extend(outs.iter().map(|(_, g)| g));
    let ms = time_graphs(backend, &graphs, reps, rounds);
    let want = base.bytes();
    rows.push(ChainRow {
        name: "k_b",
        base_ms: ms[0],
        chain: kernels
            .iter()
            .zip(&outs)
            .zip(&ms[1..])
            .map(|((&k, (out, _)), &t)| (k, t, differing(&want, &out.bytes(), 4)))
            .collect(),
    });

    // attn_v_b, then attn_output's F16 copy of its rows
    let (cb, cc) = (Context::new(8, 16), Context::new(8, 16));
    let y = linear_io(&cb, wvb, o, None, 1.0, None, LinTc::Auto, false);
    let base = f16(&cb, cb.reshape_2d(y, dh * nh, n)).as_output();
    let chain = linear_io(&cc, wvb, o, None, 1.0, None, LinTc::Auto, true).as_output();
    let (gb, gc) = (graph_of(&cb, base, bt), graph_of(&cc, chain, bt));
    let ms = time_graphs(backend, &[&gb, &gc], reps, rounds);
    let d = differing(&base.bytes(), &chain.bytes(), 2);
    rows.push(ChainRow {
        name: "v_b+cvt",
        base_ms: ms[0],
        chain: vec![(LinTc::Auto, ms[1], d)],
    });
    rows
}

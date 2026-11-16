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

/// One BF16 decode product of GLM-5.3-Flash NVFP4 (`legion lin-bench --gemv`, #560 r6): `name`,
/// `k` inputs, `m` rows, `h` matrices. `tp.*` are a tensor-parallel rank's halves over two
/// machines, `one.*` the whole products of one machine.
#[derive(Clone, Copy, Debug)]
pub struct GemvShape {
    pub name: &'static str,
    pub k: i64,
    pub m: i64,
    pub h: i64,
}

const fn gs(name: &'static str, k: i64, m: i64, h: i64) -> GemvShape {
    GemvShape { name, k, m, h }
}

/// GLM-5.3-Flash NVFP4's BF16 matvecs (KDA q/k/v and o, MLA q_a/q_b/k_b/v_b/o, the shared
/// expert, the output head), as a rank of two and as one machine runs them.
pub const GEMV_SHAPES: [GemvShape; 15] = [
    gs("tp.kda.qkv", 4096, 4096, 1),
    gs("tp.kda.o", 4096, 4096, 1),
    gs("tp.mla.q_a", 4096, 1536, 1),
    gs("tp.mla.q_b", 1536, 8192, 1),
    gs("tp.mla.k_b", 256, 512, 32),
    gs("tp.mla.v_b", 512, 256, 32),
    gs("tp.mla.o", 8192, 4096, 1),
    gs("tp.shexp.gu", 4096, 1024, 1),
    gs("tp.shexp.down", 1024, 4096, 1),
    gs("head", 4096, 154880, 1),
    gs("one.kda.qkv", 4096, 8192, 1),
    gs("one.kda.o", 8192, 4096, 1),
    gs("one.mla.o", 16384, 4096, 1),
    gs("one.shexp.gu", 4096, 2048, 1),
    gs("one.shexp.down", 2048, 4096, 1),
];

/// One [`bf16_gemv`] kernel's time in [`gemv_ms`].
pub struct GemvTiming {
    pub kernel: u8,
    /// Milliseconds per product: the best round's mean.
    pub ms: f64,
    /// [`bits_hash`] of the output.
    pub bits: u64,
}

/// Times `shape` at `n_tok` tokens (1-16) on `backend` for each [`bf16_gemv`] kernel of
/// `kernels`: BF16 weights in [-0.05, 0.05), F32 activations in [-1, 1). Each graph holds copies
/// of the product on distinct copies of the weights (~384 MB of them in all, so that no copy is
/// still in L2 when it comes round again), all on the same `x`. Each round runs every kernel's
/// graph in turn, one warm-up call and `reps` timed ones.
pub fn gemv_ms(
    backend: &mut Backend,
    shape: GemvShape,
    n_tok: usize,
    reps: usize,
    rounds: usize,
    kernels: &[u8],
) -> Vec<GemvTiming> {
    let GemvShape { k, m, h, .. } = shape;
    let n = n_tok as i64;
    let wbytes = (k * m * h * 2) as usize;
    let copies = (384usize << 20).div_ceil(wbytes).clamp(1, 48);
    let data = Context::new(copies + 2, 0);
    let ws: Vec<Tensor<'_>> = (0..copies)
        .map(|_| data.new_tensor(types::BF16, &[k, m, h]))
        .collect();
    let x = if h == 1 {
        data.new_tensor(types::F32, &[k, n])
    } else {
        data.new_tensor(types::F32, &[k, h, n])
    };
    data.alloc_tensors(backend.buffer_type()).unwrap();
    let wb: Vec<u8> = noise(ws[0].elements(), 7)
        .iter()
        .flat_map(|v| to_bf16(v * 0.05).to_ne_bytes())
        .collect();
    for w in &ws {
        w.set_bytes(&wb);
    }
    drop(wb);
    x.set_f32(&noise(x.elements(), 8));
    let bt = backend.buffer_type();
    let ctxs: Vec<Context> = kernels
        .iter()
        .map(|_| Context::new(copies + 4, copies + 16))
        .collect();
    let graphs: Vec<(Tensor<'_>, Graph<'_>)> = ctxs
        .iter()
        .zip(kernels)
        .map(|(ctx, &kernel)| {
            set_bf16_gemv(kernel);
            let mut graph = ctx.new_graph(copies + 16);
            let mut first = None;
            for &w in &ws {
                let out = linear(ctx, w, x, 1.0, None).as_output();
                graph.expand(out);
                first.get_or_insert(out);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(bt)).unwrap();
            (first.unwrap(), graph)
        })
        .collect();
    let refs: Vec<&Graph<'_>> = graphs.iter().map(|(_, g)| g).collect();
    let ms = time_graphs(backend, &refs, reps, rounds);
    kernels
        .iter()
        .zip(&graphs)
        .zip(ms)
        .map(|((&kernel, (out, _)), ms)| GemvTiming {
            kernel,
            ms: ms / copies as f64,
            bits: bits_hash(&out.to_f32()),
        })
        .collect()
}

/// GLM-5.3-Flash Maya-S's Q6_K decode matvecs (`legion lin-bench --q6k`, #1172 r3), per-layer
/// counts in the comments.
pub const Q6K_SHAPES: [GemvShape; 13] = [
    // KDA layers (34): attn_q / attn_k / attn_v, attn_output (and nextn.eh_proj)
    gs("kda.qkv", 4096, 8192, 1),
    gs("kda.o", 8192, 4096, 1),
    // MLA layers (12)
    gs("mla.q_a", 4096, 1536, 1),
    gs("mla.kv_a", 4096, 512, 1),
    gs("mla.q_b", 1536, 16384, 1),
    gs("mla.k_b", 256, 512, 64),
    gs("mla.v_b", 512, 256, 64),
    gs("mla.o", 16384, 4096, 1),
    // shared experts (43): gate / up, down
    gs("shexp.gu", 4096, 2048, 1),
    gs("shexp.down", 2048, 4096, 1),
    // the leading dense FFNs (3)
    gs("ffn.gu", 4096, 12288, 1),
    gs("ffn.down", 12288, 4096, 1),
    gs("head", 4096, 154880, 1),
];

/// One [`super::q6k_mv`] mode's time in [`q6k_ms`].
pub struct Q6kTiming {
    pub mode: u8,
    /// Milliseconds per product: the best round's mean.
    pub ms: f64,
    /// Output values whose bits differ from the first mode's (every copy's output).
    pub differ: usize,
}

/// Valid random Q6_K rows: random quants and int8 scales, block scales in [2^-12, 2^-11).
fn q6k_noise(blocks: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut out = Vec::with_capacity(blocks * 210);
    for _ in 0..blocks {
        for _ in 0..208 / 8 {
            out.extend_from_slice(&next().to_ne_bytes());
        }
        let d = (1.0 + (next() >> 40) as f32 / (1u64 << 24) as f32) / 4096.0;
        out.extend_from_slice(&f16_bytes(&[d]));
    }
    out
}

/// Times Q6_K `shape` at `n_tok` tokens (1-8) on `backend` for each [`super::q6k_mv`] mode of
/// `modes` (random Q6_K weights, F32 activations in [-1, 1)), as [`gemv_ms`] does: each graph
/// holds the product on distinct copies of the weights (~384 MB in all), back to back as a
/// step's matvecs run. Counts the output values whose bits differ from the first mode's.
pub fn q6k_ms(
    backend: &mut Backend,
    shape: GemvShape,
    n_tok: usize,
    reps: usize,
    rounds: usize,
    modes: &[u8],
) -> Vec<Q6kTiming> {
    let GemvShape { k, m, h, .. } = shape;
    let n = n_tok as i64;
    let blocks = (k / 256 * m * h) as usize;
    let copies = (384usize << 20).div_ceil(blocks * 210).clamp(1, 48);
    let data = Context::new(copies + 2, 0);
    let ws: Vec<Tensor<'_>> = (0..copies)
        .map(|_| data.new_tensor(types::Q6_K, &[k, m, h]))
        .collect();
    let x = if h == 1 {
        data.new_tensor(types::F32, &[k, n])
    } else {
        data.new_tensor(types::F32, &[k, h, n])
    };
    data.alloc_tensors(backend.buffer_type()).unwrap();
    for (i, w) in ws.iter().enumerate() {
        w.set_bytes(&q6k_noise(blocks, 7 + i as u64));
    }
    x.set_f32(&noise(x.elements(), 8));
    let bt = backend.buffer_type();
    let ctxs: Vec<Context> = modes
        .iter()
        .map(|_| Context::new(copies + 4, copies + 16))
        .collect();
    let graphs: Vec<(Vec<Tensor<'_>>, Graph<'_>)> = ctxs
        .iter()
        .zip(modes)
        .map(|(ctx, &mode)| {
            set_q6k_mv(mode);
            let mut graph = ctx.new_graph(copies + 16);
            let outs: Vec<Tensor<'_>> = ws
                .iter()
                .map(|&w| {
                    let out = linear(ctx, w, x, 1.0, None).as_output();
                    graph.expand(out);
                    out
                })
                .collect();
            ctx.alloc_graph(&graph, ComputeAllocator::new(bt)).unwrap();
            (outs, graph)
        })
        .collect();
    let refs: Vec<&Graph<'_>> = graphs.iter().map(|(_, g)| g).collect();
    let ms = time_graphs(backend, &refs, reps, rounds);
    let first: Vec<Vec<u8>> = graphs[0].0.iter().map(|o| o.bytes()).collect();
    modes
        .iter()
        .zip(&graphs)
        .zip(ms)
        .map(|((&mode, (outs, _)), ms)| Q6kTiming {
            mode,
            ms: ms / copies as f64,
            differ: outs
                .iter()
                .zip(&first)
                .map(|(o, f)| differing(f, &o.bytes(), 4))
                .sum(),
        })
        .collect()
}

/// GLM-5.3-Flash Maya-S's small decode products (`legion lin-bench --small`, #1172 r4): the F32
/// router `ffn_gate_inp` (4096 x 288, 43 MoE layers), the KDA gates' second products `ssm_f_b`
/// and `ssm_g_b` in one [`linear_set`] (Q8_0 128 x 8192 each, the first with kda_gate's
/// epilogue; 34 KDA layers) and [`hc_pre_norm`] (F32 mix rows 16384 x 24, twice a layer).
pub const SMALL_SHAPES: [&str; 3] = ["router", "kda.fgb", "hc_pre"];

/// Weight bytes one [`SMALL_SHAPES`] product reads.
pub fn small_bytes(name: &str) -> f64 {
    match name {
        "router" => (4096 * 288 * 4) as f64,
        "kda.fgb" => (2 * 8192 * 4 * 34) as f64,
        _ => (16384 * 24 * 4) as f64,
    }
}

/// One [`super::small_mv`] mode's time in [`small_ms`].
pub struct SmallTiming {
    pub mode: u8,
    /// Milliseconds per product: the best round's mean.
    pub ms: f64,
    /// Output values whose bits differ from the first mode's (the last copy's outputs).
    pub differ: usize,
}

/// Valid random Q8_0 rows: random quants, block scales in [2^-8, 2^-7).
fn q8_noise(blocks: usize, seed: u64) -> Vec<u8> {
    let d = noise(blocks, seed ^ 0x55);
    let q = noise(blocks * 32, seed);
    let mut out = Vec::with_capacity(blocks * 34);
    for (d, q) in d.iter().zip(q.chunks(32)) {
        out.extend_from_slice(&f16_bytes(&[(1.5 + 0.5 * d) / 256.0]));
        out.extend(q.iter().map(|v| (v * 127.0) as i8 as u8));
    }
    out
}

/// Times the [`SMALL_SHAPES`] product `name` at `n_tok` tokens (1-8) on `backend` for each
/// [`super::small_mv`] mode of `modes`, as [`q6k_ms`] does (copies of the weights, ~384 MB in
/// all, back to back), with random weights and inputs. Counts the output values of the last
/// copy whose bits differ from the first mode's.
pub fn small_ms(
    backend: &mut Backend,
    name: &str,
    n_tok: usize,
    reps: usize,
    rounds: usize,
    modes: &[u8],
) -> Vec<SmallTiming> {
    let n = n_tok as i64;
    let copies = (384usize << 20)
        .div_ceil(small_bytes(name) as usize)
        .clamp(1, 48);
    let data = Context::new(2 * copies + 16, 0);
    // (weights per copy, the inputs and constants)
    let (ws, xs): (Vec<Vec<Tensor<'_>>>, Vec<Tensor<'_>>) = match name {
        "router" => (
            (0..copies)
                .map(|_| vec![data.new_tensor(types::F32, &[4096, 288])])
                .collect(),
            vec![data.new_tensor(types::F32, &[4096, n])],
        ),
        "kda.fgb" => (
            (0..copies)
                .map(|_| {
                    vec![
                        data.new_tensor(types::Q8_0, &[128, 8192]),
                        data.new_tensor(types::Q8_0, &[128, 8192]),
                    ]
                })
                .collect(),
            vec![
                data.new_tensor(types::F32, &[128, n]),
                data.new_tensor(types::F32, &[128, n]),
                data.new_tensor(types::F32, &[8192]),
                data.new_tensor(types::F32, &[64]),
            ],
        ),
        _ => (
            (0..copies)
                .map(|_| vec![data.new_tensor(types::F32, &[16384, 24])])
                .collect(),
            vec![
                data.new_tensor(types::F32, &[4096, 4, n]),
                data.new_tensor(types::F32, &[3]),
                data.new_tensor(types::F32, &[24]),
                data.new_tensor(types::F32, &[4096]),
            ],
        ),
    };
    data.alloc_tensors(backend.buffer_type()).unwrap();
    for (c, w) in ws.iter().enumerate() {
        for (j, w) in w.iter().enumerate() {
            let seed = 7 + (2 * c + j) as u64;
            if w.ty() == types::Q8_0 {
                w.set_bytes(&q8_noise(w.elements() / 32, seed));
            } else {
                let v: Vec<f32> = noise(w.elements(), seed).iter().map(|v| v * 0.05).collect();
                w.set_f32(&v);
            }
        }
    }
    for (i, x) in xs.iter().enumerate() {
        let v: Vec<f32> = noise(x.elements(), 31 + i as u64)
            .iter()
            .map(|v| {
                if name == "hc_pre" && i > 0 {
                    0.5 + 0.25 * v
                } else {
                    *v
                }
            })
            .collect();
        x.set_f32(&v);
    }
    let bt = backend.buffer_type();
    let ctxs: Vec<Context> = modes
        .iter()
        .map(|_| Context::new(8 * copies + 16, 8 * copies + 16))
        .collect();
    let graphs: Vec<(Vec<Tensor<'_>>, Graph<'_>)> = ctxs
        .iter()
        .zip(modes)
        .map(|(ctx, &mode)| {
            set_small_mv(mode);
            let mut graph = ctx.new_graph(8 * copies + 16);
            let mut last = Vec::new();
            for w in &ws {
                let outs: Vec<Tensor<'_>> = match name {
                    "router" => vec![linear(ctx, w[0], xs[0], 1.0, None)],
                    "kda.fgb" => linear_set(
                        ctx,
                        &[
                            (w[0], xs[0], SetEpi::KdaGate(xs[2], xs[3], -5.0)),
                            (w[1], xs[1], SetEpi::Plain),
                        ],
                    ),
                    _ => {
                        let (p, h) = hc_pre_norm(
                            ctx, xs[0], w[0], xs[1], xs[2], 1e-6, 1e-6, 20, xs[3], 1e-5,
                        );
                        vec![p.y, p.pre, p.post, p.comb, h]
                    }
                };
                last = outs.into_iter().map(|o| o.as_output()).collect();
                for &o in &last {
                    graph.expand(o);
                }
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(bt)).unwrap();
            (last, graph)
        })
        .collect();
    let refs: Vec<&Graph<'_>> = graphs.iter().map(|(_, g)| g).collect();
    let ms = time_graphs(backend, &refs, reps, rounds);
    let first: Vec<Vec<u8>> = graphs[0].0.iter().map(|o| o.bytes()).collect();
    modes
        .iter()
        .zip(&graphs)
        .zip(ms)
        .map(|((&mode, (outs, _)), ms)| SmallTiming {
            mode,
            ms: ms / copies as f64,
            differ: outs
                .iter()
                .zip(&first)
                .map(|(o, f)| differing(f, &o.bytes(), 4))
                .sum(),
        })
        .collect()
}

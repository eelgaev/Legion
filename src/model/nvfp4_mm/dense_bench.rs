//! Dense NVFP4 matmuls at small batches in the mv kernel's variants (`legion nvfp4-mv-bench
//! --dense`): a `[k, m]` weight times `n` tokens, each variant forced through [`force_kernel`],
//! its outputs checked against the default kernel's bits and timed (the weight bytes over the
//! time: the stream's bandwidth). For tuning [`pick_dense`] at speculative verify steps (a
//! drafter's block of tokens per sequence). Also the GEMM's weight prefetch runs at prompt
//! steps ([`force_gemm_pf`], #1176 r36), for plain matmuls and the fused SwiGLU (`glu`).

use super::{
    GemmTune, Kernel, dense_mv_max_n, force_gemm_pf, force_gemm_tune, force_kernel, mul_mat,
    pick_dense, swiglu, tokens_tile,
};
use crate::ggml::{Backend, ComputeAllocator, Context, types};

/// The dense GEMM's other shapes at 64-token tiles ([`GemmTune`]'s codes of `gemm_dense` in
/// `src/cuda/nvfp4_mm/common.cuh`: warps * 1000 + stages * 10 + CTAs per SM).
pub const GEMM_DENSE_TUNES: [i32; 8] = [31, 8032, 8041, 8022, 4031, 4032, 4023, 4042];

/// The streaming GEMM's variants at 64-token tiles ([`GemmTune::stream`] codes; `nvfp4_vstream`):
/// shapes, then the diagnostics (1 ...: no copies, 2 ...: no MMAs, 3 ...: no weight copies, 4 ...:
/// no activation copies; wrong outputs, timing only), then the default shape in CTAs of 4 / 6 /
/// 8 warps (10 million * warps above the code; the same bits).
pub const GEMM_STREAM_TUNES: [i32; 14] = [
    411_099, 311_099, 321_099, 412_099, 322_099, 1_411_099, 2_411_099, 3_411_099, 4_411_099,
    1_412_099, 3_412_099, 40_411_099, 60_411_099, 80_411_099,
];

/// The streaming GEMM's variants of the fused GLU, then its default in CTAs of 2-6 warps.
pub const GEMM_STREAM_GLU_TUNES: [i32; 8] = [
    412_099, 312_099, 322_099, 20_412_099, 30_412_099, 40_412_099, 50_412_099, 60_412_099,
];

/// One measured variant.
#[derive(Clone, Debug)]
pub struct DenseRow {
    pub k: usize,
    pub m: usize,
    pub n: usize,
    /// The kernel (the default's: what [`pick_dense`] picks, `sms` and `mv_max_n` as given).
    pub kernel: Kernel,
    /// The default (unforced) run.
    pub default: bool,
    /// The default kernel's weight prefetch forced to runs of `pf` blocks; 0: its own.
    pub pf: i32,
    /// The GEMM forced to this tune ([`GEMM_DENSE_TUNES`]); `None`: its own.
    pub tune: Option<GemmTune>,
    /// The fused SwiGLU (`[k, m]` gate and up weights) instead of a plain matmul.
    pub glu: bool,
    /// Best seconds per matmul; `None`: the variant does not launch.
    pub secs: Option<f64>,
    pub bytes: f64,
    /// Outputs that differ from the default kernel's bits.
    pub differ: usize,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Random NVFP4 blocks of a `[k, m]` weight (36 bytes per 64 values: 4 sub-block scales of
/// moderate size, then 32 bytes of codes).
fn weights(k: usize, m: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut out = Vec::with_capacity(m * k / 64 * 36);
    for _ in 0..m * k / 64 {
        for _ in 0..4 {
            let r = rng.next();
            out.push(((3 + (r >> 8) % 6) << 3 | (r >> 16) & 7) as u8);
        }
        for _ in 0..4 {
            out.extend(rng.next().to_le_bytes());
        }
    }
    out
}

/// The variants tried at `n` tokens: every mv tile shape that holds the tokens in one or two
/// token tiles (16 or 32 rows, 4 or 8 warps, 1 to 8 row tiles per CTA) and the GEMM.
pub fn variants(n: usize) -> Vec<Kernel> {
    let mut out = Vec::new();
    for nt in [1, 2, 4, 8] {
        let tiles = n.div_ceil(8 * nt as usize);
        if tiles > 2 || (tiles == 2 && nt == 8) {
            continue;
        }
        for (rtw, kw, rpt) in [
            (1, 8, 1),
            (2, 8, 1),
            (1, 4, 1),
            (1, 8, 2),
            (1, 8, 4),
            (1, 8, 8),
        ] {
            out.push(Kernel::Mv { nt, rtw, kw, rpt });
        }
    }
    for nt in [1, 2, 4] {
        if 8 * nt as usize >= n {
            out.push(Kernel::Gemm { mi: 1, nt });
        }
    }
    out
}

/// Times the default and each of [`variants`] on `[k, m]` weights times each of `toks`:
/// `layers` weights a graph (`copies` distinct ones, so the graph streams past the L2),
/// `rounds` timed runs of `reps` graphs, the best counts. `sms`: the GPU's SMs (for the
/// default's pick). Then the default with each of `pfs`' prefetch runs where it runs the
/// GEMM. `glu`: the fused SwiGLU of two such weights (layer `l`'s gate and up: weights `l`
/// and `l + 1`), the default and the prefetch runs only.
#[allow(clippy::too_many_arguments)]
pub fn run(
    gpu: &mut Backend,
    shapes: &[(usize, usize)],
    toks: &[usize],
    layers: usize,
    rounds: usize,
    reps: usize,
    sms: i64,
    mv_max_n: i64,
    pfs: &[i32],
    glu: bool,
) -> Result<Vec<DenseRow>, String> {
    let mut rows = Vec::new();
    for &(k, m) in shapes {
        let weights_ctx = Context::new(layers, 0);
        let ws: Vec<_> = (0..layers)
            .map(|_| weights_ctx.new_tensor(types::NVFP4, &[k as i64, m as i64]))
            .collect();
        weights_ctx
            .alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(k, m, 5 + i as u64));
        }
        let bytes = ws[0].nbytes() as f64 * if glu { 2.0 } else { 1.0 };
        for &n in toks {
            let ctx = Context::new(4 * layers + 4, 4 * layers + 4);
            let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let mut graph = ctx.new_graph(4 * layers + 4);
            let outs: Vec<_> = (0..layers)
                .map(|l| {
                    let y = if glu {
                        swiglu(&ctx, ws[l], ws[(l + 1) % layers], x, None, None)
                    } else {
                        mul_mat(&ctx, ws[l], x, None)
                    }
                    .as_output();
                    graph.expand(y);
                    y
                })
                .collect();
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            let mut rng = Rng(0x1234_5678 + n as u64);
            let xs: Vec<f32> = (0..k * n)
                .map(|_| ((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0)
                .collect();
            x.set_f32(&xs);
            // A warm-up run: on some shapes the graph's first run gave other bits than every
            // later one, whatever the kernel (a DGX Spark (GB10); not looked into).
            gpu.compute(&graph).map_err(|e| e.to_string())?;
            // The default twice: the second's differ counts are run-to-run changes.
            let default_kernel = if !glu {
                pick_dense(k as i64, m as i64, n as i64, sms, mv_max_n)
            } else if n as i64 <= dense_mv_max_n() {
                Kernel::Mv {
                    nt: tokens_tile(n as i64, 8),
                    rtw: 1,
                    kw: 8,
                    rpt: 1,
                }
            } else {
                Kernel::Gemm {
                    mi: 2,
                    nt: tokens_tile(n as i64, 16).max(4),
                }
            };
            let mut kernels: Vec<(Option<Kernel>, i32, Option<GemmTune>)> =
                vec![(None, 0, None), (None, 0, None)];
            if !glu {
                kernels.extend(variants(n).into_iter().map(|k| (Some(k), 0, None)));
            }
            if let Kernel::Gemm { mi, nt } = default_kernel {
                kernels.extend(pfs.iter().map(|&pf| (None, pf, None)));
                // The GEMM's other shapes at 64-token tiles (#1181): the same bits.
                if nt == 8 {
                    for code in GEMM_DENSE_TUNES {
                        let tune = Some(GemmTune::from_code(code));
                        if glu {
                            kernels.push((None, 0, tune));
                        } else {
                            for mi2 in [1, 2] {
                                kernels.push((Some(Kernel::Gemm { mi: mi2, nt }), 0, tune));
                            }
                        }
                    }
                    let stream: &[i32] = if glu {
                        &GEMM_STREAM_GLU_TUNES
                    } else {
                        &GEMM_STREAM_TUNES
                    };
                    for &code in stream {
                        let tune = Some(GemmTune::from_code(code));
                        let kernel = (!glu).then_some(Kernel::Gemm { mi: 1, nt });
                        kernels.push((kernel, 0, tune));
                    }
                    if !glu {
                        kernels.push((Some(Kernel::Gemm { mi: 3 - mi, nt }), 0, None));
                        kernels.push((Some(Kernel::Gemm { mi, nt: 4 }), 0, None));
                    }
                }
            }
            let set = |(kernel, pf, tune): (Option<Kernel>, i32, Option<GemmTune>)| {
                force_kernel(kernel);
                force_gemm_pf(pf);
                force_gemm_tune(tune, None);
            };
            let elem = if glu { 2 } else { 4 };
            let mut reference: Vec<u8> = Vec::new();
            let first = rows.len();
            for &(kernel, pf, tune) in &kernels {
                set((kernel, pf, tune));
                let ok = gpu.compute(&graph).is_ok();
                let got: Vec<u8> = if ok {
                    outs.iter().flat_map(|o| o.bytes()).collect()
                } else {
                    Vec::new()
                };
                // The reference: the default's second run (a first run can give other bits).
                if rows.len() <= first + 1 {
                    if !ok {
                        set((None, 0, None));
                        return Err(format!("k {k} m {m} n {n}: the default does not run"));
                    }
                    reference = got.clone();
                }
                let differ = if ok {
                    got.chunks(elem)
                        .zip(reference.chunks(elem))
                        .filter(|(a, b)| a != b)
                        .count()
                } else {
                    0
                };
                rows.push(DenseRow {
                    k,
                    m,
                    n,
                    kernel: kernel.unwrap_or(default_kernel),
                    default: kernel.is_none() && pf == 0 && tune.is_none(),
                    pf,
                    tune,
                    glu,
                    secs: ok.then_some(f64::INFINITY),
                    bytes,
                    differ,
                });
            }
            // Rounds: the variants take turns within a round (the same clocks for all).
            for _ in 0..rounds {
                for (i, &variant) in kernels.iter().enumerate() {
                    let row = &mut rows[first + i];
                    let Some(best) = row.secs else { continue };
                    set(variant);
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        gpu.compute(&graph).map_err(|e| e.to_string())?;
                    }
                    let s = t.elapsed().as_secs_f64() / (reps * layers) as f64;
                    row.secs = Some(best.min(s));
                }
            }
            set((None, 0, None));
        }
    }
    Ok(rows)
}

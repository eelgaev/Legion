//! Dense NVFP4 matmuls at small batches in the mv kernel's variants (`legion nvfp4-mv-bench
//! --dense`): a `[k, m]` weight times `n` tokens, each variant forced through [`force_kernel`],
//! its outputs checked against the default kernel's bits and timed (the weight bytes over the
//! time: the stream's bandwidth). For tuning [`pick_dense`] at speculative verify steps (a
//! drafter's block of tokens per sequence).

use super::{Kernel, force_kernel, mul_mat, pick_dense};
use crate::ggml::{Backend, ComputeAllocator, Context, types};

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
/// default's pick).
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
        let bytes = ws[0].nbytes() as f64;
        for &n in toks {
            let ctx = Context::new(4 * layers + 4, 4 * layers + 4);
            let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let mut graph = ctx.new_graph(4 * layers + 4);
            let outs: Vec<_> = ws
                .iter()
                .map(|w| {
                    let y = mul_mat(&ctx, *w, x, None).as_output();
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
            let mut kernels = vec![None, None];
            kernels.extend(variants(n).into_iter().map(Some));
            let mut reference: Vec<f32> = Vec::new();
            let first = rows.len();
            for kernel in &kernels {
                force_kernel(*kernel);
                let ok = gpu.compute(&graph).is_ok();
                let got: Vec<f32> = if ok {
                    outs.iter().flat_map(|o| o.to_f32()).collect()
                } else {
                    Vec::new()
                };
                if rows.len() == first {
                    if !ok {
                        force_kernel(None);
                        return Err(format!("k {k} m {m} n {n}: the default does not run"));
                    }
                    reference = got.clone();
                }
                let differ = if ok {
                    got.iter()
                        .zip(&reference)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count()
                } else {
                    0
                };
                rows.push(DenseRow {
                    k,
                    m,
                    n,
                    kernel: kernel
                        .unwrap_or_else(|| pick_dense(k as i64, m as i64, n as i64, sms, mv_max_n)),
                    default: kernel.is_none(),
                    secs: ok.then_some(f64::INFINITY),
                    bytes,
                    differ,
                });
            }
            // Rounds: the variants take turns within a round (the same clocks for all).
            for _ in 0..rounds {
                for (i, kernel) in kernels.iter().enumerate() {
                    let row = &mut rows[first + i];
                    let Some(best) = row.secs else { continue };
                    force_kernel(*kernel);
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        gpu.compute(&graph).map_err(|e| e.to_string())?;
                    }
                    let s = t.elapsed().as_secs_f64() / (reps * layers) as f64;
                    row.secs = Some(best.min(s));
                }
            }
            force_kernel(None);
        }
    }
    Ok(rows)
}

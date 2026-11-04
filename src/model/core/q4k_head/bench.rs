//! The Q4_K draft LM head in isolation (#836): `w @ x` for a Q4_K `[k, rows]` and 1-8 columns on
//! one GPU, as ggml's `mul_mat` (its quantization and mmvq: the reference bits) and as the op in
//! its variants ([`super::force_variant`]), timed (the weight bytes over the time). Each timed
//! graph reads distinct copies of the weight, so that the GPU's L2 cache holds none of it.
//! [`check`] compares the bits in one graph that holds both. Used by the hidden
//! `legion q4k-head-bench` subcommand (a build tree's binary on the GPU box).

use super::{K, MAX_COLS, Q4_K_BYTES, QK_K, force_variant, mul_mat};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, types};

/// One variant: its name and the kernel variant forced (`None`: the default); ggml's `mul_mat`
/// for `op` false.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub name: &'static str,
    pub op: bool,
    pub force: Option<(i32, i32, i32)>,
}

const fn v(name: &'static str, op: bool, force: Option<(i32, i32, i32)>) -> Variant {
    Variant { name, op, force }
}

/// The variants, the reference (ggml) first: warps per CTA, most CTAs per SM (0: all that fit),
/// rows a warp step.
pub const VARIANTS: [Variant; 9] = [
    v("ggml", false, None),
    v("op", true, None),
    v("op w4r1", true, Some((4, 0, 1))),
    v("op w4r2", true, Some((4, 0, 2))),
    v("op w4r4", true, Some((4, 0, 4))),
    v("op w8r1", true, Some((8, 0, 1))),
    v("op w8r4", true, Some((8, 0, 4))),
    v("op w16r1", true, Some((16, 0, 1))),
    v("op w16r2", true, Some((16, 0, 2))),
];

/// What [`run`] and [`check`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// Values per row.
    pub k: usize,
    /// Row counts of the weights.
    pub rows: Vec<usize>,
    /// Column counts of the input.
    pub cols: Vec<usize>,
    pub rounds: usize,
    /// Weight bytes a timed graph reads at least (copies of the weight, at most [`MAX_COPIES`]).
    pub min_bytes: usize,
}

/// Most copies of a weight a graph reads.
pub const MAX_COPIES: usize = 16;

impl Default for Config {
    /// The head with `--spec-draft-vocab 65536` and the whole 248,320-row head, at one column
    /// (C=1) and four (C=4).
    fn default() -> Self {
        Self {
            k: 2560,
            rows: vec![65536, 248320],
            cols: vec![1, 4],
            rounds: 5,
            min_bytes: 1 << 30,
        }
    }
}

/// One variant of one weight and column count.
#[derive(Clone, Debug)]
pub struct Row {
    pub variant: &'static str,
    pub rows: usize,
    pub cols: usize,
    /// Best mean time per copy of the weight over the rounds (seconds).
    pub secs: f64,
    /// Weight bytes.
    pub bytes: f64,
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

/// `bytes` (whole rows of `k` values) of Q4_K weights: ggml's quantization of up to ~1 MiB of
/// rows of noise, repeated; `seed` picks them.
fn weights(bytes: usize, k: usize, seed: u64) -> Vec<u8> {
    let row = k / QK_K * Q4_K_BYTES;
    let rows = (bytes / row).clamp(1, (1 << 20) / row);
    let f: Vec<f32> = noise(rows * k, seed).iter().map(|v| 0.05 * v).collect();
    let pattern = crate::ggml::quantize(types::Q4_K, &f, k);
    pattern.iter().copied().cycle().take(bytes).collect()
}

/// The graph of one matmul per weight of `ws`, on the op or ggml's `mul_mat`: the input and the
/// graph.
fn build<'a>(
    ctx: &'a Context,
    op: bool,
    ws: &[Tensor<'a>],
    k: usize,
    n: usize,
) -> (Tensor<'a>, Graph<'a>) {
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let mut graph = ctx.new_graph(8 * ws.len() + 16);
    for &w in ws {
        let y = if op {
            mul_mat(ctx, w, x)
        } else {
            ctx.mul_mat(w, x)
        };
        graph.expand(y.as_output());
    }
    (x, graph)
}

fn check_shape(k: usize, rows: usize, n: usize) -> Result<(), String> {
    if !K.contains(&(k as i64)) || rows == 0 || n == 0 || n as i64 > MAX_COLS {
        return Err(format!(
            "k {k}, rows {rows}, cols {n}: k one of {K:?}, rows > 0, 1-{MAX_COLS} columns"
        ));
    }
    Ok(())
}

/// Every variant for every weight and column count of `cfg` on `gpu`; the variants take turns
/// within a round, the best round counts. The kernel choice is back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut out = Vec::new();
    let k = cfg.k;
    for &rows in &cfg.rows {
        check_shape(k, rows, 1)?;
        let bytes = rows * (k / QK_K) * Q4_K_BYTES;
        let copies = cfg.min_bytes.div_ceil(bytes).clamp(1, MAX_COPIES);
        let wctx = Context::new(copies + 1, 0);
        let ws: Vec<Tensor<'_>> = (0..copies)
            .map(|_| wctx.new_tensor(types::Q4_K, &[k as i64, rows as i64]))
            .collect();
        wctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(w.nbytes(), k, 3 + i as u64));
        }
        for &n in &cfg.cols {
            check_shape(k, rows, n)?;
            let kinds = [false, true];
            let ctxs = kinds.map(|_| Context::new(16 * copies + 16, 8 * copies + 16));
            let graphs: Vec<_> = kinds
                .iter()
                .zip(&ctxs)
                .map(|(&op, ctx)| build(ctx, op, &ws, k, n))
                .collect();
            let xs = noise(k * n, 5);
            for (ctx, (x, graph)) in ctxs.iter().zip(&graphs) {
                ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                x.set_f32(&xs);
            }
            let mut live = [true; VARIANTS.len()];
            let order: Vec<usize> = (0..VARIANTS.len())
                .chain((0..cfg.rounds).flat_map(|_| 0..VARIANTS.len()))
                .collect();
            let mut best = [f64::MAX; VARIANTS.len()];
            for (i, &vi) in order.iter().enumerate() {
                let var = &VARIANTS[vi];
                if !live[vi] {
                    continue;
                }
                let (_, graph) = &graphs[usize::from(var.op)];
                if var.op && !force_variant(var.force) {
                    live[vi] = false;
                    continue;
                }
                let t = std::time::Instant::now();
                let r = gpu.compute(graph);
                let secs = t.elapsed().as_secs_f64();
                force_variant(None);
                if let Err(e) = r {
                    if vi == 0 {
                        return Err(format!("{rows} rows {n} cols {}: {e}", var.name));
                    }
                    live[vi] = false;
                    continue;
                }
                if i >= VARIANTS.len() {
                    best[vi] = best[vi].min(secs / copies as f64);
                }
            }
            for (vi, var) in VARIANTS.iter().enumerate() {
                if live[vi] {
                    out.push(Row {
                        variant: var.name,
                        rows,
                        cols: n,
                        secs: best[vi],
                        bytes: bytes as f64,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// One weight and column count in [`check`]: outputs whose bits differ from ggml's over every op
/// variant, the variants that ran, the outputs per variant, and an FNV-1a hash of the op's
/// outputs (the last variant's).
#[derive(Clone, Debug)]
pub struct Check {
    pub rows: usize,
    pub cols: usize,
    pub differing: usize,
    pub variants: usize,
    pub outputs: usize,
    pub hash: u64,
}

/// FNV-1a over the bits of `v`.
fn hash(v: &[f32]) -> u64 {
    v.iter()
        .flat_map(|f| f.to_bits().to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        })
}

/// The bits of every op variant against ggml's `mul_mat` in one graph, for every weight and
/// column count of `cfg` on `gpu`, with a few inputs (one with all-zero blocks and a large value:
/// ggml's quantization's corner cases).
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    let k = cfg.k;
    for &rows in &cfg.rows {
        for &n in &cfg.cols {
            check_shape(k, rows, n)?;
            let ctx = Context::new(32, 32);
            let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let w = ctx
                .new_tensor(types::Q4_K, &[k as i64, rows as i64])
                .as_input();
            let reference = ctx.mul_mat(w, x).as_output();
            let op = mul_mat(&ctx, w, x).as_output();
            let mut graph = ctx.new_graph(32);
            graph.expand(reference);
            graph.expand(op);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            w.set_bytes(&weights(w.nbytes(), k, 2000 + rows as u64));
            let (mut differing, mut variants, mut h) = (0, 0, 0);
            for seed in 0..3u64 {
                let mut xs = noise(k * n, 1000 + seed);
                if seed == 1 {
                    xs[32..64].fill(0.0);
                    xs[7] = 300.0;
                }
                if seed == 2 {
                    // Activations of a residual's size, as the head reads them.
                    xs.iter_mut().for_each(|v| *v *= 40.0);
                }
                x.set_f32(&xs);
                for var in VARIANTS.iter().filter(|v| v.op) {
                    if !force_variant(var.force) {
                        continue;
                    }
                    let r = gpu.compute(&graph);
                    force_variant(None);
                    if r.is_err() && var.force.is_some() {
                        // A variant whose shared memory does not fit this column count.
                        continue;
                    }
                    r.map_err(|e| format!("{rows} rows {n} cols {}: {e}", var.name))?;
                    variants += 1;
                    let (a, b) = (op.to_f32(), reference.to_f32());
                    differing += a
                        .iter()
                        .zip(&b)
                        .filter(|(p, q)| p.to_bits() != q.to_bits())
                        .count();
                    h = hash(&a);
                }
            }
            out.push(Check {
                rows,
                cols: n,
                differing,
                variants,
                outputs: rows * n,
                hash: h,
            });
        }
    }
    Ok(out)
}

/// The draft pass's tail in [`run_pick`] and [`check_pick`] at `x`'s columns: what a draft pass
/// ran after the op before the pick (the split argmax, ggml's soft_max and the get_rows of each
/// pick's probability, as `mtp.rs`'s `draft_pick`), or [`super::head`]: the logits, the picks
/// and their probabilities.
fn tail<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>, pick: bool) -> [Tensor<'a>; 3] {
    if pick {
        let (l, a, p) = super::head(ctx, w, x);
        return [l, a, p];
    }
    let logits = mul_mat(ctx, w, x);
    let argmax = crate::model::cluster_select::argmax(ctx, logits);
    let probs = ctx.soft_max(logits);
    let (rows, n) = (w.ne()[1], x.ne()[1]);
    let pmax = ctx.get_rows(
        ctx.reshape_3d(probs, 1, rows, n),
        ctx.reshape_2d(argmax, 1, n),
    );
    [logits, argmax, ctx.reshape_1d(pmax, n)]
}

/// Column counts the pick takes for `rows` ([`super::q8_head::pick::coop`]): an error for
/// any other.
fn check_pick_shape(k: usize, rows: usize, n: usize) -> Result<(), String> {
    check_shape(k, rows, n)?;
    if !crate::model::core::q8_head::pick::coop(rows as i64, n as i64) {
        return Err(format!(
            "rows {rows}, cols {n}: the pick takes 1-{} columns of more than {} rows each",
            crate::model::core::q8_head::pick::MAX_COLS,
            crate::model::core::q8_head::pick::COOP_COLS,
        ));
    }
    Ok(())
}

/// The tails' names in [`run_pick`]'s rows: the op with the split argmax, soft_max and
/// get_rows, and with the pick.
pub const PICK_VARIANTS: [&str; 2] = ["head+softmax", "head+pick"];

/// The draft pass's head and pick for every weight and column count of `cfg` on `gpu`, as the
/// draft pass ran them before the pick and with it (the op in its default variant); timed as
/// [`run`] does (distinct weight copies, the two take turns, the best round counts).
pub fn run_pick(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    crate::model::cluster_select::set_cuda(true);
    let mut out = Vec::new();
    let k = cfg.k;
    for &rows in &cfg.rows {
        for &n in &cfg.cols {
            check_pick_shape(k, rows, n)?;
            out.extend(run_pick_one(gpu, cfg, rows, n)?);
        }
    }
    Ok(out)
}

/// [`run_pick`] of one weight and column count.
fn run_pick_one(
    gpu: &mut Backend,
    cfg: &Config,
    rows: usize,
    n: usize,
) -> Result<Vec<Row>, String> {
    let mut out = Vec::new();
    let k = cfg.k;
    {
        let bytes = rows * (k / QK_K) * Q4_K_BYTES;
        let copies = cfg.min_bytes.div_ceil(bytes).clamp(1, MAX_COPIES);
        let wctx = Context::new(copies + 1, 0);
        let ws: Vec<Tensor<'_>> = (0..copies)
            .map(|_| wctx.new_tensor(types::Q4_K, &[k as i64, rows as i64]))
            .collect();
        wctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(w.nbytes(), k, 3 + i as u64));
        }
        let ctxs = [false, true].map(|_| Context::new(16 * copies + 16, 16 * copies + 16));
        let mut graphs = Vec::new();
        for (pick, ctx) in [false, true].into_iter().zip(&ctxs) {
            let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let mut graph = ctx.new_graph(16 * ws.len() + 16);
            for &w in &ws {
                for t in tail(ctx, w, x, pick) {
                    graph.expand(t.as_output());
                }
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            x.set_f32(&noise(k * n, 5));
            graphs.push(graph);
        }
        let mut best = [f64::MAX; 2];
        for round in 0..=cfg.rounds {
            for (i, graph) in graphs.iter().enumerate() {
                let t = std::time::Instant::now();
                gpu.compute(graph)
                    .map_err(|e| format!("{rows} rows {n} cols {}: {e}", PICK_VARIANTS[i]))?;
                if round > 0 {
                    best[i] = best[i].min(t.elapsed().as_secs_f64() / copies as f64);
                }
            }
        }
        for (i, &name) in PICK_VARIANTS.iter().enumerate() {
            out.push(Row {
                variant: name,
                rows,
                cols: n,
                secs: best[i],
                bytes: bytes as f64,
            });
        }
    }
    Ok(out)
}

/// One weight and column count in [`check_pick`]: input columns compared (over every op
/// variant), those whose logits, pick or probability bits differ between the two tails (a NaN
/// probability equals any NaN), columns with a tied maximum, and an FNV-1a hash of the pick's
/// (pick, probability) over them.
#[derive(Clone, Debug)]
pub struct PickCheck {
    pub rows: usize,
    pub cols: usize,
    pub inputs: usize,
    pub differing: usize,
    pub ties: usize,
    pub hash: u64,
}

/// The pick's outputs against the draft pass's tail before it, both in one graph, for every
/// weight and column count of `cfg` on `gpu` and every op variant: noise inputs at a few scales
/// (a different scale in each column), an all-zero input (every logit equal: the tie order
/// decides), and the weight with one row's scale NaN (a NaN logit: never picked, a NaN sum).
/// The weights repeat every few hundred rows, so most inputs have tied maxima.
pub fn check_pick(gpu: &mut Backend, cfg: &Config) -> Result<Vec<PickCheck>, String> {
    crate::model::cluster_select::set_cuda(true);
    let mut out = Vec::new();
    for &rows in &cfg.rows {
        for &n in &cfg.cols {
            check_pick_shape(cfg.k, rows, n)?;
            out.push(check_pick_one(gpu, cfg.k, rows, n)?);
        }
    }
    Ok(out)
}

/// [`check_pick`] of one weight and column count.
fn check_pick_one(gpu: &mut Backend, k: usize, rows: usize, n: usize) -> Result<PickCheck, String> {
    {
        let ctx = Context::new(64, 64);
        let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
        let w = ctx
            .new_tensor(types::Q4_K, &[k as i64, rows as i64])
            .as_input();
        let want = tail(&ctx, w, x, false).map(|t| t.as_output());
        let got = tail(&ctx, w, x, true).map(|t| t.as_output());
        let mut graph = ctx.new_graph(64);
        for t in want.iter().chain(&got) {
            graph.expand(*t);
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        let wb = weights(w.nbytes(), k, 4000 + rows as u64);
        let (mut inputs, mut differing, mut ties) = (0, 0, 0);
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut one = |xs: &[f32], var: &Variant| -> Result<(), String> {
            if !force_variant(var.force) {
                return Ok(());
            }
            x.set_f32(xs);
            let r = gpu.compute(&graph);
            force_variant(None);
            if r.is_err() && var.force.is_some() {
                // A variant whose shared memory does not fit.
                return Ok(());
            }
            r.map_err(|e| format!("{rows} rows {n} cols {}: {e}", var.name))?;
            let (lw_all, lg_all) = (want[0].to_f32(), got[0].to_f32());
            let (aw_all, ag_all) = (want[1].to_i32(), got[1].to_i32());
            let (pw_all, pg_all) = (want[2].to_f32(), got[2].to_f32());
            for j in 0..n {
                let (lw, lg) = (&lw_all[j * rows..][..rows], &lg_all[j * rows..][..rows]);
                let (aw, ag) = (aw_all[j], ag_all[j]);
                let (pw, pg) = (pw_all[j], pg_all[j]);
                let same_p = pw.to_bits() == pg.to_bits() || (pw.is_nan() && pg.is_nan());
                let same_l = lw.iter().zip(lg).all(|(a, b)| a.to_bits() == b.to_bits());
                inputs += 1;
                if aw != ag || !same_p || !same_l {
                    differing += 1;
                    eprintln!(
                        "{rows} rows {n} cols {} input {inputs} col {j}: pick {aw} vs {ag}, p {pw:e} vs {pg:e}, logits equal {same_l}",
                        var.name
                    );
                }
                if aw >= 0 {
                    let m = lw[aw as usize];
                    ties += usize::from(lw.iter().filter(|&&v| v == m).count() > 1);
                }
                for b in ag
                    .to_le_bytes()
                    .into_iter()
                    .chain(pg.to_bits().to_le_bytes())
                {
                    h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
                }
            }
            Ok(())
        };
        for var in VARIANTS.iter().filter(|v| v.op) {
            w.set_bytes(&wb);
            for seed in 0..16u64 {
                let xs: Vec<f32> = (0..n as u64)
                    .flat_map(|j| {
                        let scale = [1.0, 10.0, 40.0, 0.01][(seed + j) as usize % 4];
                        noise(k, 7000 + 16 * j + seed)
                            .into_iter()
                            .map(move |v| v * scale)
                    })
                    .collect();
                one(&xs, var)?;
            }
            one(&vec![0.0; k * n], var)?;
            // Row 5's scale NaN (F16 0x7e00, the first block's d): its logit is NaN.
            let mut wn = wb.clone();
            let at = 5 * (k / QK_K) * Q4_K_BYTES;
            wn[at..at + 2].copy_from_slice(&0x7e00u16.to_le_bytes());
            w.set_bytes(&wn);
            one(&noise(k * n, 7100), var)?;
        }
        Ok(PickCheck {
            rows,
            cols: n,
            inputs,
            differing,
            ties,
            hash: h,
        })
    }
}

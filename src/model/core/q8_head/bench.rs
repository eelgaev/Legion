//! The one-column Q8_0 draft LM head in isolation (#814): `w @ x` for a Q8_0 `[k, rows]` and one
//! column on one GPU, as ggml's `mul_mat` (its quantization and mmvq: the reference bits) and as
//! the op in its variants ([`super::force_variant`]), timed (the weight bytes over the time).
//! Each timed graph reads distinct copies of the weight, so that the GPU's L2 cache holds none of
//! it. [`check`] compares the bits in one graph that holds both. Used by the hidden
//! `legion q8-head-bench` subcommand (a build tree's binary on the GPU box).

use super::{GROUP, K, K_OUT, Q8_0_BYTES, force_variant, mul_mat, mul_mat_out};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};

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
pub const VARIANTS: [Variant; 8] = [
    v("ggml", false, None),
    v("op", true, None),
    v("op w8", true, Some((8, 0, 1))),
    v("op w4c4", true, Some((4, 4, 1))),
    v("op w4r2", true, Some((4, 0, 2))),
    v("op w8r2", true, Some((8, 0, 2))),
    v("op w4c4r2", true, Some((4, 4, 2))),
    v("op w16r2", true, Some((16, 0, 2))),
];

/// The variants for the 6,144-value projections ([`super::mul_mat_out`], #814), the reference
/// (ggml) first. 16 warps (or 8 with 2 rows a step) do not fit their shared memory.
pub const OUT_VARIANTS: [Variant; 8] = [
    v("ggml", false, None),
    v("op", true, None),
    v("op w4r2", true, Some((4, 0, 2))),
    v("op w8", true, Some((8, 0, 1))),
    v("op w4c3", true, Some((4, 3, 1))),
    v("op w4c1", true, Some((4, 1, 1))),
    v("op w4c2", true, Some((4, 2, 1))),
    v("op w8c1", true, Some((8, 1, 1))),
];

/// The variants [`run`] and [`check`] take for rows of `k` values.
pub fn variants(k: usize) -> &'static [Variant] {
    if k as i64 == K_OUT {
        &OUT_VARIANTS
    } else {
        &VARIANTS
    }
}

/// The op's node for `w @ x`: the head's, or for rows of [`K_OUT`] values the projections'.
fn op_mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    if w.ne()[0] == K_OUT {
        mul_mat_out(ctx, w, x)
    } else {
        mul_mat(ctx, w, x)
    }
}

/// What [`run`] and [`check`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// Values per row.
    pub k: usize,
    /// Row counts of the weights.
    pub rows: Vec<usize>,
    pub rounds: usize,
    /// Weight bytes a timed graph reads at least (copies of the weight, at most [`MAX_COPIES`]).
    pub min_bytes: usize,
}

/// Most copies of a weight a graph reads.
pub const MAX_COPIES: usize = 16;

impl Default for Config {
    /// Qwen3.8-Flash-Next's whole MTP draft head, the head with `--spec-draft-vocab 65536`, and
    /// a dense projection's size.
    fn default() -> Self {
        Self {
            k: 2560,
            rows: vec![248320, 65536, 10240],
            rounds: 5,
            min_bytes: 1 << 30,
        }
    }
}

/// One variant of one weight.
#[derive(Clone, Debug)]
pub struct Row {
    pub variant: &'static str,
    pub rows: usize,
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

/// `bytes` (whole Q8_0 blocks) of Q8_0 weights: up to 1 MiB of blocks (int8 in -127..=127,
/// scales in [0.001, 0.011)), repeated; `seed` picks them.
fn weights(bytes: usize, seed: u64) -> Vec<u8> {
    let blocks = (bytes / Q8_0_BYTES).min((1 << 20) / Q8_0_BYTES);
    let qs = noise(blocks * GROUP, seed);
    let ds = noise(blocks, seed + 1);
    let mut pattern = Vec::with_capacity(blocks * Q8_0_BYTES);
    for (b, d) in ds.iter().enumerate() {
        // SAFETY: a plain conversion.
        let h = unsafe { ffi::ggml_fp32_to_fp16(0.006 + 0.005 * d) };
        pattern.extend_from_slice(&h.to_le_bytes());
        for q in &qs[b * GROUP..(b + 1) * GROUP] {
            pattern.push(((q * 127.0).round() as i8) as u8);
        }
    }
    pattern.iter().copied().cycle().take(bytes).collect()
}

/// The graph of one matmul per weight of `ws`, on the op or ggml's `mul_mat`: the input and the
/// graph.
fn build<'a>(ctx: &'a Context, op: bool, ws: &[Tensor<'a>], k: usize) -> (Tensor<'a>, Graph<'a>) {
    let x = ctx.new_tensor(types::F32, &[k as i64, 1]).as_input();
    let mut graph = ctx.new_graph(8 * ws.len() + 16);
    for &w in ws {
        let y = if op {
            op_mul_mat(ctx, w, x)
        } else {
            ctx.mul_mat(w, x)
        };
        graph.expand(y.as_output());
    }
    (x, graph)
}

fn check_shape(k: usize, rows: usize) -> Result<(), String> {
    if !K.contains(&(k as i64)) || rows == 0 {
        return Err(format!("k {k}, rows {rows}: k one of {K:?}, rows > 0"));
    }
    Ok(())
}

/// [`check_shape`] for [`run`] and [`check`], which also take rows of [`K_OUT`] values.
fn check_mv_shape(k: usize, rows: usize) -> Result<(), String> {
    if k as i64 == K_OUT && rows > 0 {
        return Ok(());
    }
    check_shape(k, rows).map_err(|e| format!("{e} (or {K_OUT})"))
}

/// Every variant for every weight of `cfg` on `gpu`; the variants take turns within a round, the
/// best round counts. The kernel choice is back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut out = Vec::new();
    let k = cfg.k;
    for &rows in &cfg.rows {
        check_mv_shape(k, rows)?;
        let vars = variants(k);
        let bytes = rows * (k / GROUP) * Q8_0_BYTES;
        let copies = cfg.min_bytes.div_ceil(bytes).clamp(1, MAX_COPIES);
        let wctx = Context::new(copies + 1, 0);
        let ws: Vec<Tensor<'_>> = (0..copies)
            .map(|_| wctx.new_tensor(types::Q8_0, &[k as i64, rows as i64]))
            .collect();
        wctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(w.nbytes(), 3 + i as u64));
        }
        let kinds = [false, true];
        let ctxs = kinds.map(|_| Context::new(16 * copies + 16, 8 * copies + 16));
        let graphs: Vec<_> = kinds
            .iter()
            .zip(&ctxs)
            .map(|(&op, ctx)| build(ctx, op, &ws, k))
            .collect();
        let xs = noise(k, 5);
        for (ctx, (x, graph)) in ctxs.iter().zip(&graphs) {
            ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            x.set_f32(&xs);
        }
        let mut live = vec![true; vars.len()];
        let order: Vec<usize> = (0..vars.len())
            .chain((0..cfg.rounds).flat_map(|_| 0..vars.len()))
            .collect();
        let mut best = vec![f64::MAX; vars.len()];
        for (i, &vi) in order.iter().enumerate() {
            let var = &vars[vi];
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
                    return Err(format!("{rows} rows {}: {e}", var.name));
                }
                live[vi] = false;
                continue;
            }
            if i >= vars.len() {
                best[vi] = best[vi].min(secs / copies as f64);
            }
        }
        for (vi, var) in vars.iter().enumerate() {
            if live[vi] {
                out.push(Row {
                    variant: var.name,
                    rows,
                    secs: best[vi],
                    bytes: bytes as f64,
                });
            }
        }
    }
    Ok(out)
}

/// One weight in [`check`]: outputs whose bits differ from ggml's over every op variant, the
/// variants that ran, the outputs per variant, and an FNV-1a hash of the op's outputs (the last
/// variant's).
#[derive(Clone, Debug)]
pub struct Check {
    pub rows: usize,
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

/// The bits of every op variant against ggml's `mul_mat` in one graph, for every weight of `cfg`
/// on `gpu`, with a few inputs (one with all-zero blocks and a large value: ggml's
/// quantization's corner cases).
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    let k = cfg.k;
    for &rows in &cfg.rows {
        check_mv_shape(k, rows)?;
        let vars = variants(k);
        let ctx = Context::new(32, 32);
        let x = ctx.new_tensor(types::F32, &[k as i64, 1]).as_input();
        let w = ctx
            .new_tensor(types::Q8_0, &[k as i64, rows as i64])
            .as_input();
        let reference = ctx.mul_mat(w, x).as_output();
        let op = op_mul_mat(&ctx, w, x).as_output();
        let mut graph = ctx.new_graph(32);
        graph.expand(reference);
        graph.expand(op);
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        w.set_bytes(&weights(w.nbytes(), 2000 + rows as u64));
        let (mut differing, mut variants, mut h) = (0, 0, 0);
        for seed in 0..3u64 {
            let mut xs = noise(k, 1000 + seed);
            if seed == 1 {
                xs[32..64].fill(0.0);
                xs[7] = 300.0;
            }
            if seed == 2 {
                // Activations of a residual's size, as the head reads them.
                xs.iter_mut().for_each(|v| *v *= 40.0);
            }
            x.set_f32(&xs);
            for var in vars.iter().filter(|v| v.op) {
                if !force_variant(var.force) {
                    continue;
                }
                let r = gpu.compute(&graph);
                force_variant(None);
                r.map_err(|e| format!("{rows} rows {}: {e}", var.name))?;
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
            differing,
            variants,
            outputs: rows,
            hash: h,
        });
    }
    Ok(out)
}

/// The draft pass's tail in [`run_pick`] and [`check_pick`]: what a draft pass ran after the
/// head before the pick ops (the split argmax, ggml's soft_max and the get_rows of the pick's
/// probability), or [`super::pick::head`]: the logits, the pick and its probability.
fn tail<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>, pick: bool) -> [Tensor<'a>; 3] {
    if pick {
        let (l, a, p) = super::pick::head(ctx, w, x);
        return [l, a, p];
    }
    let logits = mul_mat(ctx, w, x);
    let argmax = crate::model::cluster_select::argmax(ctx, logits);
    let probs = ctx.soft_max(logits);
    let rows = w.ne()[1];
    let pmax = ctx.get_rows(
        ctx.reshape_3d(probs, 1, rows, 1),
        ctx.reshape_2d(argmax, 1, 1),
    );
    [logits, argmax, ctx.reshape_1d(pmax, 1)]
}

/// The tails' names in [`run_pick`]'s rows: the head with the split argmax, soft_max and
/// get_rows, and with the pick ops.
pub const PICK_VARIANTS: [&str; 2] = ["head+softmax", "head+pick"];

/// The draft pass's head and pick for every weight of `cfg` on `gpu`, as the draft pass ran
/// them before the pick ops and with them (the head in its default variant); timed as [`run`]
/// does (distinct weight copies, the two take turns, the best round counts).
pub fn run_pick(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    crate::model::cluster_select::set_cuda(true);
    let mut out = Vec::new();
    let k = cfg.k;
    for &rows in &cfg.rows {
        check_shape(k, rows)?;
        let bytes = rows * (k / GROUP) * Q8_0_BYTES;
        let copies = cfg.min_bytes.div_ceil(bytes).clamp(1, MAX_COPIES);
        let wctx = Context::new(copies + 1, 0);
        let ws: Vec<Tensor<'_>> = (0..copies)
            .map(|_| wctx.new_tensor(types::Q8_0, &[k as i64, rows as i64]))
            .collect();
        wctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(w.nbytes(), 3 + i as u64));
        }
        let ctxs = [false, true].map(|_| Context::new(16 * copies + 16, 16 * copies + 16));
        let mut graphs = Vec::new();
        for (pick, ctx) in [false, true].into_iter().zip(&ctxs) {
            let x = ctx.new_tensor(types::F32, &[k as i64, 1]).as_input();
            let mut graph = ctx.new_graph(16 * ws.len() + 16);
            for &w in &ws {
                for t in tail(ctx, w, x, pick) {
                    graph.expand(t.as_output());
                }
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            x.set_f32(&noise(k, 5));
            graphs.push(graph);
        }
        let mut best = [f64::MAX; 2];
        for round in 0..=cfg.rounds {
            for (i, graph) in graphs.iter().enumerate() {
                let t = std::time::Instant::now();
                gpu.compute(graph)
                    .map_err(|e| format!("{rows} rows {}: {e}", PICK_VARIANTS[i]))?;
                if round > 0 {
                    best[i] = best[i].min(t.elapsed().as_secs_f64() / copies as f64);
                }
            }
        }
        for (i, &name) in PICK_VARIANTS.iter().enumerate() {
            out.push(Row {
                variant: name,
                rows,
                secs: best[i],
                bytes: bytes as f64,
            });
        }
    }
    Ok(out)
}

/// One weight in [`check_pick`]: inputs compared, those whose logits, pick or probability bits
/// differ between the two tails (a NaN probability equals any NaN), and an FNV-1a hash of the
/// pick ops' (pick, probability) over the inputs.
#[derive(Clone, Debug)]
pub struct PickCheck {
    pub rows: usize,
    pub inputs: usize,
    pub differing: usize,
    pub ties: usize,
    pub hash: u64,
}

/// The pick ops' outputs against the draft pass's tail before them, both in one graph, for
/// every weight of `cfg` on `gpu`: noise inputs at a few scales, an all-zero input (every
/// logit equal: the tie order decides), and the weight with one row's scale NaN (a NaN logit:
/// never picked, a NaN sum). The weights repeat every few hundred rows, so most inputs have
/// tied maxima.
pub fn check_pick(gpu: &mut Backend, cfg: &Config) -> Result<Vec<PickCheck>, String> {
    crate::model::cluster_select::set_cuda(true);
    let mut out = Vec::new();
    let k = cfg.k;
    for &rows in &cfg.rows {
        check_shape(k, rows)?;
        let ctx = Context::new(64, 64);
        let x = ctx.new_tensor(types::F32, &[k as i64, 1]).as_input();
        let w = ctx
            .new_tensor(types::Q8_0, &[k as i64, rows as i64])
            .as_input();
        let want = tail(&ctx, w, x, false).map(|t| t.as_output());
        let got = tail(&ctx, w, x, true).map(|t| t.as_output());
        let mut graph = ctx.new_graph(64);
        for t in want.iter().chain(&got) {
            graph.expand(*t);
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        let wb = weights(w.nbytes(), 4000 + rows as u64);
        w.set_bytes(&wb);
        let (mut inputs, mut differing, mut ties) = (0, 0, 0);
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut one = |xs: &[f32]| -> Result<(), String> {
            x.set_f32(xs);
            gpu.compute(&graph)
                .map_err(|e| format!("{rows} rows: {e}"))?;
            let (lw, lg) = (want[0].to_f32(), got[0].to_f32());
            let (aw, ag) = (want[1].to_i32()[0], got[1].to_i32()[0]);
            let (pw, pg) = (want[2].to_f32()[0], got[2].to_f32()[0]);
            let same_p = pw.to_bits() == pg.to_bits() || (pw.is_nan() && pg.is_nan());
            let same_l = lw.iter().zip(&lg).all(|(a, b)| a.to_bits() == b.to_bits());
            inputs += 1;
            if aw != ag || !same_p || !same_l {
                differing += 1;
                eprintln!(
                    "{rows} rows input {inputs}: pick {aw} vs {ag}, p {pw:e} vs {pg:e}, logits equal {same_l}"
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
            Ok(())
        };
        for seed in 0..32u64 {
            let scale = [1.0, 10.0, 40.0, 0.01][seed as usize % 4];
            let xs: Vec<f32> = noise(k, 7000 + seed).iter().map(|v| v * scale).collect();
            one(&xs)?;
        }
        one(&vec![0.0; k])?;
        // Row 5's scale NaN (F16 0x7e00): its logit is NaN.
        let mut wn = wb.clone();
        let at = 5 * (k / GROUP) * Q8_0_BYTES;
        wn[at..at + 2].copy_from_slice(&0x7e00u16.to_le_bytes());
        w.set_bytes(&wn);
        one(&noise(k, 7100))?;
        out.push(PickCheck {
            rows,
            inputs,
            differing,
            ties,
            hash: h,
        });
    }
    Ok(out)
}

/// The decode matvecs' variants in [`run_group`] and [`check_group`] (#814): ggml's `mul_mat`,
/// q8_group's copy of ggml's mmvq ([`super::set_row_mv`] off), then the kernel ([`super::row_mv`])
/// in its default and forced variants.
pub const GROUP_VARIANTS: [Variant; 9] = [
    v("ggml", false, None),
    v("group", false, None),
    v("row", true, None),
    v("row w4r1", true, Some((4, 0, 1))),
    v("row w8r1", true, Some((8, 0, 1))),
    v("row w8r2", true, Some((8, 0, 2))),
    v("row w16r1", true, Some((16, 0, 1))),
    v("row w4c4r2", true, Some((4, 4, 2))),
    v("row w4c8r1", true, Some((4, 8, 1))),
];

/// Whether a [`GROUP_VARIANTS`] entry runs on the q8_group graph (else ggml's).
fn on_group(vi: usize) -> bool {
    vi > 0
}

/// Sets the switches for [`GROUP_VARIANTS`] entry `vi`: false where the kernel lacks the variant.
fn group_variant(vi: usize) -> bool {
    let var = &GROUP_VARIANTS[vi];
    super::set_row_mv(var.op);
    !var.op || force_variant(var.force)
}

/// The graph of the decode matvecs of one copy per entry of `wss`: ggml's `mul_mat` of each
/// weight, or q8_group's (each weight its own node, or with `joined` one node for all of a
/// copy's weights, as an attention layer's Q, K and V), all reading one input.
fn build_group<'a>(
    ctx: &'a Context,
    group: bool,
    joined: bool,
    wss: &[Vec<Tensor<'a>>],
    k: usize,
) -> (Tensor<'a>, Graph<'a>, Vec<Tensor<'a>>) {
    let x = ctx.new_tensor(types::F32, &[k as i64, 1]).as_input();
    let mut graph = ctx.new_graph(16 * wss.len() * super::XQ_WEIGHTS + 16);
    let mut outs = Vec::new();
    for ws in wss {
        let ys: Vec<Tensor<'a>> = if !group {
            ws.iter().map(|&w| ctx.mul_mat(w, x)).collect()
        } else if joined && ws.len() > 1 {
            crate::model::core::q8_group::mul_mat(ctx, ws, x)
        } else {
            crate::model::core::q8_group::mul_mat_each(ctx, ws, x)
        };
        for y in ys {
            let y = y.as_output();
            graph.expand(y);
            outs.push(y);
        }
    }
    (x, graph, outs)
}

fn check_group_shape(k: usize, rows: &[usize]) -> Result<(), String> {
    if !K.contains(&(k as i64)) || rows.is_empty() || rows.len() > super::XQ_WEIGHTS {
        return Err(format!(
            "k {k}, rows {rows:?}: k one of {K:?}, 1 to {} weights",
            super::XQ_WEIGHTS
        ));
    }
    if rows.iter().any(|&r| r == 0 || r % 2 != 0) {
        return Err(format!("rows {rows:?}: even and above 0"));
    }
    Ok(())
}

/// Every [`GROUP_VARIANTS`] entry for the weights of `cfg` (`cfg.rows`: one weight each, or with
/// `joined` one node of all of them) on `gpu`, timed as [`run`] does; one [`Row`] per variant
/// with the rows of all weights. The switches are back at their defaults afterwards.
pub fn run_group(gpu: &mut Backend, cfg: &Config, joined: bool) -> Result<Vec<Row>, String> {
    let k = cfg.k;
    check_group_shape(k, &cfg.rows)?;
    let total: usize = cfg.rows.iter().sum();
    let bytes = total * (k / GROUP) * Q8_0_BYTES;
    let copies = cfg.min_bytes.div_ceil(bytes).clamp(1, MAX_COPIES);
    let wctx = Context::new(copies * cfg.rows.len() + 1, 0);
    let wss: Vec<Vec<Tensor<'_>>> = (0..copies)
        .map(|_| {
            cfg.rows
                .iter()
                .map(|&r| wctx.new_tensor(types::Q8_0, &[k as i64, r as i64]))
                .collect()
        })
        .collect();
    wctx.alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for (i, w) in wss.iter().flatten().enumerate() {
        w.set_bytes(&weights(w.nbytes(), 3 + i as u64));
    }
    let n = 32 * copies * super::XQ_WEIGHTS + 32;
    let ctxs = [false, true].map(|_| Context::new(n, n));
    let graphs: Vec<_> = [false, true]
        .iter()
        .zip(&ctxs)
        .map(|(&group, ctx)| build_group(ctx, group, joined, &wss, k))
        .collect();
    let xs = noise(k, 5);
    for (ctx, (x, graph, _)) in ctxs.iter().zip(&graphs) {
        ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        x.set_f32(&xs);
    }
    let nv = GROUP_VARIANTS.len();
    let mut live = [true; GROUP_VARIANTS.len()];
    let mut best = [f64::MAX; GROUP_VARIANTS.len()];
    let order: Vec<usize> = (0..nv).chain((0..cfg.rounds).flat_map(|_| 0..nv)).collect();
    for (i, &vi) in order.iter().enumerate() {
        if !live[vi] {
            continue;
        }
        if !group_variant(vi) {
            live[vi] = false;
            super::set_row_mv(true);
            continue;
        }
        let (_, graph, _) = &graphs[usize::from(on_group(vi))];
        let t = std::time::Instant::now();
        let r = gpu.compute(graph);
        let secs = t.elapsed().as_secs_f64();
        force_variant(None);
        super::set_row_mv(true);
        if let Err(e) = r {
            if vi < 2 {
                return Err(format!("{:?} {}: {e}", cfg.rows, GROUP_VARIANTS[vi].name));
            }
            live[vi] = false;
            continue;
        }
        if i >= nv {
            best[vi] = best[vi].min(secs / copies as f64);
        }
    }
    Ok(GROUP_VARIANTS
        .iter()
        .enumerate()
        .filter(|&(vi, _)| live[vi])
        .map(|(vi, var)| Row {
            variant: var.name,
            rows: total,
            secs: best[vi],
            bytes: bytes as f64,
        })
        .collect())
}

/// [`check`] for the decode matvecs: every q8_group variant of [`GROUP_VARIANTS`] against ggml's
/// `mul_mat` in one graph, for the weights of `cfg` (as [`run_group`] takes them); one [`Check`]
/// with the rows of all weights.
pub fn check_group(gpu: &mut Backend, cfg: &Config, joined: bool) -> Result<Vec<Check>, String> {
    let k = cfg.k;
    check_group_shape(k, &cfg.rows)?;
    let ctx = Context::new(256, 256);
    let ws: Vec<Tensor<'_>> = cfg
        .rows
        .iter()
        .map(|&r| {
            ctx.new_tensor(types::Q8_0, &[k as i64, r as i64])
                .as_input()
        })
        .collect();
    // One input for both: ggml's matmuls and q8_group's nodes in one graph.
    let x = ctx.new_tensor(types::F32, &[k as i64, 1]).as_input();
    let mut graph = ctx.new_graph(256);
    let reference: Vec<Tensor<'_>> = ws.iter().map(|&w| ctx.mul_mat(w, x).as_output()).collect();
    let ops: Vec<Tensor<'_>> = if joined && ws.len() > 1 {
        crate::model::core::q8_group::mul_mat(&ctx, &ws, x)
    } else {
        crate::model::core::q8_group::mul_mat_each(&ctx, &ws, x)
    }
    .into_iter()
    .map(|t| t.as_output())
    .collect();
    for &t in reference.iter().chain(&ops) {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .map_err(|e| e.to_string())?;
    for (i, w) in ws.iter().enumerate() {
        w.set_bytes(&weights(
            w.nbytes(),
            2000 + 7 * i as u64 + cfg.rows[i] as u64,
        ));
    }
    let (mut differing, mut variants, mut h) = (0, 0, 0);
    for seed in 0..3u64 {
        let mut xs = noise(k, 1000 + seed);
        if seed == 1 {
            xs[32..64].fill(0.0);
            xs[7] = 300.0;
        }
        if seed == 2 {
            xs.iter_mut().for_each(|v| *v *= 40.0);
        }
        x.set_f32(&xs);
        for vi in (1..GROUP_VARIANTS.len()).filter(|&vi| on_group(vi)) {
            if !group_variant(vi) {
                super::set_row_mv(true);
                continue;
            }
            let before = super::row_mv_launches();
            let r = gpu.compute(&graph);
            force_variant(None);
            super::set_row_mv(true);
            r.map_err(|e| format!("{:?} {}: {e}", cfg.rows, GROUP_VARIANTS[vi].name))?;
            if GROUP_VARIANTS[vi].op && super::row_mv_launches() == before {
                // A replayed CUDA graph launches nothing new: the bits are still the graph's.
                eprintln!(
                    "{:?} {}: no new kernel launch (a replayed graph?)",
                    cfg.rows, GROUP_VARIANTS[vi].name
                );
            }
            variants += 1;
            let a: Vec<f32> = ops.iter().flat_map(|t| t.to_f32()).collect();
            let b: Vec<f32> = reference.iter().flat_map(|t| t.to_f32()).collect();
            differing += a
                .iter()
                .zip(&b)
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count();
            h = hash(&a);
        }
    }
    Ok(vec![Check {
        rows: cfg.rows.iter().sum(),
        differing,
        variants,
        outputs: cfg.rows.iter().sum(),
        hash: h,
    }])
}

//! The decode BF16 matmuls in isolation (#836): groups of weights that read one input (one
//! weight, or e.g. a DeltaNet layer's `in_proj_qkv` and `in_proj_z`) at decode batches on one
//! GPU, run as ggml's `mul_mat` (the reference bits), as one op per weight and as one op for the
//! whole group, in the kernel's variants ([`super::force_variant`]), and times them (the weight
//! bytes over the time). Each timed graph reads distinct copies of the group's weights, so that
//! the GPU's L2 cache holds none of them. [`check`] compares the bits as the ignored GPU test
//! `dense_mv::tests::cuda_matches_ggml_bits` does (one graph holding the op and ggml's
//! `mul_mat`; comparing outputs across the timed graphs differs even between two runs of ggml's
//! own graph), for a release binary on the GPU box. Groups of F32 weights (`f32:k:m...`) are the
//! MoE router's kind. Used by the hidden `legion dense-mv-bench` subcommand (a build tree's
//! binary on the GPU box).

use super::{
    CHAIN_MAX_ROWS, CHAIN_MIN_TOKENS, MAX_TOKENS, MAX_WEIGHTS, MIN_BYTES, MIN_TOKENS,
    MIN_TOKENS_F32, WIDE_CHAIN_MAX_TOKENS, WideOrder, fits_wide_chain, force_variant, mul_mat,
    mul_mat_wide_chain, mul_mat_wide_chain_order, mul_mat_wide_chains, wide_chain_order,
};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};

/// A group of weights `[k, m_i]` that read the same input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub k: usize,
    pub ms: Vec<usize>,
    /// F32 weights (else BF16).
    pub f32: bool,
}

impl Group {
    /// `k:m1+m2+...`, e.g. `2560:10240+6144`; F32 weights `f32:k:m1+...`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let (f32, body) = match s.strip_prefix("f32:") {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        let (k, ms) = body
            .split_once(':')
            .ok_or_else(|| format!("{s}: expected k:m1+m2+..."))?;
        let num = |v: &str| {
            v.trim()
                .parse::<usize>()
                .map_err(|e| format!("{s}: {v}: {e}"))
        };
        let g = Self {
            k: num(k)?,
            ms: ms.split('+').map(num).collect::<Result<_, _>>()?,
            f32,
        };
        // Chain weights (16 or 48 rows, BF16) only after the others.
        let main = g.ms.iter().take_while(|&&m| m.is_multiple_of(32)).count();
        let chain_ok = |m: usize| !g.f32 && m % 32 == 16 && m <= CHAIN_MAX_ROWS as usize;
        if g.k == 0
            || !g.k.is_multiple_of(64)
            || main == 0
            || g.ms.contains(&0)
            || !g.ms[main..].iter().all(|&m| chain_ok(m))
        {
            return Err(format!(
                "{s}: k a multiple of 64, rows of 32 (then BF16 chain weights of 16 or 48)"
            ));
        }
        if g.ms.len() > MAX_WEIGHTS {
            return Err(format!("{s}: at most {MAX_WEIGHTS} weights"));
        }
        Ok(g)
    }

    /// Whether the op takes the group at `n` tokens (F32 weights from [`MIN_TOKENS_F32`], groups
    /// with chain weights from [`CHAIN_MIN_TOKENS`]).
    pub fn takes(&self, n: usize) -> bool {
        let min = match (self.f32, self.chain() > 0) {
            (true, _) => MIN_TOKENS_F32,
            (false, true) => CHAIN_MIN_TOKENS,
            (false, false) => MIN_TOKENS,
        };
        (min as usize..=MAX_TOKENS as usize).contains(&n)
    }

    /// The chain weights at the end ([`super::fits_chain`]: rows not a multiple of 32), which
    /// run only in the group's op.
    pub fn chain(&self) -> usize {
        self.ms.iter().filter(|&&m| !m.is_multiple_of(32)).count()
    }

    /// The weights' type.
    pub fn ty(&self) -> ffi::ggml_type {
        if self.f32 { types::F32 } else { types::BF16 }
    }

    pub fn bytes(&self) -> usize {
        let elem = if self.f32 { 4 } else { 2 };
        self.ms.iter().map(|m| m * self.k * elem).sum()
    }
}

impl std::fmt::Display for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ms: Vec<String> = self.ms.iter().map(|m| m.to_string()).collect();
        let ty = if self.f32 { "f32:" } else { "" };
        write!(f, "{ty}{}:{}", self.k, ms.join("+"))
    }
}

/// How a variant builds the group's matmuls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// ggml's `mul_mat` per weight.
    Ggml,
    /// One op per weight.
    Op,
    /// One op for the group.
    Joined,
}

/// One variant: the build and the kernel variant forced (`None`: the default choice).
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub name: &'static str,
    pub kind: Kind,
    pub force: Option<(i32, i32)>,
}

const fn v(name: &'static str, kind: Kind, force: Option<(i32, i32)>) -> Variant {
    Variant { name, kind, force }
}

/// The variants, the reference (ggml) first.
pub const VARIANTS: [Variant; 13] = [
    v("ggml", Kind::Ggml, None),
    v("op", Kind::Op, None),
    v("op t1d2", Kind::Op, Some((1, 2))),
    v("op t1d4", Kind::Op, Some((1, 4))),
    v("op t1d8", Kind::Op, Some((1, 8))),
    v("op t2d2", Kind::Op, Some((2, 2))),
    v("op t2d4", Kind::Op, Some((2, 4))),
    v("join", Kind::Joined, None),
    v("join t1d2", Kind::Joined, Some((1, 2))),
    v("join t1d4", Kind::Joined, Some((1, 4))),
    v("join t1d8", Kind::Joined, Some((1, 8))),
    v("join t2d2", Kind::Joined, Some((2, 2))),
    v("join t2d4", Kind::Joined, Some((2, 4))),
];

/// What [`run`] measures.
#[derive(Clone, Debug)]
pub struct Config {
    pub groups: Vec<Group>,
    /// Tokens per step.
    pub toks: Vec<usize>,
    pub rounds: usize,
    /// Weight bytes a timed graph reads at least (copies of the group, at most [`MAX_COPIES`]).
    pub min_bytes: usize,
}

/// Most copies of a group's weights a graph reads.
pub const MAX_COPIES: usize = 64;

impl Default for Config {
    /// Qwen3.8-Flash-Next's BF16 matmuls of an NVFP4 checkpoint: DeltaNet `in_proj_qkv`,
    /// `in_proj_z` and both (also with `in_proj_b` and `in_proj_a`), attention `q_proj` and q/k/v, the output projections, the LM head,
    /// k (or v) alone, the shared expert's gate and up, its down, and the MoE router (F32).
    fn default() -> Self {
        let g = |k: usize, ms: &[usize]| Group {
            k,
            ms: ms.to_vec(),
            f32: false,
        };
        Self {
            groups: vec![
                g(2560, &[10240]),
                g(2560, &[6144]),
                g(2560, &[10240, 6144]),
                g(2560, &[10240, 6144, 48, 48]),
                g(2560, &[12288]),
                g(2560, &[12288, 512, 512]),
                g(6144, &[2560]),
                g(2560, &[248320]),
                g(2560, &[512]),
                g(2560, &[640, 640]),
                g(640, &[2560]),
                Group {
                    k: 2560,
                    ms: vec![512],
                    f32: true,
                },
            ],
            toks: vec![4, 16],
            rounds: 5,
            min_bytes: 512 << 20,
        }
    }
}

/// One variant of one group at one step.
#[derive(Clone, Debug)]
pub struct Row {
    pub variant: &'static str,
    pub group: String,
    pub n_tok: usize,
    /// Best mean time per copy of the group over the rounds (seconds).
    pub secs: f64,
    /// Weight bytes of the group.
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

/// `bytes` of BF16 (or F32) weights: a block of noise repeated, shifted by `seed`.
fn weights(bytes: usize, seed: u64, f32: bool) -> Vec<u8> {
    let block: Vec<u8> = noise(1 << 20, seed)
        .iter()
        .flat_map(|v| {
            if f32 {
                v.to_ne_bytes().to_vec()
            } else {
                ((v.to_bits() >> 16) as u16).to_ne_bytes().to_vec()
            }
        })
        .collect();
    block.iter().copied().cycle().take(bytes).collect()
}

/// The graph of `copies` groups of `kind` on the weights `ws` (`ms.len()` per copy): the input,
/// and the graph (each copy's results contiguous outputs).
fn build<'a>(
    ctx: &'a Context,
    kind: Kind,
    ws: &[Tensor<'a>],
    k: usize,
    per: usize,
    n_tok: usize,
) -> (Tensor<'a>, Graph<'a>) {
    let x = ctx
        .new_tensor(types::F32, &[k as i64, n_tok as i64])
        .as_input();
    let mut graph = ctx.new_graph(8 * ws.len() + 16);
    for copy in ws.chunks(per) {
        let ys = match kind {
            Kind::Ggml => copy.iter().map(|&w| ctx.mul_mat(w, x)).collect(),
            // A chain weight runs only beside others: on ggml alone.
            Kind::Op => copy
                .iter()
                .map(|&w| match w.ne()[1] % 32 {
                    0 => mul_mat(ctx, &[w], x)[0],
                    _ => ctx.mul_mat(w, x),
                })
                .collect(),
            Kind::Joined => mul_mat(ctx, copy, x),
        };
        for y in ys {
            let y = ctx.cont(y).as_output();
            graph.expand(y);
        }
    }
    (x, graph)
}

/// Every variant for every group and step size of `cfg` on `gpu`; the variants take turns within
/// a round, the best round counts. The kernel choice is back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for group in &cfg.groups {
        let (k, per) = (group.k, group.ms.len());
        let copies = cfg.min_bytes.div_ceil(group.bytes()).clamp(1, MAX_COPIES);
        let weights_ctx = Context::new(per * copies + 1, 0);
        let ws: Vec<Tensor<'_>> = (0..copies)
            .flat_map(|_| group.ms.iter())
            .map(|&m| weights_ctx.new_tensor(group.ty(), &[k as i64, m as i64]))
            .collect();
        weights_ctx
            .alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(w.nbytes(), 3 + i as u64, group.f32));
        }
        for &n_tok in cfg.toks.iter().filter(|&&n| group.takes(n)) {
            let kinds = [Kind::Ggml, Kind::Op, Kind::Joined];
            let ctxs = kinds.map(|_| Context::new(16 * ws.len() + 16, 8 * ws.len() + 16));
            let graphs: Vec<_> = kinds
                .iter()
                .zip(&ctxs)
                .map(|(&kind, ctx)| build(ctx, kind, &ws, k, per, n_tok))
                .collect();
            let xs = noise(k * n_tok, 5 + n_tok as u64);
            for (ctx, (x, graph)) in ctxs.iter().zip(&graphs) {
                ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                x.set_f32(&xs);
            }
            // Joining needs two weights; a variant without a kernel or that fails drops out.
            let mut live: [bool; VARIANTS.len()] =
                std::array::from_fn(|i| VARIANTS[i].kind != Kind::Joined || per > 1);
            // A warm-up run of each variant, then the rounds.
            let order: Vec<usize> = (0..VARIANTS.len())
                .chain((0..cfg.rounds).flat_map(|_| 0..VARIANTS.len()))
                .collect();
            let mut best = [f64::MAX; VARIANTS.len()];
            for (i, &vi) in order.iter().enumerate() {
                let var = &VARIANTS[vi];
                if !live[vi] {
                    continue;
                }
                let kind_i = kinds.iter().position(|&k| k == var.kind).unwrap_or(0);
                let (_, graph) = &graphs[kind_i];
                if !force_variant(var.force) {
                    live[vi] = false;
                    continue;
                }
                let t = std::time::Instant::now();
                let r = gpu.compute(graph);
                let secs = t.elapsed().as_secs_f64();
                force_variant(None);
                if let Err(e) = r {
                    if vi == 0 {
                        return Err(format!("{group} {}: {e}", var.name));
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
                    rows.push(Row {
                        variant: var.name,
                        group: group.to_string(),
                        n_tok,
                        secs: best[vi],
                        bytes: group.bytes() as f64,
                    });
                }
            }
        }
    }
    Ok(rows)
}

/// One group at one step in [`check`]: outputs whose bits differ from ggml's, over every
/// variant (op per weight and joined), and the variants that ran.
#[derive(Clone, Debug)]
pub struct Check {
    pub group: String,
    pub n_tok: usize,
    pub differing: usize,
    pub variants: usize,
}

/// The bits of every op variant against ggml's `mul_mat` in one graph, for every group and step
/// size of `cfg` on `gpu` (the ignored test `cuda_matches_ggml_bits` for a release binary).
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    for group in &cfg.groups {
        let k = group.k as i64;
        for &n_tok in cfg.toks.iter().filter(|&&n| group.takes(n)) {
            let ctx = Context::new(64, 64);
            let x = ctx.new_tensor(types::F32, &[k, n_tok as i64]).as_input();
            let ws: Vec<_> = group
                .ms
                .iter()
                .map(|&m| ctx.new_tensor(group.ty(), &[k, m as i64]).as_input())
                .collect();
            let refs: Vec<_> = ws.iter().map(|&w| ctx.mul_mat(w, x).as_output()).collect();
            // (op output, index of its reference); chain weights only in the group's op.
            let mut ops: Vec<_> = ws
                .iter()
                .enumerate()
                .filter(|(_, w)| w.ne()[1] % 32 == 0)
                .map(|(i, &w)| (ctx.cont(mul_mat(&ctx, &[w], x)[0]).as_output(), i))
                .collect();
            if ws.len() > 1 {
                ops.extend(
                    mul_mat(&ctx, &ws, x)
                        .into_iter()
                        .enumerate()
                        .map(|(i, o)| (ctx.cont(o).as_output(), i)),
                );
            }
            let mut graph = ctx.new_graph(64);
            for &t in refs.iter().chain(ops.iter().map(|(o, _)| o)) {
                graph.expand(t);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            x.set_f32(&noise(group.k * n_tok, 1000 + n_tok as u64));
            for (i, w) in ws.iter().enumerate() {
                w.set_bytes(&weights(w.nbytes(), 2000 + i as u64, group.f32));
            }
            let (mut differing, mut variants) = (0, 0);
            for var in VARIANTS.iter().filter(|v| v.kind == Kind::Op) {
                if !force_variant(var.force) {
                    continue;
                }
                let r = gpu.compute(&graph);
                force_variant(None);
                r.map_err(|e| format!("{group} {}: {e}", var.name))?;
                variants += 1;
                for (o, i) in &ops {
                    let (a, b) = (o.to_f32(), refs[*i].to_f32());
                    differing += a
                        .iter()
                        .zip(&b)
                        .filter(|(p, q)| p.to_bits() != q.to_bits())
                        .count();
                }
            }
            out.push(Check {
                group: group.to_string(),
                n_tok,
                differing,
                variants,
            });
        }
    }
    Ok(out)
}

/// The bits of wide chains ([`mul_mat_wide_chain`]) against ggml's `mul_mat` in one graph: every
/// BF16 weight shape `k:m` of `cfg`'s groups that [`fits_wide_chain`], at every step size of
/// `cfg` from [`MAX_TOKENS`] + 1 to [`WIDE_CHAIN_MAX_TOKENS`] tokens (#836: the shapes cuBLAS
/// runs without split-K, whose bits a wide chain gives, [`super::wide_chain_bitwise`]), each
/// also timed alone against ggml's; also each group of `cfg` whose weights could all take wide
/// chains, as one launch ([`mul_mat_wide_chains`]) against ggml's matmul per weight. One
/// [`WideCheck`] per shape (or group) and step.
pub fn check_wide(gpu: &mut Backend, cfg: &Config) -> Result<Vec<WideCheck>, String> {
    // Each weight shape alone, then each group of 2 or more weights that could all take wide
    // chains (16 MiB or more each) as one launch ([`mul_mat_wide_chains`]).
    let mut cases: Vec<(usize, Vec<usize>)> = Vec::new();
    for g in cfg.groups.iter().filter(|g| !g.f32) {
        for &m in &g.ms {
            if !cases.contains(&(g.k, vec![m])) {
                cases.push((g.k, vec![m]));
            }
        }
    }
    for g in cfg.groups.iter().filter(|g| !g.f32) {
        let joined = (2..=MAX_WEIGHTS).contains(&g.ms.len())
            && g.ms.iter().all(|&m| g.k * m * 2 >= MIN_BYTES)
            && !cases.contains(&(g.k, g.ms.clone()));
        if joined {
            cases.push((g.k, g.ms.clone()));
        }
    }
    let wide = |n: &&usize| (MAX_TOKENS + 1..=WIDE_CHAIN_MAX_TOKENS).contains(&(**n as i64));
    let mut out = Vec::new();
    for (k, ms) in cases {
        let label = format!(
            "{k}:{}",
            ms.iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join("+")
        );
        for &n_tok in cfg.toks.iter().filter(wide) {
            let ctx = Context::new(64, 16);
            let x = ctx
                .new_tensor(types::F32, &[k as i64, n_tok as i64])
                .as_input();
            let ws: Vec<_> = ms
                .iter()
                .map(|&m| {
                    ctx.new_tensor(types::BF16, &[k as i64, m as i64])
                        .as_input()
                })
                .collect();
            let order = |w: Tensor<'_>| {
                let [wk, wm, _, _] = w.ne();
                wide_chain_order(wk, wm, n_tok as i64).unwrap_or_default()
            };
            if !ws
                .iter()
                .all(|&w| fits_wide_chain(w, x) && order(w) == order(ws[0]))
            {
                continue;
            }
            let references: Vec<_> = ws.iter().map(|&w| ctx.mul_mat(w, x).as_output()).collect();
            let ops: Vec<_> = if let [w] = ws[..] {
                vec![ctx.cont(mul_mat_wide_chain(&ctx, w, x)).as_output()]
            } else {
                mul_mat_wide_chains(&ctx, &ws, x)
                    .into_iter()
                    .map(|t| ctx.cont(t).as_output())
                    .collect()
            };
            let mut graph = ctx.new_graph(64);
            for &t in references.iter().chain(&ops) {
                graph.expand(t);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            x.set_f32(&noise(k * n_tok, 3000 + n_tok as u64));
            for (&w, &m) in ws.iter().zip(&ms) {
                w.set_bytes(&weights(w.nbytes(), 4000 + m as u64, false));
            }
            let err = |e| format!("{label} wide chain at {n_tok}: {e}");
            gpu.compute(&graph).map_err(err)?;
            let differing = ops
                .iter()
                .zip(&references)
                .map(|(op, r)| {
                    op.to_f32()
                        .iter()
                        .zip(&r.to_f32())
                        .filter(|(p, q)| p.to_bits() != q.to_bits())
                        .count()
                })
                .sum();
            // Each alone, taking turns: the best of `cfg.rounds` (one warm-up each first).
            let (mut ggml_graph, mut op_graph) = (ctx.new_graph(64), ctx.new_graph(64));
            for &r in &references {
                ggml_graph.expand(r);
            }
            for &op in &ops {
                op_graph.expand(op);
            }
            let mut best = [f64::MAX; 2];
            for round in 0..=cfg.rounds {
                for (i, g) in [&ggml_graph, &op_graph].into_iter().enumerate() {
                    let t = std::time::Instant::now();
                    gpu.compute(g).map_err(err)?;
                    if round > 0 {
                        best[i] = best[i].min(t.elapsed().as_secs_f64());
                    }
                }
            }
            out.push(WideCheck {
                shape: label.clone(),
                n_tok,
                differing,
                ggml_secs: best[0],
                op_secs: best[1],
            });
        }
    }
    Ok(out)
}

/// The orders [`probe_wide`] tries for a row of `k` values: slices of 16 to `k` values added in
/// order; 2-8 accumulators taking groups of steps in turn, added in order, pairwise or last to
/// first; and m16n8k8 products.
pub fn probe_orders(k: usize) -> Vec<WideOrder> {
    let mut out = Vec::new();
    let mut add = |o: WideOrder| {
        if !out.contains(&o) {
            out.push(o);
        }
    };
    // One accumulator: every slice length of whole steps up to `k`, slices added in order.
    for s in (16..=k).step_by(16) {
        add(WideOrder {
            kslice: s as i32,
            kinter: 1,
            kgroup: 16,
            ..WideOrder::default()
        });
    }
    // 2-8 accumulators over the whole row (contiguous slices or interleaved groups of steps),
    // added in order, pairwise or last to first.
    for inter in 2..=8usize {
        for group in (16..=k / inter).step_by(16) {
            if group % 64 != 0 && group > 64 {
                continue;
            }
            for kred in [0, 1, 2] {
                if kred == 1 && !inter.is_power_of_two() {
                    continue;
                }
                add(WideOrder {
                    kslice: k as i32,
                    kinter: inter as i32,
                    kgroup: group as i32,
                    kred,
                    ..WideOrder::default()
                });
            }
        }
    }
    // Interleaved groups within slices, the slices added in order; m16n8k8 products.
    for s in [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, k] {
        for inter in [2usize, 4, 8] {
            for group in [16usize, 32, 64, 128] {
                if group * inter <= s && s <= k {
                    add(WideOrder {
                        kslice: s as i32,
                        kinter: inter as i32,
                        kgroup: group as i32,
                        ..WideOrder::default()
                    });
                }
            }
        }
        if s <= k {
            add(WideOrder {
                kslice: s as i32,
                kinter: 1,
                kgroup: 16,
                kmma: 8,
                ..WideOrder::default()
            });
        }
    }
    out
}

/// One shape's [`probe_wide`] result: its name, its wide step size, the orders that match ggml's
/// bits, and the fewest differing outputs of any order.
pub type WideProbe = (String, usize, Vec<WideOrder>, usize);

/// Every order of [`probe_orders`] for every BF16 weight shape and wide step size of `cfg` (as
/// [`check_wide`]), in one graph with ggml's `mul_mat`: the orders with ggml's bits (#836: which
/// order cuBLAS's kernel sums in), and the fewest differing outputs of any order.
pub fn probe_wide(gpu: &mut Backend, cfg: &Config) -> Result<Vec<WideProbe>, String> {
    let mut shapes: Vec<(usize, usize)> = Vec::new();
    for g in cfg.groups.iter().filter(|g| !g.f32) {
        for &m in &g.ms {
            if !shapes.contains(&(g.k, m)) {
                shapes.push((g.k, m));
            }
        }
    }
    let wide = |n: &&usize| (MAX_TOKENS + 1..=WIDE_CHAIN_MAX_TOKENS).contains(&(**n as i64));
    let mut out = Vec::new();
    for (k, m) in shapes {
        let orders = probe_orders(k);
        for &n_tok in cfg.toks.iter().filter(wide) {
            let ctx = Context::new(16 + orders.len() * 2, 16 + orders.len() * 2);
            let x = ctx
                .new_tensor(types::F32, &[k as i64, n_tok as i64])
                .as_input();
            let w = ctx
                .new_tensor(types::BF16, &[k as i64, m as i64])
                .as_input();
            if !fits_wide_chain(w, x) {
                continue;
            }
            let reference = ctx.mul_mat(w, x).as_output();
            let ops: Vec<_> = orders
                .iter()
                .map(|&o| {
                    ctx.cont(mul_mat_wide_chain_order(&ctx, w, x, o))
                        .as_output()
                })
                .collect();
            let mut graph = ctx.new_graph(16 + orders.len() * 2);
            graph.expand(reference);
            for &o in &ops {
                graph.expand(o);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            x.set_f32(&noise(k * n_tok, 3000 + n_tok as u64));
            w.set_bytes(&weights(w.nbytes(), 4000 + m as u64, false));
            gpu.compute(&graph)
                .map_err(|e| format!("{k}:{m} probe at {n_tok}: {e}"))?;
            let b = reference.to_f32();
            let mut hits = Vec::new();
            let mut fewest = usize::MAX;
            for (o, t) in orders.iter().zip(&ops) {
                let d = t
                    .to_f32()
                    .iter()
                    .zip(&b)
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count();
                fewest = fewest.min(d);
                if d == 0 {
                    hits.push(*o);
                }
                // One chain: how far from ggml's bits, per token and in ulps.
                if o.kslice as usize == k && o.kinter == 1 && o.kmma != 8 {
                    let a = t.to_f32();
                    let mut ulps: Vec<i64> = a
                        .iter()
                        .zip(&b)
                        .map(|(p, q)| (p.to_bits() as i32 as i64 - q.to_bits() as i32 as i64).abs())
                        .collect();
                    let per_tok: Vec<usize> = (0..n_tok)
                        .map(|j| {
                            (0..m)
                                .filter(|&i| a[j * m + i].to_bits() != b[j * m + i].to_bits())
                                .count()
                        })
                        .collect();
                    ulps.sort_unstable();
                    let q = |f: f64| ulps[((ulps.len() - 1) as f64 * f) as usize];
                    eprintln!(
                        "{k}:{m} n={n_tok} one chain: ulps p50 {} p90 {} p99 {} max {}; differing per token {per_tok:?}",
                        q(0.5),
                        q(0.9),
                        q(0.99),
                        q(1.0)
                    );
                }
            }
            out.push((format!("{k}:{m}"), n_tok, hits, fewest));
        }
    }
    Ok(out)
}

/// One weight shape at one step in [`check_wide`]: outputs whose bits differ from ggml's, and
/// the best time of each alone (one graph launch and its wait included).
#[derive(Clone, Debug)]
pub struct WideCheck {
    pub shape: String,
    pub n_tok: usize,
    pub differing: usize,
    pub ggml_secs: f64,
    pub op_secs: f64,
}

#[cfg(test)]
mod tests;

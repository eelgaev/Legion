//! qwen4exp's mixer up matmul and gated mix with BF16 up weights (the NVFP4 checkpoints) at a
//! prompt step's shape, timed on one GPU in each of its kernels (#1183): `up_mix_bf16` (1), the
//! default (0) and `up_mix_bf16_reg` (2, [`super::force_kernel`]), each with given
//! `up_mix_bf16_reg` options ([`super::force_reg_opts`]). Each timed graph runs
//! [`Config::copies`] mixers, each with an `up` weight of its own, over the same streams (far
//! larger than L2). [`check`] compares every variant's bits with [`Config::reference`]'s, [`ab`]
//! times variants against a base one call by call. Used by the hidden `legion hc-up-mix-bench`
//! subcommand (a build tree's binary on the GPU box).

use super::{
    REG_ALL, REG_CROSS, REG_LD, REG_PD2, REG_XS, force_kernel, force_reg_opts, last_kernel,
    last_reg_opts, set_lazy_streams, set_mix_cast, up_mix, up_mix_rebuilt,
};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, types};
use crate::model::nvfp4_mm::bf16_bits;

/// A kernel ([`super::force_kernel`]'s numbers) with `up_mix_bf16_reg` options ([`REG_ALL`]
/// bits; `None`: the configured ones, [`super::DEFAULT_REG_OPTS`] unless a switch changed them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    pub kernel: i32,
    pub opts: Option<i32>,
}

/// The option names [`Variant::parse`] takes and [`Variant`]'s `Display` writes.
const OPT_NAMES: [(&str, i32); 4] = [
    ("pd2", REG_PD2),
    ("xs", REG_XS),
    ("ld", REG_LD),
    ("cross", REG_CROSS),
];

impl Variant {
    pub const fn new(kernel: i32, opts: Option<i32>) -> Self {
        Self { kernel, opts }
    }

    /// `K` (kernel K, the configured options) or `K:OPTS`, OPTS `none`, a number or option
    /// names joined by `+` (`pd2`, `xs`, `ld`, `cross`): `2:ld+xs`, `1002:none`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let (k, o) = match s.split_once(':') {
            Some((k, o)) => (k, Some(o)),
            None => (s, None),
        };
        let kernel = k
            .trim()
            .parse::<i32>()
            .map_err(|_| format!("kernel {k:?} in {s:?} is not a number"))?;
        let opts = match o.map(str::trim) {
            None => None,
            Some("none") => Some(0),
            Some(o) => Some(match o.parse::<i32>() {
                Ok(v) if v & !REG_ALL == 0 && v >= 0 => v,
                Ok(_) => return Err(format!("options {o:?} in {s:?}: bits of {REG_ALL} only")),
                Err(_) => o.split('+').try_fold(0, |acc, name| {
                    OPT_NAMES
                        .iter()
                        .find(|(n, _)| *n == name)
                        .map(|&(_, bit)| acc | bit)
                        .ok_or_else(|| {
                            format!("option {name:?} in {s:?}: none, pd2, xs, ld or cross")
                        })
                })?,
            }),
        };
        Ok(Self { kernel, opts })
    }

    /// Whether a launch that ran `kernel` ([`super::last_kernel`]) with `opts`
    /// ([`super::last_reg_opts`]) ran this variant: its kernel (any for 0) and, for
    /// `up_mix_bf16_reg` with options given, those options (not some fallback).
    pub fn ran_as(self, kernel: i32, opts: i32) -> bool {
        (self.kernel == 0 || kernel == self.kernel)
            && (self.kernel % 100 != 2 || self.opts.is_none_or(|o| o == opts))
    }

    /// Forces this variant for the launches from now on.
    fn force(self) {
        force_kernel(self.kernel);
        force_reg_opts(self.opts);
    }
}

/// Back to the default kernel and the configured options.
fn unforce() {
    force_kernel(0);
    force_reg_opts(None);
}

/// Option bits as [`Variant::parse`] reads them (`none` for 0).
pub fn opts_name(opts: i32) -> String {
    if opts == 0 {
        return "none".into();
    }
    OPT_NAMES
        .iter()
        .filter(|(_, bit)| opts & bit != 0)
        .map(|(n, _)| *n)
        .collect::<Vec<_>>()
        .join("+")
}

impl std::fmt::Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.opts {
            None => write!(f, "{}", self.kernel),
            Some(o) => write!(f, "{}:{}", self.kernel, opts_name(o)),
        }
    }
}

/// What [`run`], [`check`] and [`ab`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// Values per stream (qwen4exp: 2,560 in 4 streams).
    pub e: i64,
    /// The low rank (qwen4exp: 320).
    pub rank: i64,
    /// Tokens of the step.
    pub tokens: Vec<i64>,
    /// Variants timed or checked.
    pub variants: Vec<Variant>,
    /// What [`check`] compares every variant with.
    pub reference: Variant,
    /// Mixers a timed graph runs, each with its own `up` weight.
    pub copies: usize,
    pub rounds: usize,
}

impl Default for Config {
    /// Qwen3.8-Flash-Next's mixer at prompt steps of 2K, 4K and 8K tokens and a tail step's
    /// 3,971; `up_mix_bf16`, the default and `up_mix_bf16_reg`, checked against `up_mix_bf16`.
    fn default() -> Self {
        Self {
            e: 2560,
            rank: 320,
            tokens: vec![2048, 4096, 8192, 3971],
            variants: [1, 0, 2].map(|k| Variant::new(k, None)).to_vec(),
            reference: Variant::new(1, None),
            copies: 8,
            rounds: 8,
        }
    }
}

/// The mixer's forms: F32 streams writing the F32 values and their BF16 copy (`f32`), F32
/// streams writing the copy only (`bf16`: the node fusion's mix cast), BF16 streams writing the
/// copy only (`s16`: `legion serve --hc-bf16`), `xn` read (`xn`, F32 values only), lazy F32
/// streams (`pre`: the attention block's output combined in first, F32 values and copy).
pub const MODES: [&str; 5] = ["f32", "bf16", "s16", "xn", "pre"];

/// One variant in one mode at one token count.
#[derive(Clone, Debug)]
pub struct Row {
    pub mode: &'static str,
    pub variant: Variant,
    /// The kernel that ran ([`super::last_kernel`]) and its options ([`super::last_reg_opts`]).
    pub ran: i32,
    pub ran_opts: i32,
    pub tokens: i64,
    /// Best mean time per mixer over the rounds (seconds), `lo`'s BF16 rounding included.
    pub secs: f64,
    /// The bytes a mixer must move (streams, scales, `lo` as BF16, the outputs).
    pub bytes: f64,
}

/// One comparison of [`check`].
#[derive(Clone, Debug)]
pub struct Check {
    pub mode: &'static str,
    pub variant: Variant,
    pub ran: i32,
    pub ran_opts: i32,
    pub tokens: i64,
    pub values: usize,
    /// Values whose bits differ from the reference's (F32 values and BF16 copy together).
    pub differing: usize,
    /// FNV-1a of the outputs' bytes.
    pub hash: u64,
}

/// One pairing of [`ab`]: a variant against the base, per mixer.
#[derive(Clone, Debug)]
pub struct AbRow {
    pub mode: &'static str,
    pub tokens: i64,
    pub base: Variant,
    pub variant: Variant,
    /// What ran for the base and the variant: kernel and options.
    pub ran: [(i32, i32); 2],
    /// Seconds per mixer over the timed rounds, base then variant: median and min.
    pub median: [f64; 2],
    pub min: [f64; 2],
    pub bytes: f64,
}

impl AbRow {
    /// Base time over the variant's (> 1: the variant is faster), medians.
    pub fn speedup(&self) -> f64 {
        self.median[0] / self.median[1]
    }

    /// The same of the minimums.
    pub fn speedup_min(&self) -> f64 {
        self.min[0] / self.min[1]
    }
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

fn bf16_bytes(x: impl Iterator<Item = f32>) -> Vec<u8> {
    x.flat_map(|v| bf16_bits(v).to_ne_bytes()).collect()
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// The median of `x` (the mean of the middle two for an even count).
fn median(x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => v[n / 2],
        n => 0.5 * (v[n / 2 - 1] + v[n / 2]),
    }
}

/// The bytes one mixer of `cfg` moves at `n` tokens in `mode`.
fn bytes(cfg: &Config, mode: &str, n: i64) -> f64 {
    let (e, n) = (cfg.e as f64, n as f64);
    let streams = 4.0 * e * if mode == "s16" { 2.0 } else { 4.0 };
    let scales = if mode == "xn" { 0.0 } else { 16.0 };
    // The lazy streams' block output and inject weights.
    let pre = if mode == "pre" { 4.0 * e + 16.0 } else { 0.0 };
    let out = match mode {
        "f32" | "pre" => 6.0 * e,
        "xn" => 4.0 * e,
        _ => 2.0 * e,
    };
    n * (streams + scales + pre + 2.0 * cfg.rank as f64 + out)
}

/// A step's inputs (also marked as outputs, so ggml's allocator keeps their memory).
struct Inputs<'a> {
    lo: Tensor<'a>,
    res: Tensor<'a>,
    sc: Tensor<'a>,
    g: Tensor<'a>,
}

fn inputs<'a>(ctx: &'a Context, cfg: &Config, mode: &str, n: i64) -> Inputs<'a> {
    let k = 4 * cfg.e;
    let res = match mode {
        "s16" => ctx.new_tensor(types::BF16, &[cfg.e, 4, n]),
        "xn" => ctx.new_tensor(types::F32, &[k, n]),
        _ => ctx.new_tensor(types::F32, &[cfg.e, 4, n]),
    };
    Inputs {
        lo: ctx
            .new_tensor(types::F32, &[cfg.rank, n])
            .as_input()
            .as_output(),
        res: res.as_input().as_output(),
        sc: ctx.new_tensor(types::F32, &[4, n]).as_input().as_output(),
        g: ctx.new_tensor(types::F32, &[k]).as_input().as_output(),
    }
}

fn fill(i: &Inputs<'_>, w: &Weights<'_>, cfg: &Config, mode: &str, n: i64) {
    let (k, nu) = ((4 * cfg.e) as usize, n as usize);
    // lo as a SiLU output would be: mostly small, some larger.
    i.lo.set_f32(
        &noise(cfg.rank as usize * nu, 81)
            .iter()
            .map(|v| v * 1.5)
            .collect::<Vec<_>>(),
    );
    let r = noise(k * nu, 82).into_iter().map(|v| 4.0 * v);
    if mode == "s16" {
        i.res.set_bytes(&bf16_bytes(r));
    } else {
        i.res.set_f32(&r.collect::<Vec<_>>());
    }
    if mode == "xn" {
        // No scales or gamma in the graph (not allocated).
        return;
    }
    i.sc.set_f32(
        &noise(4 * nu, 83)
            .iter()
            .map(|v| v.abs() + 0.5)
            .collect::<Vec<_>>(),
    );
    i.g.set_f32(
        &noise(k, 84)
            .iter()
            .map(|v| 1.0 + 0.2 * v)
            .collect::<Vec<_>>(),
    );
    if mode == "pre" {
        w.pre_out.set_f32(&noise(cfg.e as usize * nu, 85));
        w.pre_inj.set_f32(
            &noise(4 * nu, 86)
                .iter()
                .map(|v| 3.0 * v)
                .collect::<Vec<_>>(),
        );
    }
}

/// `copies` BF16 `up` weights (`[rank, 4 · e]`), the BF16 copies' outputs (`[e, n]`) and the
/// lazy streams' inputs (`pre` only): the attention block's output (`[e, n]`) and its inject
/// weights (`[4, n]`).
struct Weights<'a> {
    ups: Vec<Tensor<'a>>,
    casts: Vec<Tensor<'a>>,
    pre_out: Tensor<'a>,
    pre_inj: Tensor<'a>,
}

fn weights<'a>(
    ctx: &'a Context,
    gpu: &Backend,
    cfg: &Config,
    n: i64,
) -> Result<Weights<'a>, String> {
    let ups: Vec<Tensor<'a>> = (0..cfg.copies)
        .map(|_| ctx.new_tensor(types::BF16, &[cfg.rank, 4 * cfg.e]))
        .collect();
    let casts: Vec<Tensor<'a>> = (0..cfg.copies)
        .map(|_| ctx.new_tensor(types::BF16, &[cfg.e, n]))
        .collect();
    let pre_out = ctx.new_tensor(types::F32, &[cfg.e, n]);
    let pre_inj = ctx.new_tensor(types::F32, &[4, n]);
    ctx.alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for (j, up) in ups.iter().enumerate() {
        up.set_bytes(&bf16_bytes(
            noise((cfg.rank * 4 * cfg.e) as usize, 100 + j as u64)
                .into_iter()
                .map(|v| 0.25 * v),
        ));
    }
    Ok(Weights {
        ups,
        casts,
        pre_out,
        pre_inj,
    })
}

/// A graph of one mixer per weight in `mode`, allocated, its BF16 copies (and lazy streams) set;
/// the mixers.
fn build<'a>(
    ctx: &'a Context,
    gpu: &Backend,
    w: &Weights<'a>,
    i: &Inputs<'a>,
    mode: &str,
) -> Result<(Graph<'a>, Vec<Tensor<'a>>), String> {
    let mut graph = ctx.new_graph(8 * w.ups.len() + 16);
    let mut outs = Vec::new();
    for &up in &w.ups {
        let y = if mode == "xn" {
            up_mix(ctx, up, i.lo, i.res, 4, 0.25)
        } else {
            up_mix_rebuilt(ctx, up, i.lo, i.res, i.sc, i.g, 4, 0.25)
        };
        let y = y.as_output();
        graph.expand(y);
        outs.push(y);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .map_err(|e| e.to_string())?;
    if mode != "xn" {
        for (y, c) in outs.iter().zip(&w.casts) {
            // SAFETY: nodes of the graph just allocated, which has not run; the copy holds as
            // many BF16 values as the mixer has F32 ones.
            unsafe { set_mix_cast(y.raw(), c.raw(), mode == "bf16" || mode == "s16") };
            if mode == "pre" {
                // SAFETY: as above; the streams F32 `[e, 4, n]`, the block output `[e, n]` and the
                // inject weights `[4, n]`, contiguous, allocated on the same GPU.
                unsafe { set_lazy_streams(y.raw(), i.res.raw(), w.pre_out.raw(), w.pre_inj.raw()) };
            }
        }
    }
    Ok((graph, outs))
}

fn check_config(cfg: &Config) -> Result<(), String> {
    if cfg.copies == 0
        || cfg.rounds == 0
        || cfg.e % 64 != 0
        || cfg.rank % 16 != 0
        || cfg.tokens.iter().any(|&n| n < 1)
    {
        return Err(format!(
            "copies > 0, rounds > 0, e a multiple of 64, rank of 16, tokens > 0: {cfg:?}"
        ));
    }
    Ok(())
}

/// One timed compute of `graph` with `v` forced: seconds per mixer and what ran.
fn time(
    gpu: &mut Backend,
    graph: &Graph<'_>,
    v: Variant,
    copies: usize,
    what: &str,
) -> Result<(f64, (i32, i32)), String> {
    v.force();
    let t = std::time::Instant::now();
    let r = gpu.compute(graph);
    let secs = t.elapsed().as_secs_f64();
    unforce();
    r.map_err(|e| format!("{what} {v}: {e}"))?;
    Ok((secs / copies as f64, (last_kernel(), last_reg_opts())))
}

/// Every variant of `cfg` in every mode of `modes` at every token count on `gpu`; the variants
/// take turns within a round, the best round counts. The default kernel and options are back
/// afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config, modes: &[&'static str]) -> Result<Vec<Row>, String> {
    check_config(cfg)?;
    let mut rows = Vec::new();
    for &n in &cfg.tokens {
        let wctx = Context::new(2 * cfg.copies + 4, 0);
        let w = weights(&wctx, gpu, cfg, n)?;
        for &mode in modes {
            let ctx = Context::new(16 * cfg.copies + 16, 8 * cfg.copies + 16);
            let i = inputs(&ctx, cfg, mode, n);
            let (graph, _) = build(&ctx, gpu, &w, &i, mode)?;
            fill(&i, &w, cfg, mode, n);
            let nv = cfg.variants.len();
            let mut best = vec![f64::MAX; nv];
            let mut ran = vec![(0, 0); nv];
            for round in 0..=cfg.rounds {
                // The order rotates every round, so no variant always runs first or after the
                // same one.
                for vi in (0..nv).map(|j| (round + j) % nv) {
                    let what = format!("{n} tokens {mode}");
                    let (secs, r) = time(gpu, &graph, cfg.variants[vi], cfg.copies, &what)?;
                    ran[vi] = r;
                    if round > 0 {
                        best[vi] = best[vi].min(secs);
                    }
                }
            }
            for (vi, &variant) in cfg.variants.iter().enumerate() {
                rows.push(Row {
                    mode,
                    variant,
                    ran: ran[vi].0,
                    ran_opts: ran[vi].1,
                    tokens: n,
                    secs: best[vi],
                    bytes: bytes(cfg, mode, n),
                });
            }
        }
    }
    Ok(rows)
}

/// Every variant of `cfg` other than `base` against `base` in every mode of `modes` at every
/// token count: `cfg.rounds` timed rounds of the pair after one untimed one, the base first in
/// even rounds and second in odd ones (neither always gets the warmer state), median and min
/// per mixer. The default kernel and options are back afterwards.
pub fn ab(
    gpu: &mut Backend,
    cfg: &Config,
    modes: &[&'static str],
    base: Variant,
) -> Result<Vec<AbRow>, String> {
    check_config(cfg)?;
    let mut rows = Vec::new();
    for &n in &cfg.tokens {
        let wctx = Context::new(2 * cfg.copies + 4, 0);
        let w = weights(&wctx, gpu, cfg, n)?;
        for &mode in modes {
            let ctx = Context::new(16 * cfg.copies + 16, 8 * cfg.copies + 16);
            let i = inputs(&ctx, cfg, mode, n);
            let (graph, _) = build(&ctx, gpu, &w, &i, mode)?;
            fill(&i, &w, cfg, mode, n);
            let what = format!("{n} tokens {mode}");
            for &variant in cfg.variants.iter().filter(|&&v| v != base) {
                let pair = [base, variant];
                let mut secs: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
                let mut ran = [(0, 0); 2];
                for round in 0..=cfg.rounds {
                    let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
                    for side in order {
                        let (s, r) = time(gpu, &graph, pair[side], cfg.copies, &what)?;
                        ran[side] = r;
                        if round > 0 {
                            secs[side].push(s);
                        }
                    }
                }
                let min = |x: &[f64]| x.iter().copied().fold(f64::INFINITY, f64::min);
                rows.push(AbRow {
                    mode,
                    tokens: n,
                    base,
                    variant,
                    ran,
                    median: [median(&secs[0]), median(&secs[1])],
                    min: [min(&secs[0]), min(&secs[1])],
                    bytes: bytes(cfg, mode, n),
                });
            }
        }
    }
    Ok(rows)
}

/// At every token count of `cfg` and in every mode, one mixer per variant: its F32 values (where
/// it writes them) and BF16 copy against [`Config::reference`]'s, bit for bit. The outputs are
/// filled with NaN before each run, so a value no kernel wrote cannot match.
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    check_config(cfg)?;
    let one = Config {
        copies: 1,
        ..cfg.clone()
    };
    let mut out = Vec::new();
    for &n in &cfg.tokens {
        let wctx = Context::new(8, 0);
        let w = weights(&wctx, gpu, &one, n)?;
        for mode in MODES {
            let ctx = Context::new(64, 32);
            let i = inputs(&ctx, cfg, mode, n);
            let (graph, outs) = build(&ctx, gpu, &w, &i, mode)?;
            fill(&i, &w, cfg, mode, n);
            let (y, cast) = (outs[0], w.casts[0]);
            let values = (cfg.e * n) as usize;
            let f32_out = matches!(mode, "f32" | "xn" | "pre");
            let mut get = |v: Variant| -> Result<(Vec<u8>, (i32, i32)), String> {
                y.set_f32(&vec![f32::NAN; values]);
                cast.set_bytes(&vec![0xff; values * 2]);
                v.force();
                let r = gpu.compute(&graph);
                unforce();
                r.map_err(|e| format!("{n} tokens {mode} {v}: {e}"))?;
                let mut b = Vec::new();
                if f32_out {
                    b.extend(y.bytes());
                }
                if mode != "xn" {
                    b.extend(cast.bytes());
                }
                Ok((b, (last_kernel(), last_reg_opts())))
            };
            let (want, _) = get(cfg.reference)?;
            for &v in &cfg.variants {
                let (got, (ran, ran_opts)) = get(v)?;
                // Values: 4-byte F32 ones first, then 2-byte BF16 ones.
                let f32_len = if f32_out { values * 4 } else { 0 };
                let (wa, wb) = want.split_at(f32_len);
                let (ga, gb) = got.split_at(f32_len);
                let differing = wa
                    .chunks(4)
                    .zip(ga.chunks(4))
                    .filter(|(p, q)| p != q)
                    .count()
                    + wb.chunks(2)
                        .zip(gb.chunks(2))
                        .filter(|(p, q)| p != q)
                        .count();
                out.push(Check {
                    mode,
                    variant: v,
                    ran,
                    ran_opts,
                    tokens: n,
                    values: wa.len() / 4 + wb.len() / 2,
                    differing,
                    hash: fnv(&got),
                });
            }
        }
    }
    Ok(out)
}

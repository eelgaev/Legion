//! `legion nvfp4-mvq-bench`: dense NVFP4 matvecs of 1-8 columns (Qwen3.8-27B's FFN shapes by
//! default) on ggml's quantize_q8_1 + MMVQ and on Legion's tensor-core op, timed on
//! one GPU with a bits check against ggml (#1133; a development tool, so a build tree's release
//! binary can measure the box it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::model::core::nvfp4_mvq::bench::{self, Config};

#[derive(Args)]
pub struct Nvfp4MvqBenchArgs {
    /// Columns per step, comma separated (1-8).
    #[arg(long, value_delimiter = ',')]
    cols: Vec<usize>,
    /// Weight shapes as `k x rows`, comma separated (e.g. `5120x17408,17408x5120`).
    #[arg(long, value_delimiter = ',')]
    shapes: Vec<String>,
    /// Weights per shape, each its own.
    #[arg(long, default_value_t = 4)]
    layers: usize,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// The matmuls alone, without the one-value scale after each (the 27B's weights have it).
    #[arg(long)]
    no_scale: bool,
    /// Each variant's timed rounds back to back, their start and end times on stderr (to cut
    /// a power log by), instead of in turns.
    #[arg(long)]
    blocks: bool,
    /// Launch shapes of the op to time besides the default, at 5-8 columns, comma separated:
    /// `r<region rows 8|16|32>c<CTAs per SM 1|2>k<pieces a row>a<ring pieces ahead>`, any
    /// field left out for its default (e.g. `r32,c2k4,a20`).
    #[arg(long, value_delimiter = ',')]
    tune: Vec<String>,
}

fn shape(s: &str) -> anyhow::Result<(usize, usize)> {
    let (k, rows) = s
        .split_once('x')
        .ok_or_else(|| anyhow::anyhow!("shape {s:?} is not `k x rows`"))?;
    Ok((k.trim().parse()?, rows.trim().parse()?))
}

pub fn run(args: Nvfp4MvqBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        layers: args.layers,
        rounds: args.rounds,
        scale: !args.no_scale,
        blocks: args.blocks,
        tunes: args
            .tune
            .iter()
            .map(|t| legion::model::core::nvfp4_mvq::Tune::parse(t))
            .collect::<Result<_, _>>()
            .map_err(anyhow::Error::msg)?,
        ..Config::default()
    };
    if !args.cols.is_empty() {
        cfg.cols = args.cols;
    }
    if !args.shapes.is_empty() {
        cfg.shapes = args
            .shapes
            .iter()
            .map(|s| shape(s))
            .collect::<Result<_, _>>()?;
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:>6} {:>6} {:>4} {:<12} {:>9} {:>7} {:>6} {:>9} {:>9}  first difference",
        "k", "rows", "cols", "variant", "us/mm", "vs ggml", "GB/s", "differ", "outputs"
    );
    let mut reference = 0.0;
    let mut bad = 0;
    for r in &rows {
        if r.variant == "ggml" {
            reference = r.secs;
        }
        let gbs = (r.k / 64 * 36 * r.rows) as f64 / r.secs / 1e9;
        bad += usize::from(r.differ > 0);
        let first = r.first.map_or(String::new(), |(l, c, row, got, want)| {
            format!("layer {l} col {c} row {row}: {got:e} vs {want:e}")
        });
        println!(
            "{:>6} {:>6} {:>4} {:<12} {:>9.2} {:>7.3} {:>6.1} {:>9} {:>9}  {first}",
            r.k,
            r.rows,
            r.cols,
            r.variant,
            r.secs * 1e6,
            r.secs / reference,
            gbs,
            r.differ,
            r.outputs,
        );
    }
    anyhow::ensure!(bad == 0, "{bad} variants differ from ggml's bits");
    println!("all bits equal to ggml's mul_mat_vec_q");
    Ok(())
}

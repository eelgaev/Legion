//! `legion q-small-m-bench`: GGUF-quantized matmuls of 1-16 columns as ggml's `mul_mat` (MMVQ up
//! to its bound, MMQ above) and, at 2-8 columns, as Legion's small-M op in each rows-a-warp
//! variant, timed on one GPU (#1210; a development tool, so a build tree's release binary can
//! measure the box it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`). `--check` compares the
//! bits instead.

use clap::Args;
use legion::model::core::q_small_m::{
    TYPES,
    bench::{self, Config},
};

#[derive(Args)]
pub struct QSmallMBenchArgs {
    /// Weight types (q4_K, q5_K, q6_K, iq4_xs, q8_0, iq4_nl, q3_K), comma separated; default:
    /// the dense 27B UD-Q4_K_XL's main four.
    #[arg(long, value_delimiter = ',')]
    types: Vec<String>,
    /// Weight shapes as KxROWS (K a multiple of 512), comma separated; default: the dense 27B's
    /// FFN gate/up and down, attention out, DeltaNet qkv and gate.
    #[arg(long, value_delimiter = ',')]
    shapes: Vec<String>,
    /// Columns per step (1-16), comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [1, 2, 3, 4, 5, 6, 8, 16])]
    cols: Vec<usize>,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Compare the op's bits with ggml's `mul_mat` in every variant instead of timing; fails on
    /// any differing output where ggml runs MMVQ.
    #[arg(long)]
    check: bool,
}

pub fn run(args: QSmallMBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        cols: args.cols,
        rounds: args.rounds,
        ..Config::default()
    };
    if !args.types.is_empty() {
        cfg.types = args
            .types
            .iter()
            .map(|n| {
                TYPES
                    .iter()
                    .copied()
                    .find(|&t| legion::ggml::type_name(t).eq_ignore_ascii_case(n))
                    .ok_or_else(|| anyhow::anyhow!("type {n}: not one the op takes"))
            })
            .collect::<anyhow::Result<_>>()?;
    }
    if !args.shapes.is_empty() {
        cfg.shapes = args
            .shapes
            .iter()
            .map(|s| {
                let (k, rows) = s
                    .split_once('x')
                    .ok_or_else(|| anyhow::anyhow!("shape {s}: KxROWS"))?;
                Ok((k.parse()?, rows.parse()?))
            })
            .collect::<anyhow::Result<_>>()?;
    }
    if args.check {
        let checks = bench::check(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:<7} {:>6} {:>6} {:>5} {:>5} {:>9} {:>10} {:>10} {:>9}",
            "type", "k", "rows", "cols", "ggml", "variants", "outputs", "differing", "max rel"
        );
        for c in &checks {
            println!(
                "{:<7} {:>6} {:>6} {:>5} {:>5} {:>9} {:>10} {:>10} {:>9.2e}",
                bench::type_label(c.ty),
                c.k,
                c.rows,
                c.cols,
                if c.ggml_mmvq { "mmvq" } else { "mmq" },
                c.variants,
                c.outputs,
                c.differing,
                c.max_rel
            );
        }
        let bad = checks
            .iter()
            .filter(|c| c.ggml_mmvq && (c.differing > 0 || c.variants == 0))
            .count();
        anyhow::ensure!(bad == 0, "{bad} MMVQ cases differ from ggml's bits");
        println!("all bits equal to ggml's mul_mat wherever ggml runs MMVQ");
        return Ok(());
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:<7} {:>6} {:>6} {:>5} {:<7} {:>9} {:>7} {:>7}",
        "type", "k", "rows", "cols", "variant", "µs", "GB/s", "vs ggml"
    );
    let mut reference = 0.0;
    for r in &rows {
        if r.r == 0 {
            reference = r.secs;
        }
        let variant = if r.r == 0 {
            "ggml".to_string()
        } else {
            match r.r {
                r if r > 0 => format!("op r{r}"),
                r => format!("op c{}", -r),
            }
        };
        println!(
            "{:<7} {:>6} {:>6} {:>5} {:<7} {:>9.1} {:>7.1} {:>7.3}",
            bench::type_label(r.ty),
            r.k,
            r.rows,
            r.cols,
            variant,
            r.secs * 1e6,
            r.bytes / r.secs / 1e9,
            r.secs / reference,
        );
    }
    Ok(())
}

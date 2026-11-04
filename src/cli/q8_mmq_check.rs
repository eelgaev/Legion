//! `legion q8-mmq-check`: the two-launch Q8_0 MMQ op's bits and time against ggml's `mul_mat`
//! on one GPU (#836; a development tool, so a build tree's release binary can check the box it
//! runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::model::q8_mmq::check;

#[derive(Args)]
pub struct Q8MmqCheckArgs {
    /// Q8_0 weight shapes `k x rows`, comma separated; default: Qwen3.8-Flash-Next's Q8_0
    /// projections (DeltaNet qkv and gate, attention q and k/v, attention and DeltaNet out,
    /// shared expert gate/up and down, the hyper-connection mixer's down and up).
    #[arg(
        long,
        value_delimiter = ',',
        default_values = [
            "2560x10240", "2560x6144", "2560x12288", "2560x512", "6144x2560", "10240x2560",
            "2560x640", "640x2560", "10240x320", "320x10240",
        ]
    )]
    shapes: Vec<String>,
    /// Columns (tokens) per step, comma separated (9-255).
    #[arg(long, value_delimiter = ',', default_values_t = [9, 16, 32, 48, 64, 80, 128, 255])]
    cols: Vec<usize>,
    /// The GPU's SM count for the grid (0: query the GPU).
    #[arg(long, default_value_t = 0)]
    sms: i64,
    /// Timed rounds per shape and step size (0: bits only).
    #[arg(long, default_value_t = 20)]
    rounds: usize,
    /// Weight bytes per timed graph (copies of the weight up to this).
    #[arg(long, default_value_t = 256 << 20)]
    min_bytes: usize,
    /// The op's kernels to check, comma separated: 0 ggml's MMQ tile code, 8 or 4 the
    /// streaming kernel with that many warps per CTA (#858). Each is a table of its own.
    #[arg(long, value_delimiter = ',', default_values_t = [0, 8, 4])]
    fast: Vec<i32>,
    /// How split tiles' pieces are added up, comma separated: `auto` (the server's default
    /// rule), `last` (the CTA that counts a tile's last piece, inside the matmul kernel),
    /// `launch` (a fixup launch of its own, ggml's). Each kernel and mode is a table of its own.
    #[arg(long, value_delimiter = ',', default_values = ["auto"])]
    fixup: Vec<String>,
}

/// A `--fixup` value.
fn fixup_mode(s: &str) -> anyhow::Result<check::Fixup> {
    Ok(match s.trim() {
        "auto" => check::Fixup::Auto,
        "last" => check::Fixup::Last,
        "launch" => check::Fixup::Launch,
        other => anyhow::bail!("--fixup {other}: want auto, last or launch"),
    })
}

fn gpu_sms() -> i64 {
    #[cfg(feature = "cuda")]
    if let Some(c) = legion::residency::cuda::caps(0) {
        return c.sms as i64;
    }
    0
}

/// ggml's own CUDA graphs off, as `legion serve` sets them, so both sides run as stage graphs
/// (one capture each, the op's launches inside), timed as the server runs them. An explicit
/// setting wins.
pub fn prepare_env() {
    if std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_none() {
        // SAFETY: called from `main` before the runtime starts: no other thread reads the
        // environment.
        unsafe { std::env::set_var("GGML_CUDA_DISABLE_GRAPHS", "1") };
    }
}

pub fn run(args: Q8MmqCheckArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    if std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some() {
        legion::ggml::stage_graph::set_safe_ops(legion::model::core::stage_ops::capture_safe);
        legion::ggml::stage_graph::set_enabled(true);
    }
    let sms = if args.sms > 0 { args.sms } else { gpu_sms() };
    anyhow::ensure!(sms > 0, "cannot query the GPU's SM count; pass --sms");
    let shapes = args
        .shapes
        .iter()
        .map(|s| {
            let (k, rows) = s
                .split_once('x')
                .ok_or_else(|| anyhow::anyhow!("shape {s}: want KxROWS"))?;
            Ok((k.trim().parse::<usize>()?, rows.trim().parse::<usize>()?))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let fixups = args
        .fixup
        .iter()
        .map(|s| fixup_mode(s))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut bad = 0;
    for &fast in &args.fast {
        for &fix in &fixups {
            println!(
                "kernel {fast} (0: ggml's tile code, 8/4: streaming, warps per CTA), fixup {fix:?}"
            );
            bad += table(&mut gpu, sms, fast, fix, &shapes, &args)?;
        }
    }
    anyhow::ensure!(bad == 0, "{bad} cases differ from ggml's bits");
    println!("all bits equal to ggml's ({sms} SMs)");
    Ok(())
}

/// One kernel's checks, printed; the cases whose bits differ.
fn table(
    gpu: &mut legion::ggml::Backend,
    sms: i64,
    fast: i32,
    fix: check::Fixup,
    shapes: &[(usize, usize)],
    args: &Q8MmqCheckArgs,
) -> anyhow::Result<usize> {
    let checks = check::check(
        gpu,
        sms,
        fast,
        fix,
        shapes,
        &args.cols,
        args.rounds,
        args.min_bytes,
    )
    .map_err(anyhow::Error::msg)?;
    println!(
        "{:>6} {:>6} {:>5} {:<5} {:>9} {:>10} {:>10} {:>9} {:>9} {:>7}",
        "k",
        "rows",
        "cols",
        "split",
        "outputs",
        "differing",
        "max diff",
        "ggml us",
        "op us",
        "ratio"
    );
    let mut bad = 0;
    let (mut ggml_total, mut op_total) = (0f64, 0f64);
    for c in &checks {
        println!(
            "{:>6} {:>6} {:>5} {:<5} {:>9} {:>10} {:>10.3e} {:>9.2} {:>9.2} {:>7.3}",
            c.k,
            c.rows,
            c.cols,
            if c.split { "yes" } else { "no" },
            c.outputs,
            c.differing,
            c.max_diff,
            c.ggml_secs * 1e6,
            c.op_secs * 1e6,
            if c.ggml_secs > 0.0 {
                c.op_secs / c.ggml_secs
            } else {
                0.0
            }
        );
        bad += usize::from(c.differing > 0);
        ggml_total += c.ggml_secs;
        op_total += c.op_secs;
    }
    if ggml_total > 0.0 {
        println!(
            "sum of best times: ggml {:.1} us, op {:.1} us ({:+.1}%)",
            ggml_total * 1e6,
            op_total * 1e6,
            (op_total / ggml_total - 1.0) * 100.0
        );
    }
    Ok(bad)
}

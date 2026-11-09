//! `legion lin-bench`: GLM's F16 prompt GEMM kernels (linear_tc and the shapes of
//! linear_tc_gb, #862) at Maya-S's dense shapes on one GPU, timed against each other and checked
//! for the same output bits (a development tool, so a build tree's release binary can measure
//! the box it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct LinBenchArgs {
    /// Tokens per step (a prompt chunk), comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [4096, 2048])]
    tokens: Vec<usize>,
    /// Shapes by name (see `glm5next::ops::lin_bench::SHAPES`), comma separated; all when empty.
    #[arg(long, value_delimiter = ',')]
    shapes: Vec<String>,
    /// linear_tc_gb shapes to time (indices of `tc_gb_shape`), comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [0u8, 1, 2, 3, 4, 5])]
    gb: Vec<u8>,
    /// Timed calls per kernel and round.
    #[arg(long, default_value_t = 5)]
    reps: usize,
    /// Rounds (the best counts; the kernels take turns within a round).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Also time the one-matrix F32 shapes with their F16 rows made in the call (`cvt+` rows:
    /// the conversion and an F16-input kernel, against linear_tc on the F32 rows).
    #[arg(long)]
    convert: bool,
    /// Also time each product as it runs without the GB10 kernels (`plain` rows: `serve
    /// --glm-linear-plain`, linear_mm for F16 weights; #1212).
    #[arg(long)]
    plain: bool,
    /// Time the MLA products' F16 handoff instead (#862): attn_q_b -> attn_k_b and attn_v_b ->
    /// attn_output's rows, as main runs them and in F16 (attn_k_b on each kernel); fails when
    /// a value's bits differ.
    #[arg(long)]
    chain: bool,
}

#[cfg(feature = "cuda")]
pub fn run(args: LinBenchArgs) -> anyhow::Result<()> {
    use legion::model::glm5next::ops::{LinTc, lin_bench, set_gb10_linear};
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    // the GB10 kernels: linear_tc and its shapes take F16 weights only with them on
    set_gb10_linear(true);
    let mut kernels = vec![LinTc::Tc];
    kernels.extend(args.gb.iter().map(|&v| LinTc::Gb(v)));
    kernels.push(LinTc::Auto);
    if args.chain {
        return chain(&mut gpu, &args, &kernels);
    }
    println!(
        "{:<11} {:>6} {:>6} {:>6} {:>3} {:>6} {:>8} {:>9} {:>7} {:>8} {:>18}",
        "shape", "k", "m", "h", "xh", "tokens", "kern", "ms/call", "TFLOPS", "vs tc", "output bits"
    );
    let mut differ = 0;
    for shape in lin_bench::SHAPES {
        if !args.shapes.is_empty() && !args.shapes.iter().any(|s| s == shape.name) {
            continue;
        }
        for &n in &args.tokens {
            anyhow::ensure!(
                n >= 256,
                "--tokens {n}: linear_tc takes steps of 256+ tokens"
            );
            let mut t =
                lin_bench::gemm_ms(&mut gpu, shape, n, args.reps, args.rounds, &kernels, false);
            let direct = t.len();
            let mut labels: Vec<String> = Vec::new();
            if args.convert && !shape.xh && shape.h == 1 {
                t.extend(lin_bench::gemm_ms(
                    &mut gpu,
                    shape,
                    n,
                    args.reps,
                    args.rounds,
                    &kernels,
                    true,
                ));
            }
            for (i, r) in t.iter().enumerate() {
                let kern = match r.kernel {
                    LinTc::Auto => "auto".to_string(),
                    LinTc::Tc => "tc".to_string(),
                    LinTc::Gb(v) => format!("gb{v}"),
                };
                labels.push(if i < direct {
                    kern
                } else {
                    format!("cvt+{kern}")
                });
            }
            if args.plain {
                set_gb10_linear(false);
                t.extend(lin_bench::gemm_ms(
                    &mut gpu,
                    shape,
                    n,
                    args.reps,
                    args.rounds,
                    &[LinTc::Auto],
                    false,
                ));
                set_gb10_linear(true);
                labels.push("plain".to_string());
            }
            let flop = 2.0 * (shape.k * shape.m * shape.h) as f64 * n as f64;
            for (r, kern) in t.iter().zip(labels) {
                println!(
                    "{:<11} {:>6} {:>6} {:>6} {:>3} {:>6} {:>8} {:>9.3} {:>7.1} {:>8.3} {:>18}",
                    shape.name,
                    shape.k,
                    shape.m,
                    shape.h,
                    if shape.xh { "y" } else { "n" },
                    n,
                    kern,
                    r.ms,
                    flop / r.ms / 1e9,
                    r.ms / t[0].ms,
                    format!("{:016x}", r.bits)
                );
                differ += (r.bits != t[0].bits) as usize;
            }
        }
    }
    if differ > 0 {
        anyhow::bail!("{differ} outputs differ from linear_tc's bits");
    }
    Ok(())
}

/// `--chain`: [`legion::model::glm5next::ops::lin_bench::chain_ms`] per token count.
#[cfg(feature = "cuda")]
fn chain(
    gpu: &mut legion::ggml::Backend,
    args: &LinBenchArgs,
    kernels: &[legion::model::glm5next::ops::LinTc],
) -> anyhow::Result<()> {
    use legion::model::glm5next::ops::{LinTc, lin_bench};
    println!(
        "{:<8} {:>6} {:>8} {:>9} {:>9} {:>7} {:>8}",
        "product", "tokens", "kern", "main ms", "f16 ms", "ratio", "differ"
    );
    let mut differ = 0;
    for &n in &args.tokens {
        anyhow::ensure!(
            n >= 1024,
            "--tokens {n}: the F16 handoff is for steps of 1024+ tokens"
        );
        let (mut main, mut f16) = (0.0, 0.0);
        for row in lin_bench::chain_ms(gpu, n, args.reps, args.rounds, kernels) {
            main += row.base_ms;
            // the launcher's choice counts in the sums
            f16 += row
                .chain
                .iter()
                .find(|c| c.0 == LinTc::Auto)
                .map_or(row.chain[0].1, |c| c.1);
            for (k, ms, d) in &row.chain {
                let kern = match k {
                    LinTc::Auto => "auto".to_string(),
                    LinTc::Tc => "tc".to_string(),
                    LinTc::Gb(v) => format!("gb{v}"),
                };
                println!(
                    "{:<8} {:>6} {:>8} {:>9.3} {:>9.3} {:>7.3} {:>8}",
                    row.name,
                    n,
                    kern,
                    row.base_ms,
                    ms,
                    ms / row.base_ms,
                    d
                );
                differ += d;
            }
        }
        println!(
            "{:<8} {:>6} {:>8} {:>9.3} {:>9.3} {:>7.3}",
            "sum",
            n,
            "auto",
            main,
            f16,
            f16 / main
        );
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ in bits");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: LinBenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("lin-bench needs a CUDA build")
}

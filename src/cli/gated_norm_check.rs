//! `legion gated-norm-check`: the gated DeltaNet output norm's BF16 op against ggml's chain and
//! cast on one GPU (#689; a development tool, so a build tree's release binary can check the box
//! it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::model::core::gated_norm_check::check;

#[derive(Args)]
pub struct GatedNormCheckArgs {
    /// Head layouts `head_v x n_v`, comma separated; default: Qwen3.8-27B's (128 x 48), a
    /// smaller model's (128 x 32) and heads of 64 and 256.
    #[arg(long, value_delimiter = ',', default_values = ["128x48", "128x32", "64x16", "256x8"])]
    heads: Vec<String>,
    /// Columns per step, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 32])]
    cols: Vec<usize>,
    /// The norm's epsilon.
    #[arg(long, default_value_t = 1e-6)]
    eps: f32,
}

pub fn run(args: GatedNormCheckArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let heads = args
        .heads
        .iter()
        .map(|h| -> anyhow::Result<(usize, usize)> {
            let (a, b) = h
                .split_once('x')
                .ok_or_else(|| anyhow::anyhow!("--heads {h}: want head_v x n_v"))?;
            Ok((a.trim().parse()?, b.trim().parse()?))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    println!(
        "{:<10} {:>5} {:<8} {:>9} {:>10}",
        "heads", "cols", "gate", "outputs", "differing"
    );
    let mut bad = 0;
    for c in check(&mut gpu, &heads, &args.cols, args.eps).map_err(anyhow::Error::msg)? {
        println!(
            "{:<10} {:>5} {:<8} {:>9} {:>10}",
            format!("{}x{}", c.head_v, c.n_v),
            c.cols,
            if c.silu { "silu" } else { "sigmoid" },
            c.outputs,
            c.differing
        );
        bad += usize::from(c.differing > 0);
    }
    anyhow::ensure!(bad == 0, "{bad} cases differ from ggml's bits");
    println!("all bits equal to ggml's");
    Ok(())
}

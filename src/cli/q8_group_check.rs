//! `legion q8-group-check`: the grouped Q8_0 matvec op's bits against ggml's `mul_mat` on one
//! GPU (#819; a development tool, so a build tree's release binary can check the box it runs on;
//! pick the GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::model::core::q8_group::check;

#[derive(Args)]
pub struct Q8GroupCheckArgs {
    /// Values per row (a multiple of 32).
    #[arg(long, default_value_t = 2560)]
    k: usize,
    /// Weight groups `m1+m2+...` (Q8_0 `[k, m_i]` read with one input), comma separated;
    /// default: Qwen3.8-Flash-Next's DeltaNet qkv+gate, attention q+k+v and shared expert
    /// gate+up.
    #[arg(long, value_delimiter = ',', default_values = ["10240+6144", "12288+512+512", "640+640"])]
    groups: Vec<String>,
    /// Columns per step, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [1, 2, 3, 4, 8])]
    cols: Vec<usize>,
}

pub fn run(args: Q8GroupCheckArgs) -> anyhow::Result<()> {
    // The stage graphs below capture ggml's segments themselves, so ggml's own CUDA graphs must
    // be off before CUDA starts, as `legion serve` sets them.
    if std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_none() {
        // SAFETY: before CUDA starts, as in `legion serve`; nothing else reads the environment.
        unsafe { std::env::set_var("GGML_CUDA_DISABLE_GRAPHS", "1") };
    }
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let groups = args
        .groups
        .iter()
        .map(|g| {
            g.split('+')
                .map(|m| m.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut bad = 0;
    println!(
        "{:<16} {:>5} {:<7} {:>9} {:>10} {:>10}",
        "rows", "cols", "out", "outputs", "differing", "max diff"
    );
    for glu in [false, true] {
        let pairs: Vec<Vec<usize>> = groups
            .iter()
            .filter(|g| !glu || (g.len() == 2 && g[0] == g[1]))
            .cloned()
            .collect();
        for c in
            check::check(&mut gpu, args.k, &pairs, &args.cols, glu).map_err(anyhow::Error::msg)?
        {
            let rows: Vec<String> = c.rows.iter().map(|r| r.to_string()).collect();
            println!(
                "{:<16} {:>5} {:<7} {:>9} {:>10} {:>10.3e}",
                rows.join("+"),
                c.cols,
                if c.glu { "swiglu" } else { "matmul" },
                c.outputs,
                c.differing,
                c.max_diff
            );
            bad += usize::from(c.differing > 0);
        }
    }
    // The hyper-connection mixer's Q8_1 blocks of the block input (Qwen3.8-Flash-Next's
    // n_embd 2560, 4 streams, rank 320), as the server builds it with `--q8-group` (stage
    // graphs on).
    legion::ggml::stage_graph::set_enabled(true);
    legion::model::core::set_q8_group(true);
    for (n, bytes, differing) in
        check::check_hc(&mut gpu, 2560, 4, 320, &args.cols).map_err(anyhow::Error::msg)?
    {
        match differing {
            Some(d) => {
                println!("hc mixer q8_1      {n:>5} bytes {bytes:>9} {d:>10}");
                bad += usize::from(d > 0);
            }
            // The GB10 has its own mixer kernels, which write no Q8_1 blocks (#311).
            None if legion::model::core::q8_group::gb10() => {
                println!("hc mixer q8_1      {n:>5} not written (GB10)");
            }
            None => {
                println!("hc mixer q8_1      {n:>5} not written");
                bad += 1;
            }
        }
    }
    // The shared expert's down projection reading the SwiGLU's Q8_1 blocks (n_ff 640, n_embd
    // 2560).
    for (n, outputs, differing) in check::check_swiglu_down(&mut gpu, args.k, 640, 2560, &args.cols)
        .map_err(anyhow::Error::msg)?
    {
        match differing {
            Some(d) => {
                println!("shexp swiglu+down  {n:>5} outputs {outputs:>7} {d:>10}");
                bad += usize::from(d > 0);
            }
            None => println!("shexp swiglu+down  {n:>5} not taken"),
        }
    }
    anyhow::ensure!(bad == 0, "{bad} cases differ from ggml's bits");
    println!("all bits equal to ggml's");
    Ok(())
}

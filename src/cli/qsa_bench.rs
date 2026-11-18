//! `legion qsa-bench`: qwen4exp's sparse-attention prompt kernels (#828) at the model's shape on
//! one GPU, timed against each other and checked for the same output bits (a development tool,
//! so a build tree's release binary can measure the box it runs on; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct QsaBenchArgs {
    /// Context lengths: each step is the last --tokens tokens of one, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [32768, 65536, 123_000])]
    ctx: Vec<usize>,
    /// Tokens per step (a prompt chunk).
    #[arg(long, default_value_t = 4096)]
    tokens: usize,
    /// Timed calls per kernel and round.
    #[arg(long, default_value_t = 5)]
    reps: usize,
    /// Rounds (the best counts; the kernels take turns within a round).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Query heads (24 on one GPU, 12 on a rank of two under tensor parallelism).
    #[arg(long, default_value_t = 24)]
    heads: i64,
    /// KV heads (2 on one GPU, 1 on a rank of two).
    #[arg(long, default_value_t = 2)]
    kv_heads: i64,
    /// Kernels, comma separated, the first the reference for time and bits: mma, gb, gw, gp
    /// (attn_gp_kernel: two neighbouring tokens a CTA, sharing the K / V rows both select),
    /// auto.
    #[arg(long, value_delimiter = ',', default_values_t = ["mma".to_string(), "gw".to_string(), "auto".to_string()])]
    kernels: Vec<String>,
    /// Pool layouts, comma separated: local (neighbouring tokens share most pools, as in a
    /// prompt), spread (pools all over the context).
    #[arg(long, value_delimiter = ',', default_values_t = ["local".to_string(), "spread".to_string()])]
    pools: Vec<String>,
    /// Selection rows recorded by `LEGION_DEBUG_QSA_SEL` (comma separated files): each file is
    /// one more step, its tokens and positions as recorded (block table in order), after the
    /// --ctx steps.
    #[arg(long, value_delimiter = ',')]
    sel: Vec<String>,
    /// The GPU's mma.sync F16 (F32 accumulate) rate in TFLOP/s, for the floor column: a
    /// GB10 issues about 71.
    #[arg(long, default_value_t = 71.0)]
    peak_tflops: f64,
}

#[cfg(feature = "cuda")]
pub fn run(args: QsaBenchArgs) -> anyhow::Result<()> {
    use legion::model::core::KvTypes;
    use legion::model::qwen4::{PromptKernel, qsa_bench};
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let kernels = args
        .kernels
        .iter()
        .map(|k| {
            Ok(match k.as_str() {
                "mma" => PromptKernel::Mma,
                "gb" => PromptKernel::Gb,
                "gw" => PromptKernel::Gw,
                "gp" => PromptKernel::Gp,
                "auto" => PromptKernel::Auto,
                _ => anyhow::bail!("--kernels: unknown kernel {k} (see --help)"),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(!kernels.is_empty(), "--kernels: none given");
    anyhow::ensure!(
        args.kv_heads > 0 && args.heads % args.kv_heads == 0 && args.heads / args.kv_heads <= 16,
        "--heads {} over --kv-heads {}: at most 16 query heads a kv head",
        args.heads,
        args.kv_heads
    );
    // Useful FLOPs: q.k and p.v of every query head over its token's keys. Issued: the
    // kernels' mma.m16n8k16 products, 16 rows a kv head (G heads + padding), hi and lo
    // halves of q and p, whole chunks of 32 keys. Gathered: K and V rows of each kv head.
    let hd = 256.0;
    println!(
        "heads {} / kv heads {}; TFLOP/s useful, mma TFLOP/s issued, floor = issued at {} \
         TFLOP/s; GB/s of K + V rows loaded",
        args.heads, args.kv_heads, args.peak_tflops
    );
    println!(
        "{:<7} {:>7} {:>6} {:>11} {:>9} {:>7} {:>7} {:>7} {:>6} {:>8} {:>7} {:>8} {:>18}",
        "pools",
        "ctx",
        "tokens",
        "kern",
        "ms/call",
        "vs ref",
        "TFLOP/s",
        "mma",
        "%peak",
        "floor ms",
        "GB/s",
        "rows/key",
        "output bits"
    );
    let mut differ = 0;
    let mut cases = Vec::new();
    for &n_ctx in &args.ctx {
        anyhow::ensure!(
            n_ctx >= args.tokens + 4 * 512 * 2,
            "--ctx {n_ctx} is too short for --tokens {}",
            args.tokens
        );
        for pools in &args.pools {
            anyhow::ensure!(
                pools == "local" || pools == "spread",
                "--pools: unknown layout {pools} (local, spread)"
            );
            cases.push((pools.clone(), n_ctx, None));
        }
    }
    for f in &args.sel {
        let (_, row, rows) = legion::model::qwen4::qsa_sel::read(f)
            .map_err(|e| anyhow::anyhow!("--sel {f}: {e}"))?;
        let name = std::path::Path::new(f)
            .file_stem()
            .map_or("sel".into(), |s| s.to_string_lossy().into_owned());
        cases.push((name, 0, Some((row, rows))));
    }
    println!(
        "rows/key: K / V rows loaded per selected key (gp shares the rows a neighbour already \
         holds; the others load every key)"
    );
    for (name, n_ctx, real) in &cases {
        let layout = match (name.as_str(), real) {
            (_, Some((row, rows))) => qsa_bench::Layout::Real { rows, row: *row },
            ("local", None) => qsa_bench::Layout::Local,
            _ => qsa_bench::Layout::Spread,
        };
        let t = qsa_bench::prompt_ms(
            &mut gpu,
            KvTypes::default(),
            (args.heads, args.kv_heads),
            *n_ctx,
            args.tokens,
            args.reps,
            args.rounds,
            layout,
            &kernels,
        );
        for (r, kname) in t.iter().zip(&args.kernels) {
            let useful = r.keys as f64 * args.heads as f64 * hd * 4.0;
            let issued = r.chunks as f64 * args.kv_heads as f64 * 32.0 * 16.0 * hd * 8.0;
            let loaded = if r.kernel == PromptKernel::Gp {
                r.pair_rows
            } else {
                r.keys
            };
            let bytes = loaded as f64 * args.kv_heads as f64 * hd * 4.0;
            let s = r.ms * 1e-3;
            let floor_ms = issued / (args.peak_tflops * 1e12) * 1e3;
            println!(
                "{:<7} {:>7} {:>6} {:>11} {:>9.3} {:>7.3} {:>7.1} {:>7.1} {:>5.0}% {:>8.3} {:>7.0} {:>8.3} {:>18}",
                name,
                r.ctx,
                r.tokens,
                kname,
                r.ms,
                r.ms / t[0].ms,
                useful / s * 1e-12,
                issued / s * 1e-12,
                floor_ms / r.ms * 100.0,
                floor_ms,
                bytes / s * 1e-9,
                loaded as f64 / r.keys.max(1) as f64,
                format!("{:016x}", r.bits)
            );
            differ += (r.bits != t[0].bits) as usize;
        }
    }
    if differ > 0 {
        anyhow::bail!(
            "{differ} outputs differ from the {} kernel's bits",
            args.kernels[0]
        );
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: QsaBenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("qsa-bench needs a CUDA build")
}

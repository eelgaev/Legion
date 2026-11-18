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
    /// Kernels, comma separated, the first the reference for time and bits: mma, gb, gw, auto;
    /// attn_gw_kernel variants (same bits): gw-early (cells looked up a chunk earlier), gw-half
    /// (refills in halves, each as soon as it is read), gw-early-half, gw-1cta (gw at one CTA
    /// per SM); probes (wrong output, left out of the bit check): gw-nofill, gw-nomma, gw-none
    /// (attn_gw_kernel without refills / products / both), gw-halfbytes (half of each refill:
    /// the same round trips, half the bytes), pipe-<p> and pipe1-<p> (the
    /// register-only mma patterns d1, pair, pv, gw, chain, pair16, pv16, gwdep at two / one CTA
    /// per SM).
    #[arg(long, value_delimiter = ',', default_values_t = ["mma".to_string(), "gw".to_string(), "auto".to_string()])]
    kernels: Vec<String>,
    /// Pool layouts, comma separated: local (neighbouring tokens share most pools, as in a
    /// prompt), spread (pools all over the context).
    #[arg(long, value_delimiter = ',', default_values_t = ["local".to_string(), "spread".to_string()])]
    pools: Vec<String>,
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
    const PATS: [&str; 8] = ["d1", "pair", "pv", "gw", "chain", "pair16", "pv16", "gwdep"];
    let kernels = args
        .kernels
        .iter()
        .map(|k| {
            let pipe = |pre: &str, base: u8| {
                k.strip_prefix(pre)
                    .and_then(|p| PATS.iter().position(|&x| x == p))
                    .map(|p| PromptKernel::Probe(base + p as u8))
            };
            Ok(match k.as_str() {
                "mma" => PromptKernel::Mma,
                "gb" => PromptKernel::Gb,
                "gw" => PromptKernel::Gw,
                "auto" => PromptKernel::Auto,
                "gw-nofill" => PromptKernel::Probe(17),
                "gw-nomma" => PromptKernel::Probe(18),
                "gw-none" => PromptKernel::Probe(19),
                "gw-halfbytes" => PromptKernel::Probe(30),
                "gw-early" => PromptKernel::Variant(20),
                "gw-half" => PromptKernel::Variant(24),
                "gw-early-half" => PromptKernel::Variant(28),
                "gw-1cta" => PromptKernel::Variant(31),
                _ => pipe("pipe-", 32)
                    .or_else(|| pipe("pipe1-", 48))
                    .ok_or_else(|| anyhow::anyhow!("--kernels: unknown kernel {k} (see --help)"))?,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(
        !matches!(kernels.first(), Some(PromptKernel::Probe(_))),
        "--kernels: the first (the reference) must be a real kernel"
    );
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
         TFLOP/s; GB/s of K + V rows gathered",
        args.heads, args.kv_heads, args.peak_tflops
    );
    println!(
        "{:<7} {:>7} {:>6} {:>11} {:>9} {:>7} {:>7} {:>7} {:>6} {:>8} {:>7} {:>18}",
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
        "output bits"
    );
    let mut differ = 0;
    for &n_ctx in &args.ctx {
        anyhow::ensure!(
            n_ctx >= args.tokens + 4 * 512 * 2,
            "--ctx {n_ctx} is too short for --tokens {}",
            args.tokens
        );
        for pools in &args.pools {
            let local = match pools.as_str() {
                "local" => true,
                "spread" => false,
                _ => anyhow::bail!("--pools: unknown layout {pools} (local, spread)"),
            };
            let t = qsa_bench::prompt_ms(
                &mut gpu,
                KvTypes::default(),
                (args.heads, args.kv_heads),
                n_ctx,
                args.tokens,
                args.reps,
                args.rounds,
                local,
                &kernels,
            );
            for (r, name) in t.iter().zip(&args.kernels) {
                let useful = r.keys as f64 * args.heads as f64 * hd * 4.0;
                let issued = r.chunks as f64 * args.kv_heads as f64 * 32.0 * 16.0 * hd * 8.0;
                let bytes = r.keys as f64 * args.kv_heads as f64 * hd * 4.0;
                let s = r.ms * 1e-3;
                let floor_ms = issued / (args.peak_tflops * 1e12) * 1e3;
                let probe = matches!(r.kernel, PromptKernel::Probe(_));
                println!(
                    "{:<7} {:>7} {:>6} {:>11} {:>9.3} {:>7.3} {:>7.1} {:>7.1} {:>5.0}% {:>8.3} {:>7.0} {:>18}",
                    pools,
                    n_ctx,
                    args.tokens,
                    name,
                    r.ms,
                    r.ms / t[0].ms,
                    useful / s * 1e-12,
                    issued / s * 1e-12,
                    floor_ms / r.ms * 100.0,
                    floor_ms,
                    bytes / s * 1e-9,
                    if probe {
                        "probe".to_string()
                    } else {
                        format!("{:016x}", r.bits)
                    }
                );
                differ += (!probe && r.bits != t[0].bits) as usize;
            }
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

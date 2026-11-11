//! `legion paged-qm-bench`: paged attention prefill with heads of 256 on compute capability 12.x
//! through the queries-as-M kernel (#1133) against `paged_prefill_mma`: the edge cases bit for
//! bit, then both timed at a model's shape over prompt chunks at several depths and checked for
//! the same bits there too (a development tool, so a build tree's release binary can check the
//! GPU it runs on; pick it with `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct PagedQmBenchArgs {
    /// Prompt chunks as `n_past:n_tok` (cached positions before the chunk, its tokens), comma
    /// separated.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "0:2048,6144:2048,30720:2048,0:4096,28672:4096,0:8192,0:16384,16384:16384"
    )]
    chunks: Vec<String>,
    /// Query heads, kv heads (heads of 256; default: Qwen3.8-27B's).
    #[arg(long, value_delimiter = ',', default_values_t = [24, 4])]
    heads: Vec<usize>,
    /// Sliding window (0: none).
    #[arg(long, default_value_t = 0)]
    window: usize,
    /// Timed runs per op and round.
    #[arg(long, default_value_t = 10)]
    reps: usize,
    /// Rounds (the best counts).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Skip the edge cases.
    #[arg(long)]
    no_edges: bool,
}

#[cfg(feature = "cuda")]
pub fn run(args: PagedQmBenchArgs) -> anyhow::Result<()> {
    use legion::model::core::paged_qm_bench as bench;
    use legion::model::core::paged_split_bench::{Group, pool};
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut differ = 0;
    if !args.no_edges {
        let (cases, ran, total, bad) = bench::edges(&mut gpu);
        println!(
            "edges: {cases} cases ({ran} on the queries-as-M kernel), {bad} of {total} values differ"
        );
        differ += bad;
    }
    let [n_head, n_head_kv] = args.heads[..] else {
        anyhow::bail!("--heads takes query heads, kv heads");
    };
    let heads = (256, n_head, n_head_kv);
    let mut chunks = Vec::new();
    for c in &args.chunks {
        let (p, t) = c
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("--chunks takes n_past:n_tok, not {c}"))?;
        chunks.push((p.trim().parse::<usize>()?, t.trim().parse::<usize>()?));
    }
    let max_ctx = chunks.iter().map(|(p, t)| p + t).max().unwrap_or(0);
    let n_blocks = max_ctx.div_ceil(64).max(1);
    let kv = pool(&gpu, heads, n_blocks);
    // Block i of the sequence in pool block i * 37 mod n_blocks.
    let table: Vec<u32> = (0..n_blocks).map(|i| (i * 37 % n_blocks) as u32).collect();
    println!(
        "{:>7} {:>6} {:>10} {:>10} {:>7} {:>8} {:>8} {:>6}",
        "n_past", "n_tok", "mma ms", "qm ms", "speedup", "mma TF", "qm TF", "bits"
    );
    for &(n_past, n_tok) in &chunks {
        let past = [n_past];
        let len = (n_past + n_tok).div_ceil(64);
        let tables = [table[..len].to_vec()];
        let g = Group {
            heads,
            n_tok,
            n_past: &past,
            tables: &tables,
            width: len,
            window: args.window,
            ring: 0,
            scale: 1.0 / 16.0,
        };
        // Both orders (mma, qm, qm, mma), each kernel's faster run kept, so warm-up and clock
        // drift do not favour the side that runs second.
        let mut t = |qm| bench::run(&mut gpu, &kv, &g, qm, args.reps, args.rounds);
        let (a1, m1, _) = t(false);
        let (b1, q1, ran) = t(true);
        let (_, q2, _) = t(true);
        let (_, m2, _) = t(false);
        let (mma, qm) = (m1.min(m2), q1.min(q2));
        let bad = a1
            .iter()
            .zip(&b1)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        differ += bad;
        let tf = |ms: f64| 2.0 * bench::macs(&g) / (ms * 1e-3) / 1e12;
        println!(
            "{n_past:>7} {n_tok:>6} {mma:>10.3} {qm:>10.3} {:>7.3} {:>8.1} {:>8.1} {:>6}",
            mma / qm,
            tf(mma),
            tf(qm),
            if !ran {
                "no qm"
            } else if bad == 0 {
                "same"
            } else {
                "DIFFER"
            }
        );
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ from paged_prefill_mma's bits");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: PagedQmBenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("paged-qm-bench needs a CUDA build")
}

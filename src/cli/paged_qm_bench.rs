//! `legion paged-qm-bench`: paged attention prefill with heads of 256 on compute capability 12.x
//! through the queries-as-M kernel (#1133) against `paged_prefill_mma`: the edge cases bit for
//! bit, then both timed at a model's shape over prompt chunks at several depths and checked for
//! the same bits there too (a development tool, so a build tree's release binary can check the
//! GPU it runs on; pick it with `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct PagedQmBenchArgs {
    /// Groups as `n_past:n_tok` (cached positions before the group, its tokens per sequence),
    /// comma separated.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "0:2048,6144:2048,30720:2048,0:4096,28672:4096,0:8192,0:16384,16384:16384"
    )]
    chunks: Vec<String>,
    /// Sequences per group, each at the same depth over its own shuffled table.
    #[arg(long, default_value_t = 1)]
    seqs: usize,
    /// Query heads, kv heads (heads of 256; default: Qwen3.8-27B's).
    #[arg(long, value_delimiter = ',', default_values_t = [24, 4])]
    heads: Vec<usize>,
    /// Sliding window (0: none).
    #[arg(long, default_value_t = 0)]
    window: usize,
    /// The kernel's modes timed against paged_prefill_mma: 1 the default (where its grid is the
    /// faster one), 2 for every group it covers.
    #[arg(long, value_delimiter = ',', default_values_t = [1])]
    modes: Vec<i32>,
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
    // Bad arguments fail before the edge cases run.
    anyhow::ensure!(args.reps > 0, "--reps takes 1 or more timed runs");
    anyhow::ensure!(args.seqs > 0, "--seqs takes 1 or more sequences");
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
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut differ = 0;
    if !args.no_edges {
        for &mode in &args.modes {
            let (cases, ran, total, bad) = bench::edges(&mut gpu, mode);
            println!(
                "edges, mode {mode}: {cases} cases ({ran} on the queries-as-M kernel), {bad} of \
                 {total} values differ"
            );
            differ += bad;
        }
    }
    let max_ctx = chunks.iter().map(|(p, t)| p + t).max().unwrap_or(0);
    let n_blocks = max_ctx.div_ceil(64).max(1);
    let kv = pool(&gpu, heads, n_blocks);
    let seqs = args.seqs.max(1);
    // Block i of sequence s in pool block (i * 37 + s * 11) mod n_blocks.
    let table = |s: usize, len: usize| -> Vec<u32> {
        (0..len)
            .map(|i| ((i * 37 + s * 11) % n_blocks) as u32)
            .collect()
    };
    print!(
        "{:>7} {:>6} {:>4} {:>9}",
        "n_past", "n_tok", "seqs", "mma ms"
    );
    for m in &args.modes {
        print!(" {:>9} {:>6} {:>6}", format!("m{m} ms"), "x", "bits");
    }
    println!();
    for &(n_past, n_tok) in &chunks {
        let past = vec![n_past; seqs];
        let len = (n_past + n_tok).div_ceil(64);
        let tables: Vec<Vec<u32>> = (0..seqs).map(|s| table(s, len)).collect();
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
        let mut t = |mode| bench::run(&mut gpu, &kv, &g, mode, args.reps, args.rounds);
        let (a1, m1, _) = t(0);
        let mut cols = Vec::new();
        for &mode in &args.modes {
            let (b1, q1, ran) = t(mode);
            let (_, q2, _) = t(mode);
            cols.push((b1, q1, q2, ran));
        }
        let (_, m2, _) = t(0);
        let mma = m1.min(m2);
        print!("{n_past:>7} {n_tok:>6} {seqs:>4} {mma:>9.3}");
        for (b1, q1, q2, ran) in cols {
            let qm = q1.min(q2);
            let bad = a1
                .iter()
                .zip(&b1)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            differ += bad;
            let bits = if !ran {
                "no qm"
            } else if bad == 0 {
                "same"
            } else {
                "DIFFER"
            };
            print!(" {qm:>9.3} {:>6.3} {bits:>6}", mma / qm);
        }
        println!();
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

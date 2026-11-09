//! `legion paged-split-bench`: the paged attention of 2-16-token groups (speculative verify
//! steps) on sm_70 / sm_75, split over the key blocks (#1211) against the WMMA kernel: the edge
//! cases bit for bit, then both timed at a model's shape over growing contexts and checked for
//! the same bits there too (a development tool, so a build tree's release binary can check the
//! GPU it runs on; pick it with `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct PagedSplitBenchArgs {
    /// Context lengths (keys before the group), comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [128, 1024, 4096, 16384, 32768, 65536])]
    ctx: Vec<usize>,
    /// Tokens per sequence of the group (MTP k = 3: 4), comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [3, 4, 8])]
    tokens: Vec<usize>,
    /// Sequences of the group, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [1, 4])]
    seqs: Vec<usize>,
    /// Head size, query heads, kv heads (default: Qwen3.8-27B's).
    #[arg(long, value_delimiter = ',', default_values_t = [256, 24, 4])]
    heads: Vec<usize>,
    /// Table entries (a served context's blocks: 4,096 for 262,144 positions).
    #[arg(long, default_value_t = 4096)]
    width: usize,
    /// Timed runs per op and round.
    #[arg(long, default_value_t = 20)]
    reps: usize,
    /// Rounds (the best counts).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Skip the edge cases.
    #[arg(long)]
    no_edges: bool,
}

#[cfg(feature = "cuda")]
pub fn run(args: PagedSplitBenchArgs) -> anyhow::Result<()> {
    use legion::model::core::paged_split_bench::{self as bench, Group};
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut differ = 0;
    if !args.no_edges {
        let shapes = [
            (128, 32, 2),
            (128, 8, 8),
            (128, 16, 4),
            (256, 16, 8),
            (256, 16, 4),
            (256, 24, 4),
            (256, 24, 2),
            (256, 32, 4),
            (512, 16, 2),
            (512, 32, 4),
        ];
        let (cases, total, bad) = bench::edges(&mut gpu, &shapes, &[2, 3, 4, 5, 7, 8, 13, 16]);
        println!("edges: {cases} cases, {bad} of {total} values differ from the WMMA kernel's");
        differ += bad;
    }
    let [d, n_head, n_head_kv] = args.heads[..] else {
        anyhow::bail!("--heads takes head size, query heads, kv heads");
    };
    let heads = (d, n_head, n_head_kv);
    let max_ctx = args.ctx.iter().max().copied().unwrap_or(0) + 16;
    let n_blocks = max_ctx.div_ceil(64);
    anyhow::ensure!(
        n_blocks <= args.width,
        "--width {} is below the longest context",
        args.width
    );
    let kv = bench::pool(&gpu, heads, n_blocks);
    // Block i of every sequence in pool block i * 37 mod n_blocks.
    let table: Vec<u32> = (0..n_blocks).map(|i| (i * 37 % n_blocks) as u32).collect();
    println!(
        "{:>7} {:>6} {:>4} {:>10} {:>10} {:>7} {:>6} {:>6}",
        "ctx", "tokens", "seqs", "wmma ms", "split ms", "speedup", "bits", "first"
    );
    // Which kernel is timed first alternates per case, so warm-up and clock drift do not
    // favour one side in every row.
    let mut case = 0usize;
    for &n_ctx in &args.ctx {
        for &n_tok in &args.tokens {
            for &ns in &args.seqs {
                let n_past = vec![n_ctx; ns];
                let len = (n_ctx + n_tok).div_ceil(64);
                let tables = vec![table[..len].to_vec(); ns];
                let g = Group {
                    heads,
                    n_tok,
                    n_past: &n_past,
                    tables: &tables,
                    width: args.width,
                    window: 0,
                    ring: 0,
                    scale: 1.0 / (d as f32).sqrt(),
                };
                if !bench::splits(&g) {
                    println!("{n_ctx:>7} {n_tok:>6} {ns:>4}  (no split: too many CTAs already)");
                    continue;
                }
                let (bad, _) = bench::compare(&mut gpu, &kv, &g, None);
                differ += bad;
                let split_first = case % 2 == 1;
                case += 1;
                let (wmma, split) = if split_first {
                    let split = bench::time(&mut gpu, &kv, &g, true, args.reps, args.rounds);
                    let wmma = bench::time(&mut gpu, &kv, &g, false, args.reps, args.rounds);
                    (wmma, split)
                } else {
                    let wmma = bench::time(&mut gpu, &kv, &g, false, args.reps, args.rounds);
                    let split = bench::time(&mut gpu, &kv, &g, true, args.reps, args.rounds);
                    (wmma, split)
                };
                println!(
                    "{n_ctx:>7} {n_tok:>6} {ns:>4} {wmma:>10.4} {split:>10.4} {:>7.2} {:>6} {:>6}",
                    wmma / split,
                    if bad == 0 { "same" } else { "DIFFER" },
                    if split_first { "split" } else { "wmma" }
                );
            }
        }
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ from the WMMA kernel's bits");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: PagedSplitBenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("paged-split-bench needs a CUDA build")
}

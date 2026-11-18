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
    /// faster one), 2 for every group it covers; plus 256 times a variant (the same bits, 4 and
    /// 6 query heads a kv head): 1 the chunk loads issued after a part's first MMA step, 2
    /// chunks of 64 dims, 3 the same in 4 stages, 4 warp pairs sharing their rows' P V, 5 warp
    /// pairs with the late loads (e.g. `2,258,1026`).
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
    /// Heads of 512 instead (`--heads 32,4`: Gemma 4 31B's global layers): these kernels against
    /// `paged_prefill_mma` (#1181), `--modes` ignored: `auto` (the warp-pair kernel where the
    /// default picks it), `pair` (the warp-pair kernel for every group), `split` (over the key
    /// blocks as `--paged-verify-split`; other bits, timed for reference).
    #[arg(long, value_delimiter = ',')]
    d512: Vec<String>,
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
    if !args.d512.is_empty() {
        return run_d512(&args, (512, n_head, n_head_kv), &chunks);
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
        // The output starts as NaN, so a value no kernel wrote would match on both sides.
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        let (a1, m1, _) = t(0);
        anyhow::ensure!(finite(&a1), "{n_past}:{n_tok}: non-finite mma output");
        let mut cols = Vec::new();
        for &mode in &args.modes {
            let (b1, q1, ran) = t(mode);
            anyhow::ensure!(
                finite(&b1),
                "{n_past}:{n_tok}: non-finite mode {mode} output"
            );
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

/// `--d512`: the warp-pair kernel (#1181) and the split against `paged_prefill_mma` at heads of
/// 512.
#[cfg(feature = "cuda")]
fn run_d512(
    args: &PagedQmBenchArgs,
    heads: (usize, usize, usize),
    chunks: &[(usize, usize)],
) -> anyhow::Result<()> {
    use legion::model::core::paged_d512_bench as bench;
    use legion::model::core::paged_split_bench::{Group, pool};
    for k in &args.d512 {
        anyhow::ensure!(
            k != "main" && bench::KERNELS.contains(&k.as_str()),
            "--d512 takes {:?}, not {k}",
            &bench::KERNELS[1..]
        );
    }
    anyhow::ensure!(
        heads.1 == 8 * heads.2,
        "--d512 takes 8 query heads a kv head"
    );
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let sms = legion::hw::per_device::sm_count(0).max(0) as usize;
    println!("{sms} SMs");
    let mut differ = 0;
    if !args.no_edges {
        for k in args.d512.iter().filter(|k| *k != "split") {
            let (cases, ran, total, bad) = bench::edges(&mut gpu, k, sms);
            println!(
                "edges, {k}: {cases} cases ({ran} on the pair kernel), {bad} of {total} values differ"
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
        "n_past", "n_tok", "seqs", "main ms"
    );
    for k in &args.d512 {
        print!(" {:>9} {:>6} {:>6}", format!("{k} ms"), "x", "bits");
    }
    println!();
    for &(n_past, n_tok) in chunks {
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
            scale: 1.0 / (heads.0 as f32).sqrt(),
        };
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        // main, each kernel twice, main: each one's faster run kept.
        let mut t = |k: &str| bench::run(&mut gpu, &kv, &g, k, args.reps, args.rounds);
        let (a1, m1) = t("main");
        anyhow::ensure!(finite(&a1), "{n_past}:{n_tok}: non-finite main output");
        let mut cols = Vec::new();
        for k in &args.d512 {
            let (b1, q1) = t(k);
            anyhow::ensure!(finite(&b1), "{n_past}:{n_tok}: non-finite {k} output");
            let (_, q2) = t(k);
            cols.push((k, b1, q1.min(q2)));
        }
        let (_, m2) = t("main");
        let main = m1.min(m2);
        print!("{n_past:>7} {n_tok:>6} {seqs:>4} {main:>9.3}");
        for (k, b1, ms) in cols {
            let bad = a1
                .iter()
                .zip(&b1)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            let bits = match (k.as_str(), bad) {
                ("split", 0) => "same",
                ("split", _) => "other",
                (_, 0) if bench::on_pair(&g, k, sms) => "pair",
                (_, 0) => "mma",
                _ => {
                    differ += bad;
                    "DIFFER"
                }
            };
            print!(" {ms:>9.3} {:>6.3} {bits:>6}", main / ms);
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

//! `legion ep-place` (#803): builds a `--tp-moe-ep-place` placement from router choices that
//! `legion serve --moe-route-record` wrote, and reports each layer's mean max-over-ranks expert
//! count for the contiguous split and the placement.

use std::path::PathBuf;

use clap::Args;
use legion::tp_net::ep_place::{self, Placement, Recorded, Rows};

#[derive(Args)]
pub struct EpPlaceArgs {
    /// Record files written by `legion serve --moe-route-record` (their tokens are joined).
    #[arg(required = true)]
    records: Vec<PathBuf>,
    /// Where to write the placement (JSON, for `legion serve --tp-moe-ep-place`). Without it,
    /// only the report.
    #[arg(long, short)]
    out: Option<PathBuf>,
    /// Ranks to split the experts among (--tp-rank's count).
    #[arg(long, default_value_t = 2)]
    ranks: usize,
    /// The share of each layer's recorded tokens, the last ones, held out of the split and
    /// reported apart (how the placement does on tokens it was not built from).
    #[arg(long, default_value_t = 0.0)]
    holdout: f64,
    /// Sweeps of pairwise swaps after the greedy split.
    #[arg(long, default_value_t = 8)]
    passes: usize,
    /// Report this placement instead of building one.
    #[arg(long, conflicts_with = "out")]
    eval: Option<PathBuf>,
}

pub fn run(args: EpPlaceArgs) -> anyhow::Result<()> {
    anyhow::ensure!(args.ranks >= 2, "--ranks must be at least 2");
    anyhow::ensure!(
        (0.0..1.0).contains(&args.holdout),
        "--holdout must be in [0, 1)"
    );
    let mut rec = Recorded::default();
    for p in &args.records {
        let bytes = std::fs::read(p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
        let r = Recorded::parse(&bytes).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
        rec.extend(r)
            .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    }
    anyhow::ensure!(!rec.layers.is_empty(), "no records");
    let (n_expert, ranks) = (rec.n_expert, args.ranks);
    anyhow::ensure!(
        n_expert.is_multiple_of(ranks),
        "{n_expert} experts do not split evenly among {ranks} ranks"
    );
    // Each layer's tokens, split into the ones to build from and the held-out tail.
    let parts: Vec<(usize, Rows, Rows)> = rec
        .layers
        .iter()
        .map(|(&il, rows)| {
            let n = rows.tokens();
            let train = n - (n as f64 * args.holdout).round() as usize;
            (il, rows.slice(0..train), rows.slice(train..n))
        })
        .collect();
    let place = match &args.eval {
        Some(p) => {
            let place = Placement::load(p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
            place
                .fits(n_expert, ranks)
                .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
            place
        }
        None => {
            let owners = split_all(&parts, n_expert, ranks, args.passes);
            Placement {
                n_expert,
                ranks,
                layers: parts.iter().map(|p| p.0).zip(owners).collect(),
            }
        }
    };
    let contiguous = ep_place::contiguous(n_expert, ranks);
    let held = args.holdout > 0.0;
    println!(
        "{n_expert} experts, {ranks} ranks; mean max-over-ranks experts per token (floor k/ranks)"
    );
    if held {
        println!("layer  tokens  k   contiguous  placed  | held out  contiguous  placed");
    } else {
        println!("layer  tokens  k   contiguous  placed");
    }
    let mut sums = [0f64; 4];
    for (il, train, test) in &parts {
        let owner = place.owner(*il);
        let m = [
            ep_place::mean_max(train, &contiguous, ranks),
            ep_place::mean_max(train, &owner, ranks),
            ep_place::mean_max(test, &contiguous, ranks),
            ep_place::mean_max(test, &owner, ranks),
        ];
        for (s, v) in sums.iter_mut().zip(m) {
            *s += v;
        }
        let line = format!(
            "{il:>5}  {:>6}  {:>2}  {:>10.3}  {:>6.3}",
            train.tokens(),
            train.k,
            m[0],
            m[1]
        );
        if held {
            println!(
                "{line}  | {:>8}  {:>10.3}  {:>6.3}",
                test.tokens(),
                m[2],
                m[3]
            );
        } else {
            println!("{line}");
        }
    }
    let n = parts.len() as f64;
    let line = format!(
        "mean               {:>10.3}  {:>6.3}",
        sums[0] / n,
        sums[1] / n
    );
    if held {
        println!(
            "{line}  |           {:>10.3}  {:>6.3}",
            sums[2] / n,
            sums[3] / n
        );
    } else {
        println!("{line}");
    }
    if let Some(out) = &args.out {
        std::fs::write(out, place.to_json())
            .map_err(|e| anyhow::anyhow!("{}: {e}", out.display()))?;
        println!("placement written: {}", out.display());
    }
    Ok(())
}

/// [`ep_place::split`] of every layer's training tokens, layers in parallel.
fn split_all(
    parts: &[(usize, Rows, Rows)],
    n_expert: usize,
    ranks: usize,
    passes: usize,
) -> Vec<Vec<u8>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(vec![Vec::new(); parts.len()]);
    std::thread::scope(|s| {
        for _ in 0..threads.min(parts.len()) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((_, train, _)) = parts.get(i) else {
                        break;
                    };
                    let owner = ep_place::split(train, n_expert, ranks, passes);
                    out.lock().unwrap()[i] = owner;
                }
            });
        }
    });
    out.into_inner().unwrap()
}

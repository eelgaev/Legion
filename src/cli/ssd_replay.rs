//! `legion ssd-replay`: replays an SSD expert tier routing trace (`LEGION_SSD_EXPERT_TRACE`)
//! under several slot policies and prints their hit rates (a development tool, #605).

use std::path::PathBuf;

use clap::Args;
use legion::residency::PROFILE_SEED;
use legion::residency::profile::Profile;
use legion::residency::ssd_experts::lookahead::{self, Lookahead};
use legion::residency::ssd_experts::replay::{self, Outcome, Policy, Setup};
use legion::residency::ssd_experts::trace::{Predictor, Trace};

#[derive(Args)]
pub struct SsdReplayArgs {
    /// The trace a `legion serve` run with LEGION_SSD_EXPERT_TRACE=<path> wrote.
    trace: PathBuf,
    /// Slots per layer to replay at, comma separated (default: the traced run's).
    #[arg(long, value_delimiter = ',')]
    slots: Vec<usize>,
    /// An expert profile (--expert-profile format) from an earlier run on other prompts: adds
    /// the seeded rows (its hottest experts fill the slots before the first step).
    #[arg(long)]
    profile: Option<PathBuf>,
    /// The routing heat a seeded row's hottest expert starts with.
    #[arg(long, default_value_t = PROFILE_SEED)]
    seed_top: f32,
    /// Also write this trace's routing counts as an expert profile, to seed another replay.
    #[arg(long)]
    save_profile: Option<PathBuf>,
    /// A trace with router lookahead predictions (#611): the margins (experts per token beyond
    /// the routed count) its recall and prefetch tables show.
    #[arg(long, value_delimiter = ',', default_values_t = [0, 4, 8, 16])]
    margins: Vec<usize>,
    /// Only the lookahead tables (skip the slot policies).
    #[arg(long)]
    lookahead_only: bool,
}

pub fn run(args: SsdReplayArgs) -> anyhow::Result<()> {
    let trace = Trace::read(&args.trace)?;
    let h = &trace.header;
    if let Some(p) = &args.save_profile {
        replay::profile(&trace).write(p)?;
        println!("profile written: {}", p.display());
    }
    let seed = match &args.profile {
        Some(p) => Some(replay::seed_heat(
            &Profile::read(p)?,
            &h.layers,
            h.n_expert,
            args.seed_top,
        )),
        None => None,
    };
    let slots = if args.slots.is_empty() {
        vec![h.n_slots]
    } else {
        args.slots.clone()
    };
    let small = trace
        .records
        .iter()
        .filter(|r| r.ord == 0 && !r.big)
        .count();
    let big = trace.records.iter().filter(|r| r.ord == 0 && r.big).count();
    println!(
        "{}: {} layers, {} experts, traced at {} slots; {small} small and {big} big steps per layer",
        args.trace.display(),
        h.layers.len(),
        h.n_expert,
        h.n_slots
    );
    if !trace.predictions.is_empty() {
        lookahead_tables(&trace, &slots, &args.margins);
    }
    if args.lookahead_only {
        return Ok(());
    }
    let mut rows: Vec<(String, Policy, bool)> = vec![("LRU (today)".into(), Policy::Lru, false)];
    for prompts in [false, true] {
        for decay in [0.8, 0.9, 0.95, 0.99, 0.999] {
            let p = if prompts { ", prompts count" } else { "" };
            rows.push((
                format!("heat x{decay}{p}"),
                Policy::Heat { decay, prompts },
                false,
            ));
        }
    }
    for decay in [0.95, 0.99] {
        for frac in [0.25, 0.5, 0.75] {
            rows.push((
                format!("pinned {:.0}% (heat x{decay}) + LRU", frac * 100.0),
                Policy::Pinned { frac, decay },
                false,
            ));
        }
    }
    rows.push(("LRU-2".into(), Policy::Lru2, false));
    if seed.is_some() {
        let seeded: Vec<(String, Policy, bool)> = rows
            .iter()
            .map(|(n, p, _)| (format!("{n}, profile seed"), *p, true))
            .collect();
        rows.extend(seeded);
    }
    rows.push(("Belady (ceiling)".into(), Policy::Belady, false));
    print!("| policy |");
    for s in &slots {
        print!(" {s}/{} hit | misses/token |", h.n_expert);
    }
    println!();
    print!("|---|");
    for _ in &slots {
        print!("---|---|");
    }
    println!();
    for (name, policy, seeded) in rows {
        print!("| {name} |");
        for &n_slots in &slots {
            let o: Outcome = replay::replay(
                &trace,
                &Setup {
                    policy,
                    n_slots,
                    seed: if seeded { seed.clone() } else { None },
                },
            );
            print!(
                " {:.2}% | {:.1} |",
                100.0 * o.hit_rate(),
                o.misses_per_token()
            );
        }
        println!();
    }
    Ok(())
}

/// The router lookahead's tables (#611): each predictor's recall, then LRU with its prefetch
/// against LRU alone and Belady, per slot count.
fn lookahead_tables(trace: &Trace, slots: &[usize], margins: &[usize]) {
    let k = lookahead::n_used(trace);
    let present: Vec<Predictor> = Predictor::ALL
        .into_iter()
        .filter(|p| trace.predictions.iter().any(|x| x.predictor == *p))
        .collect();
    for &n_slots in slots {
        println!();
        println!("Recall of the routed experts (distinct per small step), {n_slots} slots:");
        println!();
        println!(
            "| predictor | m | steps predicted | recall | LRU misses named | named per step |"
        );
        println!("|---|---|---|---|---|---|");
        for &p in &present {
            for &m in margins {
                let r = lookahead::recall(trace, p, m, n_slots);
                println!(
                    "| {} | {m} (top {}) | {:.1}% | {:.2}% | {:.2}% | {:.1} |",
                    p.name(),
                    k + m,
                    100.0 * r.coverage(),
                    100.0 * r.recall(),
                    100.0 * r.miss_recall(),
                    r.named as f64 / r.predicted_steps.max(1) as f64
                );
            }
        }
        println!();
        println!("Prefetch replays, {n_slots} slots (reads are experts per token, all layers):");
        println!();
        println!("| prefetch | hit | step reads | ahead reads | wasted | all reads |");
        println!("|---|---|---|---|---|---|");
        let base = |policy| {
            replay::replay(
                trace,
                &Setup {
                    policy,
                    n_slots,
                    seed: None,
                },
            )
        };
        for (name, o) in [
            ("none (LRU, today)", base(Policy::Lru)),
            ("none (Belady, ceiling)", base(Policy::Belady)),
        ] {
            println!(
                "| {name} | {:.2}% | {:.1} | 0 | 0 | {:.1} |",
                100.0 * o.hit_rate(),
                o.misses_per_token(),
                o.misses_per_token()
            );
        }
        // Each margin with every rank read ahead, then fewer ranks than routed (#615,
        // `--ssd-prefetch-ranks`).
        let mut rows: Vec<(usize, usize)> = margins.iter().map(|&m| (m, usize::MAX)).collect();
        rows.extend((k / 2..k).map(|r| (0, r)));
        for &p in &present {
            for &(m, ranks) in &rows {
                let o = lookahead::replay(
                    trace,
                    &Lookahead {
                        predictor: p,
                        margin: m,
                        n_slots,
                        budget: usize::MAX,
                        ranks,
                    },
                );
                let name = if ranks < usize::MAX {
                    format!("{}, top {ranks}", p.name())
                } else {
                    format!("{}, m={m}", p.name())
                };
                println!(
                    "| {name} | {:.2}% | {:.1} | {:.1} | {:.1} | {:.1} |",
                    100.0 * o.hit_rate(),
                    o.per_token(o.misses),
                    o.per_token(o.prefetched),
                    o.per_token(o.wasted),
                    o.per_token(o.reads())
                );
            }
        }
    }
}

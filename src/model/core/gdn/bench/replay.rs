//! `legion gdn-bench --replay` (#836 part 31): a deferred commit run inside the next verify step
//! ([`rows_replay`] of the packed rows the last verify step stored, which then stores its own
//! there, part 39) against the commit and the verify step on their own ([`commit_slots`], then
//! [`rows`]) on the GPU, as `--gdn-commit-in-verify` and the default run them: per set of keep
//! counts the states and the rows must be the same bits, the fused step must give the same bits
//! again when it runs a second time from the same inputs, and the packed rows it leaves must hold
//! its tokens' k, v, g and beta (and the rest as it found them). Each is timed per layer over
//! `layers` layers with their own states.

use super::super::*;
use super::{COMMIT_KEEPS, COMMIT_POOL, Config, noise};
use crate::ggml::{Backend, ComputeAllocator, Graph};

/// Tokens per sequence of the verify step after the commit: C=16 rounds of three drafts.
pub const REPLAY_VERIFY: [i64; 2] = [4, 1];

/// The fused step's kernel variants compared (`src/cuda/gdn.cu`'s `replay_variant`): the
/// per-lane kernel first (the row's `fused`), then the one with the tokens' inputs in shared
/// memory, then the one for few sequences (#689).
pub const REPLAY_VARIANTS: [(i32, &str); 3] = [(0, "plain"), (1, "smem"), (2, "few")];

/// Tokens a sequence's deferred commit keeps in [`run_replay`]'s sets of `n` sequences
/// ([`Config::replay_seqs`]): each set gives every sequence the same count.
pub const FEW_KEEPS: [i32; 5] = [0, 1, 3, 6, 8];

/// One set of keep counts and verify step size ([`run_replay`]).
#[derive(Clone, Debug)]
pub struct ReplayRow {
    /// Tokens each slot's deferred commit keeps (0: none).
    pub keep: Vec<i32>,
    /// Verify tokens per sequence.
    pub verify: i64,
    /// State values whose bits differ between the separate and the fused pair.
    pub differ_state: usize,
    /// Row values whose bits differ between the separate and the fused pair.
    pub differ_rows: usize,
    /// State and row values whose bits differ between two runs of the fused pair.
    pub differ_rerun: usize,
    /// Packed row values the fused step left other than its tokens' inputs (or, past them, what
    /// was there), and counters it did not set back to 0.
    pub differ_store: usize,
    /// Seconds per layer: commit then verify, and the fused verify (the replay, then the step's
    /// store into the packed rows).
    pub separate: f64,
    pub fused: f64,
    /// Seconds per layer of the separate pair's commit on its own.
    pub commit: f64,
    /// Per [`REPLAY_VARIANTS`] entry: the fused step's seconds per layer, and its state, row and
    /// store values whose bits differ from the separate pair's (the first is `fused`, `differ_*`).
    pub variants: Vec<(f64, usize)>,
}

/// The tensors both pairs read and write, in their own buffer (as the spec and state pools).
struct Pools<'a> {
    /// The verify pool's stored inputs of the committed step, `[a, COMMIT_POOL, n]`.
    pool: [Tensor<'a>; 5],
    /// The next verify step's inputs.
    x: [Tensor<'a>; 5],
    keep: Tensor<'a>,
    /// The fused step's counters (`[n_k * n]`, zeros).
    arrive: Tensor<'a>,
    /// Per layer: its state and its packed replay rows.
    states: Vec<Tensor<'a>>,
    replay: Vec<Tensor<'a>>,
}

impl<'a> Pools<'a> {
    fn new(ctx: &'a Context, h: i64, nk: i64, ns: i64, nt: i64, layers: usize) -> Self {
        let t = |ne: &[i64]| ctx.new_tensor(types::F32, ne);
        let p = COMMIT_POOL;
        Self {
            pool: [
                t(&[S * nk, p, ns]),
                t(&[S * nk, p, ns]),
                t(&[S * h, p, ns]),
                t(&[h, p, ns]),
                t(&[h, p, ns]),
            ],
            x: [
                t(&[S, nk, nt, ns]),
                t(&[S, nk, nt, ns]),
                t(&[S, h, nt, ns]),
                t(&[1, h, nt, ns]),
                t(&[1, h, nt, ns]),
            ],
            keep: ctx.new_tensor(types::I32, &[ns]),
            arrive: ctx.new_tensor(types::I32, &[nk * ns]),
            states: (0..layers).map(|_| t(&[S, S, h, ns])).collect(),
            replay: (0..layers)
                .map(|_| t(&[replay_row(nk, h), p, ns]))
                .collect(),
        }
    }

    /// The same inputs and starting states every time (q, k about unit length per head, log
    /// decays in (-1, 0], betas in (0, 1), as in [`super::run`]); the packed rows hold the
    /// pool's tokens, as the last verify step left them. Returns the packed rows, and what the
    /// fused step must leave there: the step's tokens' k, v, g and beta over theirs.
    fn fill(&self, keep: &[i32]) -> (Vec<f32>, Vec<f32>) {
        let mut host = Vec::new();
        for (i, ts) in [self.pool, self.x].iter().enumerate() {
            let s = 10 * i as u64;
            let f: [&dyn Fn(f32) -> f32; 5] = [
                &|x| 0.15 * x,
                &|x| 0.15 * x,
                &|x| x,
                &|x| -0.5 * (x + 1.0),
                &|x| 0.5 * (x + 1.0),
            ];
            let mut set = Vec::new();
            for (j, t) in ts.iter().enumerate() {
                let x: Vec<f32> = noise(t.elements(), s + 1 + j as u64)
                    .iter()
                    .map(|&x| f[j](x))
                    .collect();
                t.set_f32(&x);
                set.push(x);
            }
            host.push(set);
        }
        for st in &self.states {
            st.set_f32(
                &noise(st.elements(), 6)
                    .iter()
                    .map(|&x| 0.1 * x)
                    .collect::<Vec<_>>(),
            );
        }
        let [row, p, ns, _] = self.replay[0].ne().map(|n| n as usize);
        let nt = self.x[2].ne()[2] as usize;
        // Per source: floats per token.
        let widths: Vec<usize> = self.pool.iter().map(|t| t.ne()[0] as usize).collect();
        let pack = |dst: &mut [f32], set: &[Vec<f32>], n_tok: usize, from: usize| {
            for s in 0..ns {
                for t in 0..n_tok {
                    let mut off = (s * p + t) * row;
                    for (j, (x, &w)) in set.iter().zip(&widths).enumerate() {
                        if j >= from {
                            let at = (s * n_tok + t) * w;
                            dst[off..off + w].copy_from_slice(&x[at..at + w]);
                        }
                        off += w;
                    }
                }
            }
        };
        let mut packed = vec![0f32; row * p * ns];
        pack(&mut packed, &host[0], p, 0);
        let mut stored = packed.clone();
        // All but q.
        pack(&mut stored, &host[1], nt, 1);
        for r in &self.replay {
            r.set_f32(&packed);
        }
        self.keep.set_i32(keep);
        self.arrive.set_i32(&vec![0; self.arrive.elements()]);
        (packed, stored)
    }

    /// Every state, then every layer's rows of `rows`.
    fn read(&self, rows: &[Tensor<'_>]) -> Vec<f32> {
        let mut out = Vec::new();
        for st in &self.states {
            out.extend(st.to_f32());
        }
        for r in rows {
            out.extend(r.to_f32());
        }
        out
    }
}

/// A pair's graphs: the commit (the separate pair only) and the verify step, with its rows per
/// layer.
struct Pair<'a> {
    commit: Option<Graph<'a>>,
    verify: Graph<'a>,
    rows: Vec<Tensor<'a>>,
}

fn pair<'a>(cc: &'a Context, cv: &'a Context, p: &Pools<'a>, fused: bool) -> Pair<'a> {
    let n = 8 * p.states.len() + 16;
    let mut commit = cc.new_graph(n);
    let mut verify = cv.new_graph(n);
    let [q, k, v, g, beta] = p.x;
    let [pq, pk, pv, pg, pb] = p.pool;
    let ns = p.keep.elements() as i64;
    let (nk, h) = (q.ne()[1], v.ne()[1]);
    let tok = |t: Tensor<'a>, a: i64, b: i64| cc.reshape_4d(t, [a, b, COMMIT_POOL, ns]);
    let mut rows = Vec::new();
    for (&st, &r) in p.states.iter().zip(&p.replay) {
        if fused {
            // The next verify step: the replay of the packed rows (the last verify step's
            // store), and its own store there.
            let out =
                rows_replay(cv, q, k, v, g, beta, st, p.keep, r, (p.arrive, 0), None).as_output();
            verify.expand(out);
            rows.push(out);
        } else {
            commit.expand(commit_slots(
                cc,
                tok(pq, S, nk),
                tok(pk, S, nk),
                tok(pv, S, h),
                tok(pg, 1, h),
                tok(pb, 1, h),
                st,
                p.keep,
            ));
            let out = super::super::rows(cv, q, k, v, g, beta, st).as_output();
            verify.expand(out);
            rows.push(out);
        }
    }
    Pair {
        commit: (!fused).then_some(commit),
        verify,
        rows,
    }
}

/// The separate and the fused pair over [`COMMIT_KEEPS`] (or, with [`Config::replay_seqs`],
/// sets of that many sequences each keeping one of [`FEW_KEEPS`]) and [`REPLAY_VERIFY`] (or
/// [`Config::replay_verify`]): bits (one layer) and time (`cfg.layers` layers).
pub fn run_replay(gpu: &mut Backend, cfg: &Config) -> Result<Vec<ReplayRow>, String> {
    let (h, nk) = cfg.heads;
    let err = |e: crate::ggml::GgmlError| format!("gdn-bench: {e:?}");
    let keeps: Vec<Vec<i32>> = match cfg.replay_seqs {
        Some(n) => FEW_KEEPS.iter().map(|&k| vec![k; n]).collect(),
        None => COMMIT_KEEPS.iter().map(|k| k.to_vec()).collect(),
    };
    let verify = if cfg.replay_verify.is_empty() {
        REPLAY_VERIFY.to_vec()
    } else {
        cfg.replay_verify.clone()
    };
    let mut out = Vec::new();
    for &nt in &verify {
        for keep in &keeps {
            let ns = keep.len() as i64;
            // A run of a pair from the filled inputs: the states and rows after it, the
            // fused step's store differences, and the times per layer.
            let mut once = |fused: bool, layers: usize, timed: bool| {
                let pc = Context::new(32 + 2 * layers, 0);
                let pools = Pools::new(&pc, h, nk, ns, nt, layers);
                pc.alloc_tensors(gpu.buffer_type()).map_err(err)?;
                let n = 8 * layers + 16;
                let (cc, cv) = (Context::new(4 * n, n), Context::new(4 * n, n));
                let x = pair(&cc, &cv, &pools, fused);
                if let Some(c) = &x.commit {
                    cc.alloc_graph(c, ComputeAllocator::new(gpu.buffer_type()))
                        .map_err(err)?;
                }
                cv.alloc_graph(&x.verify, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(err)?;
                let (packed, stored) = pools.fill(&keep);
                if let Some(c) = &x.commit {
                    gpu.compute(c).map_err(err)?;
                }
                gpu.compute(&x.verify).map_err(err)?;
                let bits = pools.read(&x.rows);
                let differ = |a: &[f32], b: &[f32]| {
                    a.iter()
                        .zip(b)
                        .filter(|(p, q)| p.to_bits() != q.to_bits())
                        .count()
                };
                let want = if fused { &stored } else { &packed };
                let store = pools
                    .replay
                    .iter()
                    .map(|r| differ(&r.to_f32(), want))
                    .sum::<usize>()
                    + pools.arrive.to_i32().iter().filter(|&&c| c != 0).count();
                let (mut best, mut first) = (f64::INFINITY, f64::INFINITY);
                if timed {
                    // The states drift from run to run; the time does not depend on them.
                    for _ in 0..cfg.rounds {
                        let t = std::time::Instant::now();
                        for _ in 0..cfg.reps {
                            if let Some(c) = &x.commit {
                                gpu.compute(c).map_err(err)?;
                            }
                            gpu.compute(&x.verify).map_err(err)?;
                        }
                        let s = t.elapsed().as_secs_f64() / (cfg.reps * layers) as f64;
                        best = best.min(s);
                        if let Some(c) = &x.commit {
                            let t = std::time::Instant::now();
                            for _ in 0..cfg.reps {
                                gpu.compute(c).map_err(err)?;
                            }
                            let s = t.elapsed().as_secs_f64() / (cfg.reps * layers) as f64;
                            first = first.min(s);
                        }
                    }
                }
                Ok::<_, String>((bits, store, best, first))
            };
            let differ = |x: &[f32], y: &[f32]| {
                x.iter()
                    .zip(y)
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count()
            };
            let (a, ..) = once(false, 1, false)?;
            let (_, _, separate, commit) = once(false, cfg.layers, true)?;
            let mut variants = Vec::new();
            for (v, _) in REPLAY_VARIANTS {
                set_replay_variant(v);
                let bits = once(true, 1, false).and_then(|(b, store, ..)| {
                    let (_, _, secs, _) = once(true, cfg.layers, true)?;
                    Ok((b, store, secs))
                });
                set_replay_variant(-1);
                let (b, store, secs) = bits?;
                variants.push((secs, differ(&a, &b) + store));
            }
            set_replay_variant(0);
            let fresh = once(true, 1, false);
            let rerun = fresh.and_then(|(b, store, ..)| {
                let (c, ..) = once(true, 1, false)?;
                Ok((b, store, c))
            });
            set_replay_variant(-1);
            let (b, store, c) = rerun?;
            assert_eq!(a.len(), b.len());
            let n_state = (S * S * h * ns) as usize;
            let fused = variants[0].0;
            out.push(ReplayRow {
                keep: keep.to_vec(),
                verify: nt,
                differ_state: differ(&a[..n_state], &b[..n_state]),
                differ_rows: differ(&a[n_state..], &b[n_state..]),
                differ_rerun: differ(&b, &c),
                differ_store: store,
                separate,
                fused,
                commit,
                variants,
            });
        }
    }
    Ok(out)
}

//! The share's worker pool timed on this process's CPUs (`legion cpu-bench pool`, #637): jobs
//! of `tokens` x `experts` host pairs (every token choosing the same experts, as a verify
//! step's tokens often do) over more experts than the caches hold, at several core and SMT
//! counts. Bind the process with `numactl` to pick the socket the threads and the experts'
//! pages are on.

use std::sync::atomic::AtomicU32;
use std::time::Instant;

use super::HostExperts;
use super::pool::{Exchange, Job, Pool, run_now};
use crate::cpu::microbench::{ExpertShape, Rng, random_experts};
use crate::hw::numa::{current_affinity, pin_current_thread};

/// What [`run`] measures.
#[derive(Debug, Clone)]
pub struct PoolBench {
    pub shape: ExpertShape,
    /// Experts in memory, cycled through by the jobs.
    pub n_expert: usize,
    /// Cores of the pool (the first of the allowed CPUs' cores).
    pub cores: Vec<usize>,
    /// Threads per core.
    pub smt: Vec<usize>,
    /// Tokens per job.
    pub tokens: Vec<usize>,
    /// Experts per job (each token's host pairs).
    pub experts: Vec<usize>,
    /// Timed jobs per point (after a few warm-up jobs).
    pub jobs: usize,
}

/// The allowed CPUs grouped by core, in order.
fn cores_of(cpus: &[u32]) -> Vec<Vec<u32>> {
    let mut cores: Vec<Vec<u32>> = Vec::new();
    for &c in cpus {
        let sib = crate::hw::host_core::siblings(c);
        match cores.iter_mut().find(|k| k.iter().any(|o| sib.contains(o))) {
            Some(core) => core.push(c),
            None => cores.push(vec![c]),
        }
    }
    cores
}

/// Runs the grid and calls `line` with each table line. The calling thread (the "GPU" that
/// raises the flags and spins for the result) is pinned to the last allowed core, which the
/// pools never use.
pub fn run(cfg: &PoolBench, line: &mut dyn FnMut(String)) -> std::io::Result<()> {
    let shape = cfg.shape;
    let (ne, nf) = (shape.n_embd, shape.n_ff);
    let kinds = random_experts(shape, cfg.n_expert);
    let he = HostExperts {
        base: kinds.each_ref().map(|k| k.as_ptr() as usize),
        expert_bytes: [shape.gate_bytes(), shape.gate_bytes(), shape.down_bytes()],
        up_gate: shape.up_gate,
        down: shape.down,
        n_embd: ne,
        n_ff: nf,
        n_expert: cfg.n_expert,
    };
    let allowed = current_affinity()?;
    let cores = cores_of(&allowed);
    let Some((own, pool_cores)) = cores.split_last() else {
        return Ok(());
    };
    let previous = pin_current_thread(&own[..1])?;
    let mb = shape.bytes() as f64 / 1e6;
    line(format!(
        "share pool: {:?}/{:?} experts, {ne} x {nf}, {mb:.2} MB each, {} cycled; {} cores \
         allowed (the last runs the caller)",
        shape.up_gate,
        shape.down,
        cfg.n_expert,
        cores.len()
    ));
    line("cores  smt  threads  tokens  experts   us/job  us/expert    GB/s".into());
    let mut rng = Rng::new(1);
    let mut seq = 0u32;
    for &smt in &cfg.smt {
        for &nc in &cfg.cores {
            if nc > pool_cores.len() {
                continue;
            }
            let cpus: Vec<u32> = pool_cores[..nc]
                .iter()
                .flat_map(|c| c.iter().copied().take(smt))
                .collect();
            let pool = Pool::new(&cpus, cpus.len());
            let queue = pool.add_queue();
            let flags = [AtomicU32::new(seq), AtomicU32::new(seq)];
            for &n_tok in &cfg.tokens {
                for &n_used in &cfg.experts {
                    let mut ids = vec![-1i32; n_tok * n_used];
                    let x: Vec<f32> = (0..n_tok * ne).map(|_| rng.unit()).collect();
                    let mut out = vec![0.0f32; n_tok * n_used * ne];
                    let mut times = Vec::with_capacity(cfg.jobs);
                    for j in 0..cfg.jobs + 8 {
                        for (i, id) in ids.iter_mut().enumerate() {
                            *id = ((j * n_used + i % n_used) % cfg.n_expert) as i32;
                        }
                        seq = seq.wrapping_add(1);
                        let job = Job {
                            seq,
                            layer: 0,
                            n_tok,
                            n_used,
                            experts: he,
                            exchange: Exchange {
                                ready: &flags[0],
                                done: &flags[1],
                                arrived: std::ptr::null(),
                                ids: ids.as_ptr(),
                                x: x.as_ptr(),
                                out: out.as_mut_ptr(),
                                max_tok: n_tok,
                                max_used: n_used,
                                n_embd: ne,
                            },
                        };
                        let t = Instant::now();
                        // SAFETY: the buffers and `kinds` outlive the call.
                        unsafe { run_now(&pool, queue, job) };
                        if j >= 8 {
                            times.push(t.elapsed().as_secs_f64());
                        }
                    }
                    times.sort_by(f64::total_cmp);
                    let t = times[times.len() / 2];
                    line(format!(
                        "{nc:>5} {smt:>4} {:>8} {n_tok:>7} {n_used:>8} {:>8.1} {:>10.1} {:>7.2}",
                        cpus.len(),
                        t * 1e6,
                        t * 1e6 / n_used as f64,
                        mb * n_used as f64 / 1e3 / t,
                    ));
                }
            }
            drop(pool);
        }
    }
    pin_current_thread(&previous)?;
    Ok(())
}

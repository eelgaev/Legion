//! The share's worker pool timed on this process's CPUs (`legion cpu-bench pool`, #637): jobs
//! of `tokens` x `experts` host pairs (every token choosing the same experts, as a verify
//! step's tokens often do) over more experts than the caches hold, at several core and SMT
//! counts. Bind the process with `numactl` to pick the socket the threads and the experts'
//! pages are on.
//!
//! `exact`: the exact share's jobs (#37, x86 with AVX2, FMA and F16C), driven as the GPU
//! drives them: "ready", the pool's gate and up sums, "gate/up done", the GPU's SwiGLU blocks
//! (`act_us` microseconds, the GPU's round trip, before "act ready"), the down rows, "done".
//! The columns then also give the time to "gate/up done" and from "act ready" to "done".

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use super::pool::{ExactExchange, Exchange, Job, Pool, Split, run_now};
use super::{HostExperts, THRESHOLD_ONE};
use crate::cpu::gpu_exact::{self, Q8_1_BLOCK_BYTES};
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
    /// The exact share's jobs instead of the host kernels' (see the module docs).
    pub exact: bool,
    /// The exact jobs' wait between "gate/up done" and "act ready", in microseconds.
    pub act_us: u64,
    /// Cuts of each phase into tasks ([`Split::tasks_per_thread`]), each a row of the table.
    pub tasks_per_thread: Vec<usize>,
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

/// Random Q8_1 blocks (scale, sum, 32 values), as the GPU's quantize_x writes them.
fn q8_1_blocks(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * Q8_1_BLOCK_BYTES);
    for _ in 0..n {
        let d = gpu_exact::f32_to_f16(rng.unit().abs() * 0.02 + 1e-4);
        let s = gpu_exact::f32_to_f16(rng.unit() * 4.0);
        out.extend_from_slice(&d.to_le_bytes());
        out.extend_from_slice(&s.to_le_bytes());
        out.extend((0..32).map(|_| (rng.unit() * 127.0) as i8 as u8));
    }
    out
}

/// Spins until `flag` reaches `seq`.
fn spin_until(flag: &AtomicU32, seq: u32) {
    let start = Instant::now();
    while flag.load(Ordering::Acquire).wrapping_sub(seq) >= u32::MAX / 2 {
        assert!(start.elapsed() < Duration::from_secs(30), "pool stuck");
        std::hint::spin_loop();
    }
}

/// The exact share's buffers of one job size.
struct ExactBufs {
    xq: Vec<u8>,
    aq: Vec<u8>,
    gu: Vec<f32>,
    /// gate/up done, act ready, act arrived, n_miss.
    flags: [AtomicU32; 4],
    thr: [AtomicU32; 1],
    cost: [f32; 1],
}

/// Runs the grid and calls `line` with each table line. The calling thread (the "GPU" that
/// raises the flags and spins for the result) is pinned to the last allowed core, which the
/// pools never use.
pub fn run(cfg: &PoolBench, line: &mut dyn FnMut(String)) -> std::io::Result<()> {
    let shape = cfg.shape;
    let (ne, nf) = (shape.n_embd, shape.n_ff);
    if cfg.exact
        && (!gpu_exact::simd()
            || gpu_exact::units_per_row(shape.up_gate, ne).is_none()
            || gpu_exact::units_per_row(shape.down, nf).is_none()
            || !ne.is_multiple_of(256)
            || !nf.is_multiple_of(32))
    {
        line("share pool: no exact host kernels for these experts on this CPU".into());
        return Ok(());
    }
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
        "share pool{}: {:?}/{:?} experts, {ne} x {nf}, {mb:.2} MB each, {} cycled; {} cores \
         allowed (the last runs the caller){}",
        if cfg.exact { " (exact)" } else { "" },
        shape.up_gate,
        shape.down,
        cfg.n_expert,
        cores.len(),
        if cfg.exact {
            format!("; act ready {} us after gate/up done", cfg.act_us)
        } else {
            String::new()
        }
    ));
    line(
        "cores  smt  threads  tasks/t  tokens  experts   us/job  us/expert    GB/s  gate_up_us  \
         down_us"
            .into(),
    );
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
            for &tpt in &cfg.tasks_per_thread {
                let split = Split {
                    tasks_per_thread: tpt.max(1),
                    ..Split::default()
                };
                let pool = Pool::with_split(&cpus, cpus.len(), split);
                let queue = pool.add_queue();
                let flags = [AtomicU32::new(seq), AtomicU32::new(seq)];
                for &n_tok in &cfg.tokens {
                    for &n_used in &cfg.experts {
                        let mut ids = vec![-1i32; n_tok * n_used];
                        let x: Vec<f32> = (0..n_tok * ne).map(|_| rng.unit()).collect();
                        let mut out = vec![0.0f32; n_tok * n_used * ne];
                        let mut eb = cfg.exact.then(|| ExactBufs {
                            xq: q8_1_blocks(&mut rng, n_tok * ne / 32),
                            aq: q8_1_blocks(&mut rng, n_tok * n_used * nf / 32),
                            gu: vec![0.0; n_tok * n_used * 2 * nf],
                            flags: std::array::from_fn(|_| AtomicU32::new(seq)),
                            thr: [AtomicU32::new(THRESHOLD_ONE)],
                            cost: [0.0],
                        });
                        let mut times = Vec::with_capacity(cfg.jobs);
                        let (mut gate_up, mut down) = (Vec::new(), Vec::new());
                        for j in 0..cfg.jobs + 8 {
                            for (i, id) in ids.iter_mut().enumerate() {
                                *id = ((j * n_used + i % n_used) % cfg.n_expert) as i32;
                            }
                            seq = seq.wrapping_add(1);
                            let exact = eb.as_mut().map(|b| ExactExchange {
                                gu_done: &b.flags[0],
                                act_ready: &b.flags[1],
                                act_arrived: &b.flags[2],
                                n_miss: &b.flags[3],
                                thr: b.thr.as_ptr(),
                                cost: b.cost.as_mut_ptr(),
                                n_layers: 1,
                                xq: b.xq.as_ptr(),
                                gu: b.gu.as_mut_ptr(),
                                aq: b.aq.as_ptr(),
                                n_ff: nf,
                                auto: false,
                                cap: THRESHOLD_ONE,
                                link_us_per_byte: 0.0,
                            });
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
                                    exact,
                                },
                            };
                            let t = Instant::now();
                            match &eb {
                                None => {
                                    // SAFETY: the buffers and `kinds` outlive the call.
                                    unsafe { run_now(&pool, queue, job) };
                                }
                                Some(b) => {
                                    pool.submit(queue, job);
                                    flags[0].store(seq, Ordering::Release);
                                    spin_until(&b.flags[0], seq);
                                    let gu = t.elapsed();
                                    let until = Instant::now() + Duration::from_micros(cfg.act_us);
                                    while Instant::now() < until {
                                        std::hint::spin_loop();
                                    }
                                    let a = Instant::now();
                                    b.flags[1].store(seq, Ordering::Release);
                                    spin_until(&flags[1], seq);
                                    if j >= 8 {
                                        gate_up.push(gu.as_secs_f64());
                                        down.push(a.elapsed().as_secs_f64());
                                    }
                                }
                            }
                            if j >= 8 {
                                times.push(t.elapsed().as_secs_f64());
                            }
                        }
                        let median = |v: &mut Vec<f64>| {
                            v.sort_by(f64::total_cmp);
                            v.get(v.len() / 2).copied()
                        };
                        let t = median(&mut times).unwrap_or(0.0);
                        let us = |v: Option<f64>| {
                            v.map_or_else(|| "-".to_string(), |v| format!("{:.1}", v * 1e6))
                        };
                        // The pool's own time: the job without the GPU's wait.
                        let busy = t - cfg.exact as u8 as f64 * cfg.act_us as f64 * 1e-6;
                        line(format!(
                            "{nc:>5} {smt:>4} {:>8} {tpt:>8} {n_tok:>7} {n_used:>8} {:>8.1} \
                             {:>10.1} {:>7.2} {:>11} {:>8}",
                            cpus.len(),
                            t * 1e6,
                            busy * 1e6 / n_used as f64,
                            mb * n_used as f64 / 1e3 / busy,
                            us(median(&mut gate_up)),
                            us(median(&mut down)),
                        ));
                    }
                }
                drop(pool);
            }
        }
    }
    pin_current_thread(&previous)?;
    Ok(())
}

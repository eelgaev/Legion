//! A long prompt's pipelined step without rounds (`--long-dataflow`).
//!
//! [`Executor::step`] runs a step's micro-chunks in rounds: in round `r` stage `s` computes
//! micro-chunk `r - s`, and the next round starts once every stage of this one is done and the
//! host has read their rows, filled the next inputs, handed the residuals over, let the expert
//! cache learn the routing and queued the first staging copies. On 4 V100s with a 123K prompt
//! the first stage is the slowest in nearly every round, and the host's work after it kept its
//! GPU idle ~20 ms a round (a profile: 714 ms over 29 rounds).
//!
//! Here each stage starts its next micro-chunk as soon as what it waits on is done, as
//! Strata-AC922's pipeline threads do, each on its own GPU (eelgaev/Strata-AC922, MIT): stage
//! `s`'s micro-chunk `m` waits for its own `m - 1` (its graph and buffers), for stage `s + 1`
//! to have taken its residual of `m - 1` (the residual may share memory with the inputs, and a
//! rebuilt graph may drop it), and for stage `s - 1`'s `m` (the residual it takes). A stage
//! thread per GPU runs the graphs for the whole step; the host thread does everything else, the
//! first stage's work first. A residual is handed over by a copy queued between the two GPUs'
//! streams, not one the host waits for (a 4-GPU profile: ~6 ms per hand-off past the first
//! socket, now and then over 100 ms).
//!
//! On CUDA a stage's residual goes into a mailbox on the next stage's GPU, [`MAILBOX`] deep,
//! as soon as its graph is done (its thread copies it), and the next stage copies it in when it
//! starts that micro-chunk: a stage then waits only for its own graph and a free mailbox, and the
//! first stage runs up to [`MAILBOX`] micro-chunks ahead of the second as Strata-AC922's stage
//! queues let it, instead of waiting each micro-chunk for the next stage to take the residual.
//!
//! Each device runs the same graphs on the same inputs in the same order as with rounds, so
//! outputs are byte-identical. Only while the expert cache is lent (or there is none): the
//! cache's work after a stage's pass then only learns the routing of that stage's layers, which
//! no other stage reads.
//!
//! With `--long-dataflow-async` ([`hand_off`]) the residuals go over without host waits, and a
//! stage before the last that reads nothing back may have one more micro-chunk queued behind
//! its running graph: the host fills its inputs, takes its residual and launches it as soon
//! as the stage's thread has queued the running one, all on the stage's stream, so the GPU
//! goes from one graph to the next without waiting for the host.

use std::sync::{Arc, mpsc};
use std::time::Instant;

use super::{
    BOUNCE_MIN_BYTES, DeferredRows, Executor, Group, GroupShape, HiddenRead, Inputs, Kind,
    ModelError, Stage, StepOutput, bounce_for, kept_in,
};
use crate::ggml::Launcher;
use crate::ggml::model_bytes::{MAX_STAGES, Phase};

mod hand_off;
#[cfg(test)]
mod tests;
mod timeline;

use hand_off::{Links, Out};
use timeline::Timeline;

/// Milliseconds from `t0` to `t` (0 if `t` is earlier).
fn ms_since(t0: Instant, t: Instant) -> f64 {
    t.saturating_duration_since(t0).as_secs_f64() * 1e3
}

/// Micro-chunks a stage's residuals wait in for the next stage (CUDA).
const MAILBOX: usize = 2;
// The async hand-off's events take turns over the mailbox entries.
const _: () = assert!(MAILBOX == hand_off::TURNS);

/// A stage's fill of a micro-chunk this slow gets a `slow flow fill` debug line with its parts
/// (#794: usually 1-30 ms, now and then 230-330 ms on the last stage).
const SLOW_FILL_MS: f64 = 50.0;

/// Pinned memory a micro-chunk's queued inputs take at most per stage: positions, cells,
/// tables and sparse-attention lists of a few thousand tokens (~1 MB), and on the first stage
/// the per-layer embedding rows (~31 MB per 4096 tokens of Qwen3.8-Flash-Next).
const FILL_PINNED_MAX: usize = 64 << 20;

/// A stage thread's job: a graph, then what follows it.
type Job = (Launcher, After);

/// What a stage thread does with its graph.
enum After {
    /// Runs it, then copies its output and waits for that, `(dst, src, bytes)` addresses: its
    /// residual into the next stage's mailbox, or the last stage's captured rows into pinned
    /// host memory (`hidden_read`). Reports it done.
    Run(Option<(usize, usize, usize)>),
    /// Queues it and its hand-off ([`hand_off`]) and reports it queued; the stage's waiter
    /// reports it done.
    Queue(Out),
}

/// What the stage threads report to the host.
enum Note {
    /// Stage `s` queued its graph (and hand-off) on its stream.
    Queued(usize),
    /// Stage `s`'s graph is done (or failed), at that time.
    Done(Done),
}

type Done = (usize, Result<(), crate::ggml::GgmlError>, Instant);

/// A step's micro-chunks and what the rounds would have used of the step.
pub(super) struct Flow<'a, 'g> {
    pub micros: &'a [Vec<Group<'g>>],
    pub shapes: &'a [Vec<GroupShape>],
    pub kind: Kind,
    pub paged: bool,
    /// The last stage's deferred cache work runs beside the first stage's first graph.
    pub overlap_mtp: bool,
}

/// Where a stage's next micro-chunk is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Prep {
    /// Nothing done yet.
    No,
    /// Graph built and inputs filled; the residual not handed over yet.
    Filled,
    /// Ready to run.
    Ready,
}

/// The step's outputs [`Executor::step_dataflow`] adds to.
pub(super) struct Sinks<'a> {
    pub deferred: &'a mut DeferredRows,
    pub hidden: &'a mut [Vec<(usize, Vec<f32>)>],
    pub out: &'a mut StepOutput,
    pub phase: &'a mut Phase,
    pub stage_phase: &'a mut [Phase; MAX_STAGES],
}

/// The row sinks of [`Sinks`] the flow's helpers write to: deferred rows, captured rows, logits.
type SinkRefs<'a> = (
    &'a mut DeferredRows,
    &'a mut [Vec<(usize, Vec<f32>)>],
    &'a mut StepOutput,
);

/// What the host may do next for stage `s` (`None`: it waits on a running graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Work {
    Fill,
    HandOff,
    Launch,
}

/// Per stage: micro-chunks launched, finished, and whose residual the next stage took.
#[derive(Debug, Clone, Default)]
struct Progress {
    pub launched: Vec<usize>,
    pub finished: Vec<usize>,
    pub taken: Vec<usize>,
    /// Mailbox depth: 0 when the next stage copies a residual straight out of the stage's
    /// graph (which then may not run again until it did).
    pub mailbox: usize,
    /// Micro-chunks whose graph (and hand-off) the stage's thread has queued
    /// (`--long-dataflow-async`; stages that hand their residual over only).
    pub queued: Vec<usize>,
    /// Per stage: it may fill, take and launch its next micro-chunk while one graph of it
    /// runs, once that one is queued (`--long-dataflow-async`).
    pub ahead: Vec<bool>,
    /// A stage takes a residual once the stage before queued its copy (`queued`), not once
    /// its graph is done: the take waits for the copy on the GPU ([`hand_off`]).
    pub early: bool,
}

impl Progress {
    pub fn new(n_stages: usize, mailbox: usize) -> Self {
        Self {
            launched: vec![0; n_stages],
            finished: vec![0; n_stages],
            taken: vec![0; n_stages],
            mailbox,
            queued: vec![0; n_stages],
            ahead: vec![false; n_stages],
            early: false,
        }
    }
}

/// The next piece of work for stage `s` at `prep`, of a step of `n_micro` micro-chunks.
fn next_work(p: &Progress, prep: Prep, s: usize, n_micro: usize) -> Option<Work> {
    let n_stages = p.launched.len();
    let m = p.launched[s];
    // Done with the step, or running (one queued graph may run ahead: `ahead`).
    let running = m - p.finished[s];
    if m >= n_micro || (running > 0 && !(p.ahead[s] && running == 1 && p.queued[s] == m)) {
        return None;
    }
    let last = s + 1 == n_stages;
    match prep {
        Prep::No => {
            let taken = last || p.mailbox > 0 || p.taken[s] >= m;
            taken.then_some(Work::Fill)
        }
        Prep::Filled => {
            let up = |u: usize| if p.early { p.queued[u] } else { p.finished[u] };
            (s == 0 || up(s - 1) > m).then_some(Work::HandOff)
        }
        // Its residual goes into the mailbox entry of micro-chunk `m - mailbox`.
        Prep::Ready => {
            (last || p.mailbox == 0 || p.taken[s] + p.mailbox > m).then_some(Work::Launch)
        }
    }
}

impl Executor {
    /// Whether a step of `n_micro` micro-chunks runs as [`Self::step_dataflow`].
    pub(super) fn dataflow_for(&self, n_micro: usize) -> bool {
        self.dataflow
            && n_micro > 1
            && self.remote.is_none()
            && self.stages.len() > 1
            && self.model.experts().is_none_or(|r| r.is_lent())
    }

    /// Runs every micro-chunk of a pipelined step through every stage (see the module docs).
    pub(super) fn step_dataflow(
        &mut self,
        f: Flow<'_, '_>,
        k: Sinks<'_>,
    ) -> Result<(), ModelError> {
        let n_stages = self.stages.len();
        let stage_cpus: Vec<Vec<u32>> = self
            .stage_cpus
            .iter()
            .map(|c| crate::hw::host_core::off_host_core(c))
            .collect();
        let mail = self.mailboxes(&f);
        // The output buffers outlive the links (dropped first, below).
        let (links, _out) = match mail.as_ref().filter(|_| self.dataflow_async) {
            Some(m) => self.links(m).unzip(),
            None => (None, None),
        };
        let links = links.map(Arc::new);
        std::thread::scope(|scope| {
            let (done_tx, done_rx) = mpsc::channel();
            let mut jobs = Vec::with_capacity(n_stages);
            for s in 0..n_stages {
                let (tx, rx) = mpsc::channel::<Job>();
                let done_tx = done_tx.clone();
                let cpus = stage_cpus.get(s).cloned().unwrap_or_default();
                // The stage's waiter: reports its queued graphs done (`After::Queue`).
                let (wait_tx, wait_rx) =
                    mpsc::channel::<(Result<(), crate::ggml::GgmlError>, Out)>();
                if links.as_ref().is_some_and(|l| l.sends(s)) {
                    let done_tx = done_tx.clone();
                    let cpus = cpus.clone();
                    scope.spawn(move || {
                        if !cpus.is_empty() {
                            let _ = crate::hw::numa::pin_current_thread(&cpus);
                        }
                        for (r, out) in wait_rx {
                            #[cfg(feature = "cuda")]
                            let r = r.and_then(|()| {
                                out.wait().map_err(crate::ggml::GgmlError::External)
                            });
                            #[cfg(not(feature = "cuda"))]
                            let _ = out;
                            if done_tx.send(Note::Done((s, r, Instant::now()))).is_err() {
                                break;
                            }
                        }
                    });
                }
                scope.spawn(move || {
                    if !cpus.is_empty() {
                        // Best effort: a failed pin leaves the inherited affinity.
                        let _ = crate::hw::numa::pin_current_thread(&cpus);
                    }
                    for (launcher, after) in rx {
                        let out = match after {
                            After::Run(out) => out,
                            After::Queue(q) => {
                                #[cfg(feature = "cuda")]
                                let r = {
                                    let stream = launcher.cuda_stream();
                                    launcher.start().and_then(|()| {
                                        let stream = stream.ok_or_else(|| {
                                            crate::ggml::GgmlError::External(
                                                "an async hand-off off CUDA".into(),
                                            )
                                        })?;
                                        // SAFETY: the residual of the graph just queued on
                                        // the stage's stream; a mailbox entry the next stage
                                        // took two micro-chunks ago (`next_work`).
                                        unsafe { q.queue(stream) }
                                            .map_err(crate::ggml::GgmlError::External)
                                    })
                                };
                                #[cfg(not(feature = "cuda"))]
                                let r = launcher.start();
                                // Queued, even on an error: the host stops at its done.
                                if done_tx.send(Note::Queued(s)).is_err()
                                    || wait_tx.send((r, q)).is_err()
                                {
                                    break;
                                }
                                continue;
                            }
                        };
                        #[allow(unused_mut)]
                        let mut r = launcher.run();
                        #[cfg(feature = "cuda")]
                        if let (Ok(()), Some((dst, src, n))) = (&r, out) {
                            // SAFETY: the graph's residual output, done, and a mailbox entry
                            // the next stage is done with (`next_work`).
                            r = unsafe { crate::residency::cuda::copy_now(dst, src, n) }
                                .map_err(crate::ggml::GgmlError::External);
                        }
                        #[cfg(not(feature = "cuda"))]
                        let _ = out;
                        if done_tx.send(Note::Done((s, r, Instant::now()))).is_err() {
                            break;
                        }
                    }
                });
                jobs.push(tx);
            }
            drop(done_tx);
            let depth = if mail.is_some() { MAILBOX } else { 0 };
            let mut p = Progress::new(n_stages, depth);
            if let Some(l) = &links {
                p.early = true;
                for s in 0..n_stages {
                    // Nothing read back after its graphs, so nothing the next graph overwrites
                    // before the host has it.
                    let parts = self.parts(s);
                    p.ahead[s] = l.sends(s) && parts.n_hidden == 0 && !parts.head;
                }
            }
            let ahead = p.ahead.clone();
            let mut running = 0usize;
            let r = self.flow_loop(
                &f,
                k,
                &jobs,
                &done_rx,
                &mut p,
                &mut running,
                mail.as_ref(),
                links.as_ref(),
                &ahead,
            );
            // On an error, the graphs still running must end before the step returns (the
            // caller may reuse their buffers).
            drop(jobs);
            while running > 0 {
                match done_rx.recv() {
                    Ok(Note::Done(_)) => running -= 1,
                    Ok(Note::Queued(_)) => {}
                    Err(_) => break,
                }
            }
            r
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn flow_loop(
        &mut self,
        f: &Flow<'_, '_>,
        k: Sinks<'_>,
        jobs: &[mpsc::Sender<Job>],
        done_rx: &mpsc::Receiver<Note>,
        p: &mut Progress,
        running: &mut usize,
        mail: Option<&Mailboxes>,
        links: Option<&Arc<Links>>,
        ahead: &[bool],
    ) -> Result<(), ModelError> {
        let Sinks {
            deferred,
            hidden,
            out,
            phase,
            stage_phase,
        } = k;
        let mut sinks = (deferred, hidden, out);
        let n_stages = self.stages.len();
        let n_micro = f.micros.len();
        let mut prep = vec![Prep::No; n_stages];
        // Per stage and micro-chunk parity: when its graph was launched (two may be in flight).
        let mut started = vec![[Instant::now(); 2]; n_stages];
        // Per stage: when its last graph was done.
        let mut gpu_done: Vec<Option<Instant>> = vec![None; n_stages];
        let t0 = Instant::now();
        // Per stage: host time between its graphs (launch to launch minus busy), for the log.
        let mut idle_ms = vec![0f64; n_stages];
        let mut last_done: Vec<Option<Instant>> = vec![None; n_stages];
        // Per stage: launch to done of its graphs, for the log.
        let mut busy_ms = vec![0f64; n_stages];
        // Per stage: a hand-off out of its memory is queued on its stream (`hand_off_queued`).
        let mut copying = vec![false; n_stages];
        // Per stage: the captured rows its running graph's thread reads back (`hidden_read`).
        let mut reads: Vec<Option<HiddenRead>> = vec![None; n_stages];
        // Each stage's pinned memory its inputs are queued through (`fill_queued`), all it may
        // take, before any graph runs: grown mid-step, it waited for the other GPUs' graphs.
        for st in &mut self.stages {
            st.flow_io.reserve(&mut st.backend, FILL_PINNED_MAX);
            if links.is_some() {
                st.flow_io_b.reserve(&mut st.backend, FILL_PINNED_MAX);
            }
        }
        // Per stage: host milliseconds in fills, hand-offs, launches and done graphs, for the log.
        let mut host_ms = vec![[0f64; 4]; n_stages];
        // Per stage: pipeline fill, drain and the host's lateness, for the log.
        let mut tl = Timeline::new(n_stages, n_micro);
        // A note `recv` waited for, handled with those that came meanwhile.
        let mut inbox: Option<Note> = None;
        loop {
            // Everything the host can do now, one piece at a time, the earliest stage's first:
            // the first stage sets the pace (it also embeds the tokens), and a later stage's
            // work comes first only where the first one waits on it. `next_work` keeps the
            // order the memory needs.
            loop {
                // Graphs queued or done meanwhile first: the first stage's may be among them.
                while let Some(note) = inbox.take().or_else(|| done_rx.try_recv().ok()) {
                    let done = match note {
                        Note::Queued(s) => {
                            p.queued[s] += 1;
                            continue;
                        }
                        Note::Done(done) => done,
                    };
                    *running -= 1;
                    let (s, gpu_at) = (done.0, done.2);
                    let m = p.finished[s];
                    // The graph ran from its launch, or from the end of the one before it.
                    let start = match gpu_done[s] {
                        Some(d) if d > started[s][m % 2] => d,
                        _ => started[s][m % 2],
                    };
                    let seen = self.flow_done(
                        f,
                        done,
                        p,
                        start,
                        reads[s].take(),
                        stage_phase,
                        &mut sinks,
                    )?;
                    gpu_done[s] = Some(gpu_at);
                    tl.finished(s, m, ms_since(t0, gpu_at));
                    last_done[s] = Some(seen);
                    busy_ms[s] += seen.saturating_duration_since(start).as_secs_f64() * 1e3;
                    host_ms[s][3] += seen.elapsed().as_secs_f64() * 1e3;
                }
                // A stage runs its next micro-chunk ahead only into a graph it keeps: building
                // one may drop the graph that runs.
                for s in 0..n_stages {
                    let m = p.launched[s];
                    p.ahead[s] = ahead[s]
                        && m < n_micro
                        && (prep[s] != Prep::No || self.stages[s].kept(&f.shapes[m]).is_some());
                }
                let Some((s, work)) =
                    (0..n_stages).find_map(|s| next_work(p, prep[s], s, n_micro).map(|w| (s, w)))
                else {
                    break;
                };
                let m = p.launched[s];
                let w0 = Instant::now();
                let wi = work as usize;
                match work {
                    Work::Fill => {
                        // The residual copied out of its memory, before the host writes it.
                        if std::mem::take(&mut copying[s]) {
                            self.stages[s].backend.synchronize();
                        }
                        tracing::trace!(
                            stage = s,
                            micro = m,
                            at_ms = t0.elapsed().as_secs_f64() * 1e3,
                            "flow fill"
                        );
                        let t_prep = Instant::now();
                        self.prepare(s, &f.shapes[m])?;
                        let prep_ms = t_prep.elapsed().as_secs_f64() * 1e3;
                        // Two pinned buffers take turns where a graph may still run.
                        let turn = links
                            .is_some()
                            .then(|| (m % 2 == 1, p.launched[s] > p.finished[s]));
                        let parts = self.fill_queued(s, &f.micros[m], &f.shapes[m], turn);
                        let total = w0.elapsed().as_secs_f64() * 1e3;
                        if total >= SLOW_FILL_MS {
                            tracing::debug!(
                                stage = s,
                                micro = m,
                                ms = total,
                                prepare_ms = prep_ms,
                                begin_ms = parts[0],
                                inputs_ms = parts[1],
                                extra_ms = parts[2],
                                "slow flow fill"
                            );
                        }
                        prep[s] = Prep::Filled;
                    }
                    Work::HandOff => {
                        tracing::trace!(
                            stage = s,
                            micro = m,
                            at_ms = t0.elapsed().as_secs_f64() * 1e3,
                            "flow hand-off"
                        );
                        if let Some(mail) = mail.filter(|_| s > 0) {
                            self.take_mail(s, m, &f.shapes[m], mail, links.map(|l| &**l))?;
                        } else if self.hand_off_queued(s, &f.shapes[m]) {
                            copying[s - 1] = true;
                        }
                        if s > 0 {
                            p.taken[s - 1] = m + 1;
                        }
                        prep[s] = Prep::Ready;
                    }
                    Work::Launch => {
                        reads[s] = self.flow_launch(
                            f,
                            s,
                            m,
                            phase,
                            stage_phase,
                            &jobs[s],
                            mail,
                            links,
                            &mut sinks,
                        )?;
                        *running += 1;
                        // Queued behind a running graph of the stage: no idle time.
                        let behind = p.launched[s] > p.finished[s];
                        p.launched[s] += 1;
                        prep[s] = Prep::No;
                        let now = Instant::now();
                        if let Some(d) = last_done[s].filter(|_| !behind) {
                            idle_ms[s] += (now - d).as_secs_f64() * 1e3;
                        }
                        started[s][m % 2] = now;
                        tl.launched(s, m, ms_since(t0, now));
                        tracing::trace!(
                            stage = s,
                            micro = m,
                            at_ms = (now - t0).as_secs_f64() * 1e3,
                            "flow launch"
                        );
                        // The first stage's next n-gram rows, while it computes.
                        if m + 1 < n_micro {
                            self.gather_ahead(&[(s, m + 1)], f.micros);
                        }
                        if s == 0 && m == 0 && f.overlap_mtp {
                            self.flush_deferred_last()?;
                        }
                    }
                }
                host_ms[s][wi] += w0.elapsed().as_secs_f64() * 1e3;
            }
            if p.finished.iter().all(|&n| n == n_micro) {
                break;
            }
            inbox = Some(
                done_rx
                    .recv()
                    .expect("a stage thread reports every graph it runs"),
            );
        }
        let end = t0.elapsed().as_secs_f64() * 1e3;
        let round = |v: &[f64]| v.iter().map(|x| x.round()).collect::<Vec<_>>();
        tracing::debug!(
            stages = n_stages,
            n_micro,
            ms = end,
            busy_ms = ?busy_ms.iter().map(|x| x.round()).collect::<Vec<_>>(),
            idle_ms = ?idle_ms.iter().map(|x| x.round()).collect::<Vec<_>>(),
            fill_ms = ?round(&tl.fill()),
            drain_ms = ?round(&tl.drain(end)),
            late_ms = ?round(tl.late()),
            fill_handoff_launch_done_ms = ?host_ms
                .iter()
                .map(|h| h.map(|x| x.round()))
                .collect::<Vec<_>>(),
            "step dataflow"
        );
        Ok(())
    }

    /// After a stage's graph is done (`done`; it ran from `start`): reads its rows and lets the
    /// cache learn its routing. Returns when the host saw it done.
    #[allow(clippy::too_many_arguments)]
    fn flow_done(
        &mut self,
        f: &Flow<'_, '_>,
        done: Done,
        p: &mut Progress,
        start: Instant,
        read: Option<HiddenRead>,
        stage_phase: &mut [Phase; MAX_STAGES],
        sinks: &mut SinkRefs<'_>,
    ) -> Result<Instant, ModelError> {
        let (s, r, at) = done;
        r?;
        let seen = Instant::now();
        let m = p.finished[s];
        let busy = at.saturating_duration_since(start).as_secs_f64();
        stage_phase[s.min(MAX_STAGES - 1)].seconds += busy;
        let n_micro = f.micros.len();
        self.stage_output(
            s,
            m,
            m + s,
            &f.micros[m],
            &f.shapes[m],
            n_micro,
            sinks.0,
            sinks.1,
            sinks.2,
            read,
        );
        // The cache learns the micro-chunk's routing before the stage's next graph.
        let n = f.shapes[m].iter().map(|g| g.tokens()).sum();
        let layers = self.stages[s].layers.clone();
        self.after_pass(n, |il| layers.contains(&il), f.kind)?;
        p.finished[s] += 1;
        tracing::trace!(
            stage = s,
            micro = m,
            busy_ms = busy * 1e3,
            "flow graph done"
        );
        Ok(seen)
    }

    /// Starts stage `s`'s graph of micro-chunk `m` on its thread (`job`).
    #[allow(clippy::too_many_arguments)]
    fn flow_launch(
        &mut self,
        f: &Flow<'_, '_>,
        s: usize,
        m: usize,
        phase: &mut Phase,
        stage_phase: &mut [Phase; MAX_STAGES],
        job: &mpsc::Sender<Job>,
        mail: Option<&Mailboxes>,
        links: Option<&Arc<Links>>,
        sinks: &mut SinkRefs<'_>,
    ) -> Result<Option<HiddenRead>, ModelError> {
        let n = f.shapes[m].iter().map(|g| g.tokens()).sum();
        let layers = self.stages[s].layers.clone();
        self.begin_pass(n, |il| layers.contains(&il), f.kind)?;
        let sp = &mut stage_phase[s.min(MAX_STAGES - 1)];
        if let Some(kept) = self.stages[s].kept(&f.shapes[m]) {
            phase.add(kept.model_bytes(), 0.0);
            sp.add_graph(kept.model_bytes());
        }
        // The KV its attention reads (#391).
        let kv = &self.stages[s].kv;
        let tier = (kv.hbm_blocks() < kv.n_blocks()).then_some(kv.hbm_blocks());
        let (dev, host) = self.kv_rates[s].step(&f.micros[m], f.paged, tier);
        phase.kv += dev;
        phase.kv_host += host;
        sp.kv += dev;
        sp.kv_host += host;
        // Where its residual goes: the next stage's mailbox entry.
        let out = match mail.filter(|_| s + 1 < self.stages.len()) {
            Some(mail) => {
                let kept = self.stages[s].kept(&f.shapes[m]).expect("prepared");
                let src = Inputs::from_parts(&f.shapes[m], self.parts(s), |i| kept.tensor(i))
                    .res_out
                    .expect("a stage before the last hands its residual on");
                let n = src.nbytes();
                if n > mail.bytes[s + 1] {
                    return Err(ModelError::Ggml(crate::ggml::GgmlError::External(format!(
                        "a residual of {n} bytes over its mailbox's {}",
                        mail.bytes[s + 1]
                    ))));
                }
                // SAFETY: reading the data pointer of a live tensor.
                let src = unsafe { (*src.raw()).data } as usize;
                Some((mail.entry(s + 1, m), src, n))
            }
            // The last stage's captured rows: its thread reads them into the round's bounce
            // buffer, off the host's loop.
            None if s + 1 == self.stages.len() => self
                .hidden_read(f, s, m, sinks)
                .map(|r| (r.dst, r.src, r.bytes)),
            None => None,
        };
        let read = out
            .filter(|_| s + 1 == self.stages.len())
            .map(|(dst, src, bytes)| HiddenRead { src, dst, bytes });
        let after = match (links.filter(|l| l.sends(s)), out) {
            (Some(l), Some((dst, src, bytes))) => After::Queue(Out {
                links: l.clone(),
                s,
                m,
                src,
                dst,
                bytes,
            }),
            _ => After::Run(out),
        };
        let stage = &mut self.stages[s];
        let kept = kept_in(&stage.decode, &stage.mixed, &f.shapes[m]).expect("prepared");
        // SAFETY: until the stage's thread reports this graph done, nothing else uses this
        // stage's backend or graph: the host fills, hands over or reads a stage only between
        // its graphs (`next_work`), and reads a residual only of a done graph. With
        // `--long-dataflow-async`, a stage that reads nothing back may get its next graph
        // once its thread has queued this one: the host then only queues work on the stage's
        // stream (inputs through the other pinned buffer, the residual's take, the graph),
        // which runs after this graph, and the graph's buffers are the ones it keeps.
        let launcher = unsafe { stage.backend.launcher(kept)? };
        job.send((launcher, after)).map_err(|_| {
            ModelError::Ggml(crate::ggml::GgmlError::External("stage thread gone".into()))
        })?;
        // The next micro-chunk's first staging copies run behind this one's last layers.
        if let (Some(next), Some(r)) = (f.shapes.get(m + 1), self.model.experts_mut()) {
            let n = next.iter().map(|g| g.tokens()).sum();
            r.stage_ahead(n, |il| layers.contains(&il)).map_err(|e| {
                ModelError::Ggml(crate::ggml::GgmlError::External(format!(
                    "expert cache: {e}"
                )))
            })?;
        }
        Ok(read)
    }

    /// Where the last stage's thread reads micro-chunk `m`'s captured rows to after its graph:
    /// the round's bounce buffer, as [`Executor::stage_output`] would (once the rows of two
    /// rounds before are out of it). `None` where `stage_output` reads them itself.
    fn hidden_read(
        &mut self,
        f: &Flow<'_, '_>,
        s: usize,
        m: usize,
        sinks: &mut SinkRefs<'_>,
    ) -> Option<HiddenRead> {
        if f.micros.len() < 2 || self.captured(s).len() != 1 {
            return None;
        }
        let parts = self.parts(s);
        let Stage {
            decode,
            mixed,
            bounce,
            bounce_b,
            backend,
            ..
        } = &mut self.stages[s];
        backend.device().discrete_cuda()?;
        let kept = kept_in(decode, mixed, &f.shapes[m])?;
        let t = *Inputs::from_parts(&f.shapes[m], parts, |i| kept.tensor(i))
            .hidden
            .first()?;
        let bytes = t.nbytes();
        if bytes < BOUNCE_MIN_BYTES {
            return None;
        }
        // The round `stage_output` is called with.
        let parity = (m + s) % 2;
        let slot = if parity == 0 { bounce } else { bounce_b };
        sinks.0.join(Some(parity), sinks.1);
        let via = bounce_for(slot, backend, bytes)?;
        // SAFETY: reading the data pointers of live tensors.
        let (src, dst) = unsafe { ((*t.raw()).data as usize, (*via.raw()).data as usize) };
        Some(HiddenRead { src, dst, bytes })
    }

    /// The step's mailboxes, if every stage is on a CUDA GPU and they fit: per stage after the
    /// first, [`MAILBOX`] residuals of the step's biggest micro-chunk in its GPU's memory.
    fn mailboxes(&mut self, f: &Flow<'_, '_>) -> Option<Mailboxes> {
        #[cfg(feature = "cuda")]
        {
            let n_stages = self.stages.len();
            if self
                .stages
                .iter()
                .any(|st| st.backend.device().discrete_cuda().is_none())
            {
                return None;
            }
            let tokens = |m: usize| f.shapes[m].iter().map(|g| g.tokens()).sum::<usize>();
            let max_tokens = (0..f.shapes.len()).map(tokens).max()?;
            let mut mail = Mailboxes {
                ctx: Vec::new(),
                addr: vec![Vec::new(); n_stages],
                bytes: vec![0; n_stages],
            };
            for s in 1..n_stages {
                // The residual of the first micro-chunk sizes it, per token.
                self.prepare(s, &f.shapes[0]).ok()?;
                let stage = &self.stages[s];
                let kept = stage.kept(&f.shapes[0])?;
                let res =
                    Inputs::from_parts(&f.shapes[0], self.parts(s), |i| kept.tensor(i)).res_in?;
                let bytes = res.nbytes().div_ceil(tokens(0)) * max_tokens;
                let ctx = crate::ggml::Context::new(MAILBOX, 0);
                for i in 0..MAILBOX {
                    ctx.new_tensor(crate::ggml::types::F32, &[bytes.div_ceil(4) as i64])
                        .named(&format!("mail{i}"));
                }
                if let Err(e) = ctx.alloc_tensors(stage.backend.buffer_type()) {
                    tracing::debug!(stage = s, bytes, error = %e, "no dataflow mailbox");
                    return None;
                }
                for i in 0..MAILBOX {
                    let t = ctx.tensor(&format!("mail{i}")).expect("made");
                    // SAFETY: reading the data pointer of a live tensor.
                    mail.addr[s].push(unsafe { (*t.raw()).data } as usize);
                }
                mail.bytes[s] = bytes;
                mail.ctx.push(ctx);
            }
            if self.dataflow_peer {
                // Neighbouring stages' GPUs; once on, a step finds them already enabled.
                let dev = |s: usize| self.stages[s].backend.device().discrete_cuda();
                let pairs: Vec<(usize, usize)> = (1..n_stages)
                    .filter_map(|s| Some((dev(s - 1)?, dev(s)?)))
                    .filter(|(a, b)| a != b)
                    .collect();
                if let Err(e) = crate::model::core::pair::link::enable_peers(&pairs) {
                    // The hand-off copies still work, through host memory.
                    tracing::debug!(error = %e, "dataflow: no peer access");
                }
            }
            Some(mail)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = f;
            None
        }
    }

    /// [`Executor::fill_inputs`] with the step's own inputs as copies queued on the stage's
    /// stream through its `flow_io` (pinned), before the graph launched next: the host does
    /// not wait for each copy behind the expert staging's. The same bytes. Returns the
    /// milliseconds of its parts, for the log: waiting for the last copies, the step's inputs,
    /// the architecture's.
    ///
    /// `turn` (`--long-dataflow-async`): whether it is the second pinned buffer's turn (odd
    /// micro-chunks), and whether a graph of the stage still runs. A buffer's last copies were
    /// queued before a graph that is done by then (at most one runs), so it does not wait for
    /// them; while one runs, inputs that cannot be queued wait for it.
    fn fill_queued(
        &mut self,
        s: usize,
        groups: &[Group<'_>],
        shape: &[GroupShape],
        turn: Option<(bool, bool)>,
    ) -> [f64; 3] {
        let parts = self.parts(s);
        let Self { model, stages, .. } = self;
        let stage = &mut stages[s];
        let Stage {
            decode,
            mixed,
            backend,
            rec,
            kv,
            spec,
            flow_io,
            flow_io_b,
            ..
        } = stage;
        let io = match turn {
            Some((true, _)) => flow_io_b,
            _ => flow_io,
        };
        if turn.is_some() {
            io.settle();
        }
        io.set_ordered(turn.is_some_and(|(_, behind)| behind));
        let kept = kept_in(decode, mixed, shape).expect("prepared");
        let inputs = Inputs::from_parts(shape, parts, |i| kept.tensor(i));
        let t0 = Instant::now();
        io.begin(backend, FILL_PINNED_MAX);
        let t1 = Instant::now();
        let mut sink = super::mtp_window::Queued { io, backend };
        inputs.fill_into(groups, shape, &mut sink);
        let t2 = Instant::now();
        model.fill_extra(&inputs, groups, shape, rec, kv, spec.as_ref(), &mut sink);
        [
            ms_since(t0, t1),
            ms_since(t1, t2),
            ms_since(t2, Instant::now()),
        ]
    }

    /// Copies stage `s`'s residual of micro-chunk `m` out of its mailbox into its graph: now,
    /// or with `links` queued on the stage's stream behind the stage before's copy into the
    /// mailbox ([`Links::take`]).
    fn take_mail(
        &self,
        s: usize,
        m: usize,
        shape: &[GroupShape],
        mail: &Mailboxes,
        links: Option<&Links>,
    ) -> Result<(), ModelError> {
        let kept = self.stages[s].kept(shape).expect("prepared");
        let dst = Inputs::from_parts(shape, self.parts(s), |i| kept.tensor(i))
            .res_in
            .expect("a stage after the first takes a residual");
        #[cfg(feature = "cuda")]
        {
            let ext = |e: String| ModelError::Ggml(crate::ggml::GgmlError::External(e));
            // SAFETY: reading the data pointer of a live tensor.
            let to = unsafe { (*dst.raw()).data } as usize;
            if let Some(l) = links {
                let stream = self.stages[s]
                    .backend
                    .cuda_stream()
                    .ok_or_else(|| ext("an async hand-off off CUDA".into()))?;
                // SAFETY: the input of the stage's graph launched next on its stream (a graph
                // of the stage running ahead of it is queued before, `next_work`), and the
                // entry stage `s - 1`'s thread queued its copy of micro-chunk `m` into.
                return unsafe { l.take(s, m, to, mail.entry(s, m), dst.nbytes(), stream) }
                    .map_err(ext);
            }
            // SAFETY: the stage's graph input (the stage is idle) and the entry stage `s - 1`'s
            // thread filled before it reported micro-chunk `m` done.
            unsafe { crate::residency::cuda::copy_now(to, mail.entry(s, m), dst.nbytes()) }
                .map_err(ext)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dst, m, mail, links);
            unreachable!("mailboxes are CUDA's")
        }
    }

    /// The step's async hand-off ([`hand_off`]) over `mail`: per stage before the last, an
    /// output buffer of a residual on its GPU and a copy stream. `None` (the plain hand-off)
    /// where they do not fit or CUDA fails. The contexts keep the output buffers.
    fn links(&mut self, mail: &Mailboxes) -> Option<(Links, Vec<crate::ggml::Context>)> {
        #[cfg(feature = "cuda")]
        {
            let n_stages = self.stages.len();
            let mut dev = Vec::with_capacity(n_stages);
            for st in &self.stages {
                dev.push(st.backend.device().discrete_cuda()?);
            }
            let mut out = Vec::with_capacity(n_stages);
            let mut ctx = Vec::with_capacity(n_stages);
            for s in 0..n_stages - 1 {
                // A residual of the step's biggest micro-chunk: the next stage's mailbox entry.
                let bytes = mail.bytes[s + 1];
                let c = crate::ggml::Context::new(1, 0);
                c.new_tensor(crate::ggml::types::F32, &[bytes.div_ceil(4) as i64])
                    .named("flow_out");
                if let Err(e) = c.alloc_tensors(self.stages[s].backend.buffer_type()) {
                    tracing::debug!(stage = s, bytes, error = %e, "no async dataflow hand-off");
                    return None;
                }
                let t = c.tensor("flow_out").expect("made");
                // SAFETY: reading the data pointer of a live tensor.
                out.push(unsafe { (*t.raw()).data } as usize);
                ctx.push(c);
            }
            match Links::new(dev, out) {
                Ok(l) => Some((l, ctx)),
                Err(e) => {
                    tracing::debug!(error = %e, "no async dataflow hand-off");
                    None
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = mail;
            None
        }
    }
}

/// Per stage after the first, [`MAILBOX`] residual buffers on its GPU.
struct Mailboxes {
    /// Keeps the buffers.
    #[allow(dead_code)]
    ctx: Vec<crate::ggml::Context>,
    addr: Vec<Vec<usize>>,
    bytes: Vec<usize>,
}

impl Mailboxes {
    /// Where stage `s`'s residual of micro-chunk `m` waits.
    fn entry(&self, s: usize, m: usize) -> usize {
        self.addr[s][m % self.addr[s].len()]
    }
}

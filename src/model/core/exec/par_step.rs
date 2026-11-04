//! GLM-5.3-Flash's parallel steps (`--glm-parallel-decode`, #558): every GPU runs a graph over
//! all the layers at once ([`Model::glm_par_stage`](crate::model::core::Model)), its share of
//! the work (its heads of every attention layer, its rows of every dense and shared FFN, its
//! experts), one all-reduce per sublayer in between. Every GPU keeps its heads' KV and states
//! of every layer ([`Executor::new_glm_par`]), so every step runs this way: prompt chunks and
//! decode steps and verify steps of up to [`MAX_TOKENS`] tokens (a prompt borrowing the expert
//! cache up to [`GlmPar::max_tokens`](crate::model::glm5next::par::GlmPar)). A verify step keeps every
//! GPU's replay inputs of its heads ([`Executor::enable_spec`]); the commit replays each GPU's
//! heads with its own shards of the convolutions. The MTP head drafts on the last GPU, from
//! the head input its graph outputs.

pub(super) mod pool;

use super::{Executor, Kind, MAX_STAGES, kept_in, mtp_window};
use crate::ggml::model_bytes::DECODE_MAX_TOKENS;
use crate::model::core::{Group, GroupShape, Inputs, ModelError, StepOutput, batch_shape, with_kv};
use crate::model::glm5next::par::MAX_TOKENS;

/// Steps between two reports of the parallel decode's exchange waits.
const REPORT_EVERY: u64 = 512;

/// Why a parallel step of `n` tokens cannot run (none when it can): over `max` (the step's
/// limit, [`GlmPar::max_tokens`]), verify rows without replay buffers (`unreplayable`), or
/// captures of `hidden` other than the last layer's head input (of a model of `n_layer`
/// layers), which the MTP head reads.
///
/// [`GlmPar::max_tokens`]: crate::model::glm5next::par::GlmPar
pub(super) fn par_refusal(
    n: usize,
    max: usize,
    unreplayable: bool,
    hidden: &[usize],
    n_layer: usize,
) -> Option<String> {
    if n > max {
        Some(format!("a parallel step of {n} tokens (at most {max})"))
    } else if unreplayable {
        Some("a parallel verify step without replay buffers (Executor::enable_spec)".into())
    } else if !matches!(hidden, [] | [_]) || hidden.iter().any(|&il| il + 1 != n_layer) {
        Some(format!(
            "GLM's parallel steps capture the last layer's head input only, not {hidden:?}"
        ))
    } else {
        None
    }
}

/// The exchange's waits `matrix` (`[reader][writer]`, ns) summed: per GPU the time it waited for
/// the others (its row), and the time the others waited for it (its column, its own wait
/// left out).
pub(super) fn wait_sums(matrix: &[Vec<u64>]) -> (Vec<u64>, Vec<u64>) {
    let waits = matrix.iter().map(|r| r.iter().sum()).collect();
    let late = (0..matrix.len())
        .map(|w| {
            matrix
                .iter()
                .enumerate()
                .filter(|&(d, _)| d != w)
                .map(|(_, r)| r.get(w).copied().unwrap_or(0))
                .sum()
        })
        .collect();
    (waits, late)
}

/// Where the host's time between two parallel steps goes (#676), summed in ns since the last
/// report: the gap is from one step's graphs done to the next step's launches, the GPUs idle
/// but for the commit and the MTP passes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) enum GapPhase {
    /// Deferred expert-cache work run before the step's launches.
    #[default]
    Deferred,
    /// The step's graphs found or built and made ready.
    Prepare,
    /// The step's inputs written.
    Fill,
    /// The expert cache's pass start, the launchers.
    Launch,
    /// The last GPU's output rows (logits or argmax, the MTP head's input) read.
    Output,
    /// The expert cache's work after the step (routing learned, swaps, tables).
    After,
    /// The verify commit's launches.
    Commit,
    /// The MTP head's passes (on the last GPU; on every GPU with `--glm-parallel-mtp`).
    Mtp,
    /// The step's end: the commit waited for, recurrent states moved between slots.
    States,
    /// The rest: the engine's sampling, streaming, planning and packing.
    Engine,
}

const GAP_PHASES: usize = 10;

impl GapPhase {
    const ALL: [GapPhase; GAP_PHASES] = [
        GapPhase::Deferred,
        GapPhase::Prepare,
        GapPhase::Fill,
        GapPhase::Launch,
        GapPhase::Output,
        GapPhase::After,
        GapPhase::Commit,
        GapPhase::Mtp,
        GapPhase::States,
        GapPhase::Engine,
    ];
}

/// The gap between parallel steps by [`GapPhase`] ([`ParStats::gap`]). Only gaps between two
/// decode steps count: a prompt chunk's gap is the prompt's planning and sampling, not the
/// decode's.
#[derive(Debug, Default)]
pub(super) struct GapClock {
    /// When the last decode step's graphs were done (none before the first step, or after a
    /// prompt step).
    done: Option<std::time::Instant>,
    /// Since the last report: ns per phase, and the gaps' whole ns over `gaps` gaps.
    ns: [u64; GAP_PHASES],
    total: u64,
    gaps: u64,
    /// The ns per phase measured inside the current gap, so far.
    cur: [u64; GAP_PHASES],
    /// Since the last report: the steps' graphs built (not kept), over every GPU and step.
    pub builds: u64,
    /// Since the last report: the verify commits' graphs built (not kept), over every GPU.
    pub commit_builds: u64,
}

impl GapClock {
    /// Adds `d` to `phase` of the current gap.
    pub(super) fn add(&mut self, phase: GapPhase, d: std::time::Duration) {
        self.cur[phase as usize] += d.as_nanos() as u64;
    }

    /// Adds the time from `*at` to now to `phase` and moves `*at` to now.
    pub(super) fn lap(&mut self, phase: GapPhase, at: &mut std::time::Instant) {
        let now = std::time::Instant::now();
        self.add(phase, now.saturating_duration_since(*at));
        *at = now;
    }

    /// The next step's launches start at `at`: closes the gap since the last step's graphs
    /// were done, the time not measured in a phase going to [`GapPhase::Engine`]. The gap
    /// counts if both steps decode (`decode`: this one does).
    pub(super) fn launch(&mut self, at: std::time::Instant, decode: bool) {
        let cur = std::mem::take(&mut self.cur);
        let Some(done) = self.done.take().filter(|_| decode) else {
            return;
        };
        let gap = at.saturating_duration_since(done).as_nanos() as u64;
        self.total += gap;
        self.gaps += 1;
        for (s, c) in self.ns.iter_mut().zip(cur) {
            *s += c;
        }
        let inside: u64 = cur.iter().sum();
        self.ns[GapPhase::Engine as usize] += gap.saturating_sub(inside);
    }

    /// The step's graphs are done at `at` (`decode`: a decode step's, whose gap to the next
    /// step may count).
    pub(super) fn done(&mut self, at: std::time::Instant, decode: bool) {
        self.done = decode.then_some(at);
        self.cur = [0; GAP_PHASES];
    }

    /// The gaps counted, the ms per gap: the whole gap, then per [`GapPhase`]; resets the sums.
    pub(super) fn take_ms(&mut self) -> (u64, f64, [(GapPhase, f64); GAP_PHASES]) {
        let gaps = self.gaps;
        let n = self.gaps.max(1) as f64;
        let ms = |ns: u64| (ns as f64 / 1e6 / n * 100.0).round() / 100.0;
        let total = ms(self.total);
        let phases = GapPhase::ALL.map(|p| (p, ms(self.ns[p as usize])));
        self.ns = [0; GAP_PHASES];
        self.total = 0;
        self.gaps = 0;
        (gaps, total, phases)
    }
}

/// The parallel decode's running counters.
#[derive(Debug, Default)]
pub(super) struct ParStats {
    /// Every step runs parallel ([`Executor::new_glm_par`]): the stages are the GPUs' shares,
    /// not a pipeline.
    pub every_step: bool,
    /// `--glm-mtp-overlap-seqs` (#723, #725, [`Executor::set_glm_mtp_overlap`]): the most
    /// sequences of a step the overlap applies to (0: never).
    pub mtp_overlap_seqs: usize,
    /// Whether the overlap applies to the current step, its commit and the MTP pass after it
    /// (set by [`Executor::step_par`]).
    pub overlap_step: bool,
    /// Steps run, ever: the exchanges' values start from it ([`Exchange::begin_step`]).
    ///
    /// [`Exchange::begin_step`]: crate::model::glm5next::par::exchange::Exchange::begin_step
    pub step: u64,
    /// Since the last report: steps, tokens, seconds, per GPU the ns it waited for the others.
    pub steps: u64,
    pub tokens: u64,
    pub seconds: f64,
    pub waits: Vec<u64>,
    /// Since the last report, per GPU: the ns the others waited for it (how late it was).
    pub late: Vec<u64>,
    /// Per GPU: its expert cache's host reads (misses) at the last report, in bytes.
    pub host_seen: Vec<u64>,
    /// Per GPU: its expert cache's VRAM reads (hits) at the last report, in bytes.
    pub device_seen: Vec<u64>,
    /// Per GPU: its expert cache's layer passes with a host read at the last report.
    pub miss_passes_seen: Vec<u64>,
    /// The GPUs' launch threads, from the first step on.
    pub pool: Option<pool::LaunchPool>,
    /// The host's time between steps, by phase.
    pub gap: GapClock,
    /// The verify commit's launches held for the MTP pass ([`Executor::commit_par`]).
    pub held: HeldCommits,
    /// Since the last report: MTP passes on every GPU (`--glm-parallel-mtp`), and per GPU the
    /// ns the others waited for it in them.
    pub mtp_passes: u64,
    pub mtp_late: Vec<u64>,
}

/// The verify commit's graphs of every GPU but the last (every GPU under
/// `--glm-mtp-overlap`, #723), made ready but not started ([`Executor::commit_par`], #714):
/// their launch threads start them once the MTP pass is queued on the last GPU
/// ([`ParStats::start_held`]), so their ~1 ms of launches (the driver takes launches on all
/// four GPUs about one at a time) run beside the head's compute instead of before it. Each
/// GPU's stream keeps its order (the step, its commit, the next step).
#[derive(Default)]
pub(super) struct HeldCommits {
    /// Per GPU: (GPU, its launch thread's CPUs, the commit's launch).
    jobs: Vec<(usize, Vec<u32>, pool::Launch)>,
    /// A started launch's error, for the next call that can return it.
    error: Option<crate::ggml::GgmlError>,
}

impl std::fmt::Debug for HeldCommits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpus: Vec<usize> = self.jobs.iter().map(|j| j.0).collect();
        f.debug_struct("HeldCommits")
            .field("gpus", &gpus)
            .field("error", &self.error)
            .finish()
    }
}

impl ParStats {
    /// Starts the held commit launches on their threads ([`HeldCommits`]); returns at once.
    pub(super) fn start_held(&mut self) {
        let jobs = std::mem::take(&mut self.held.jobs);
        if jobs.is_empty() {
            return;
        }
        let pool = self
            .pool
            .as_ref()
            .expect("held commits without launch threads");
        for (d, cpus, launch) in jobs {
            pool.start(d, (cpus, launch));
        }
    }

    /// The held commit launches, per GPU (none for a GPU without one), for a caller that
    /// starts them itself: an MTP pass on every GPU queues each GPU's after its commit
    /// ([`super::par_mtp`]).
    pub(super) fn take_held(&mut self, n_dev: usize) -> Vec<Option<pool::Launch>> {
        let mut out: Vec<_> = (0..n_dev).map(|_| None).collect();
        for (d, _, launch) in std::mem::take(&mut self.held.jobs) {
            out[d] = Some(launch);
        }
        out
    }

    /// Starts any held commit launches and waits until every started one has queued its
    /// graph; a failed launch's error is kept for [`Self::launch_result`].
    pub(super) fn join_held(&mut self) {
        self.start_held();
        let Some(pool) = &self.pool else {
            return;
        };
        for r in pool.join() {
            if let Err(e) = r {
                self.held.error.get_or_insert(e);
            }
        }
    }

    /// The error of a commit launch joined since the last call ([`Self::join_held`]).
    pub(super) fn launch_result(&mut self) -> Result<(), ModelError> {
        match self.held.error.take() {
            Some(e) => Err(e.into()),
            None => Ok(()),
        }
    }
}

impl Executor {
    /// Whether `groups` run as a parallel step: the executor's stages are the GPUs' shares
    /// ([`Executor::new_glm_par`]) and the step has tokens to run.
    pub(super) fn par_fits(&self, groups: &[Group<'_>]) -> bool {
        self.par.every_step
            && groups
                .iter()
                .any(|g| g.seqs.iter().any(|s| !s.tokens.is_empty()))
    }

    /// Why a parallel step cannot run `groups` (none when it can): more tokens than the
    /// exchange's slots hold, verify rows without replay buffers, captured layers other than
    /// the MTP head's.
    pub(super) fn par_refuses(&self, groups: &[Group<'_>]) -> Option<String> {
        let n: usize = groups
            .iter()
            .flat_map(|g| &g.seqs)
            .map(|s| s.tokens.len())
            .sum();
        let verify = groups.iter().any(|g| g.verify);
        let max = self
            .model
            .glm_par
            .as_ref()
            .map_or(MAX_TOKENS, |p| p.max_tokens);
        par_refusal(
            n,
            max,
            verify && self.stages.iter().any(|s| s.spec.is_none()),
            &self.hidden,
            self.model.hp.n_layer,
        )
    }

    /// Runs `groups` as a parallel step ([`Self::par_fits`]).
    pub(super) fn step_par(&mut self, groups: &[Group<'_>]) -> Result<StepOutput, ModelError> {
        if let Some(why) = self.par_refuses(groups) {
            return Err(ModelError::Unsupported(why));
        }
        // A commit held for an MTP pass that did not come (no drafts): before this step.
        self.par.join_held();
        self.par.launch_result()?;
        // This step's commit and the MTP pass after it take the #723 order if the step is
        // small enough (#725).
        let seqs = groups.iter().map(|g| g.seqs.len()).sum();
        self.par.overlap_step = super::par_mtp::overlap_applies(self.par.mtp_overlap_seqs, seqs);
        let t0 = std::time::Instant::now();
        let mut lap = t0;
        self.flush_deferred()?;
        self.par.gap.lap(GapPhase::Deferred, &mut lap);
        let n_stages = self.stages.len();
        let table = self
            .model
            .paged_for(&self.stages[0].kv, groups)
            .then_some(self.max_seq_blocks);
        let arch = &*self.model.hp.arch;
        let mut shape = with_kv(
            batch_shape(groups, table, Some(arch)),
            groups,
            &self.stages[0].kv,
        );
        for g in &mut shape {
            g.par = true;
        }
        let n: usize = shape.iter().map(GroupShape::tokens).sum();
        for s in 0..n_stages {
            if self.stages[s].kept(&shape).is_none() {
                self.par.gap.builds += 1;
            }
            self.prepare(s, &shape)?;
            self.par.gap.lap(GapPhase::Prepare, &mut lap);
            self.fill_inputs_queued(s, groups, &shape);
            self.par.gap.lap(GapPhase::Fill, &mut lap);
        }
        // Every entry of the expert cache but the MTP head's ran (the copies included; not the
        // head's copies of `--glm-parallel-mtp`).
        let ran = self.target_entries();
        let prompt = groups
            .iter()
            .any(|g| g.seqs.iter().any(|s| s.tokens.len() > DECODE_MAX_TOKENS));
        let kind = Kind::of_step(prompt);
        self.begin_pass(n, &ran, kind)?;
        let par = self.model.glm_par.as_ref().expect("par_fits");
        let ex = par.ex.clone();
        self.par.step += 1;
        ex.begin_step(self.par.step);
        let stage_cpus = self.par_cpus();
        if self.par.pool.as_ref().is_none_or(|p| p.len() != n_stages) {
            self.par.pool = Some(pool::LaunchPool::new(n_stages).map_err(|e| {
                ModelError::Unsupported(format!("parallel decode: launch threads: {e}"))
            })?);
        }
        let mut jobs = Vec::with_capacity(n_stages);
        for (s, st) in self.stages.iter_mut().enumerate() {
            let kept = kept_in(&st.decode, &st.mixed, &shape).expect("prepared");
            // SAFETY: nothing else uses a stage's backend or graph until the pool below has
            // run every launch.
            let launcher = unsafe { st.backend.launcher(kept)? };
            let cpus = stage_cpus.get(s).cloned().unwrap_or_default();
            jobs.push((cpus, Box::new(move || launcher.run()) as pool::Launch));
        }
        self.par.gap.lap(GapPhase::Launch, &mut lap);
        self.par.gap.launch(lap, !prompt);
        // One host thread per GPU: every graph must be running for its exchanges to complete.
        let results = self.par.pool.as_ref().expect("made above").run(jobs);
        lap = std::time::Instant::now();
        self.par.gap.done(lap, !prompt);
        self.finish_fills();
        for r in results {
            r?;
        }
        if ex.failed() {
            return Err(ModelError::Unsupported(
                "parallel decode: a GPU waited over 10 s for another's exchange".into(),
            ));
        }
        let mut out = StepOutput::default();
        let mut deferred = super::DeferredRows::new(vec![n]);
        let mut hidden = vec![Vec::new()];
        let last = n_stages - 1;
        self.stage_output(
            last,
            0,
            0,
            groups,
            &shape,
            1,
            &mut deferred,
            &mut hidden,
            &mut out,
            None,
        );
        // The MTP head's input rows, from the last GPU's graph.
        out.hidden = self
            .hidden
            .iter()
            .map(|&il| super::rows_out::take_layer(&mut hidden, il))
            .collect();
        self.par.gap.lap(GapPhase::Output, &mut lap);
        if self.after_overlap && self.mtp_enabled {
            // Run beside the next MTP pass, which reads the MTP head's entry only (or before
            // the next step), as on the pipeline (`Executor::step_here`).
            self.deferred_target = Some((n, kind));
        } else {
            self.after_pass(n, &ran, kind)?;
        }
        self.par.gap.lap(GapPhase::After, &mut lap);
        let secs = t0.elapsed().as_secs_f64();
        let mut phase = crate::ggml::model_bytes::Phase::default();
        for s in 0..n_stages.min(MAX_STAGES) {
            if let Some(kept) = self.stages[s].kept(&shape) {
                phase.add(kept.model_bytes(), 0.0);
            }
        }
        phase.seconds = secs;
        phase.steps = 1;
        self.bytes.phase_mut(kind).merge(&phase);
        self.par_report(n, secs, ex.take_wait_matrix());
        Ok(out)
    }

    /// [`Executor::commit`] under parallel steps: every GPU replays its own heads. The last
    /// GPU's commit graph is started from this thread now, ahead of the MTP pass on its
    /// stream; the other GPUs' are held ([`HeldCommits`]) and started by their launch threads
    /// once the pass is queued, all at once (#676, #714). Before #714 all four started here
    /// at once, and the pass waited for all their launches (1.8 ms) where it needs the last
    /// GPU's (0.7 ms). Returns once the last GPU's graph is queued; [`ParStats::join_held`]
    /// (the MTP pass, the next step, [`Executor::wait_commit`]) waits for the others'.
    ///
    /// With `--glm-mtp-overlap` (and the MTP pass on every GPU) the last GPU's is held too, and
    /// the pass queues every GPU's commit after its own graph (#723): the head reads no state
    /// a commit writes, and the last GPU's commit (its launches take 1.4 ms at C=1, 4 ms at
    /// C=4) kept the head's front, which every GPU waits for, from starting.
    pub(super) fn commit_par(&mut self, commits: &[(usize, usize)]) -> Result<(), ModelError> {
        self.par.join_held();
        self.par.launch_result()?;
        let stage_cpus = self.par_cpus();
        let hold_last = self.mtp_overlap();
        let model = &self.model;
        let last = self.stages.len() - 1;
        let mut held = Vec::with_capacity(last + 1);
        let mut res = Ok(());
        // The last GPU's first: its launches are the MTP pass's critical path.
        for s in std::iter::once(last).chain(0..last) {
            let st = &mut self.stages[s];
            let spec = st
                .spec
                .as_ref()
                .expect("commit without Executor::enable_spec");
            if !spec.is_kept(&st.commits, commits) {
                self.par.gap.commit_builds += 1;
            }
            if s == last && !hold_last {
                // SAFETY: as in `commit`.
                res = unsafe {
                    spec.commit(&mut st.backend, &mut st.commits, &st.rec, commits, model)
                };
                if res.is_err() {
                    break;
                }
                continue;
            }
            // SAFETY: the stage's kept commit graphs are dropped before its pools and the
            // model's weights (as in `commit`); nothing computes or synchronizes on the
            // stage's backend until its launch thread has queued the graph (`join_held`, which
            // `wait_commit`, the next step and a pass on the last GPU alone call first; the
            // pass on every GPU readies its own graphs and queues their inputs, then queues
            // each commit after its GPU's graph).
            match unsafe {
                spec.commit_launcher(&mut st.backend, &mut st.commits, &st.rec, commits, model)
            } {
                Ok(Some(l)) => {
                    let cpus = stage_cpus.get(s).cloned().unwrap_or_default();
                    held.push((s, cpus, Box::new(move || l.start()) as pool::Launch));
                }
                // Built and started already.
                Ok(None) => {}
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        // The launchers made so far are started either way: each holds its stage's graph
        // as computed.
        self.par.held.jobs = held;
        if res.is_err() {
            self.wait_commit();
        }
        res.map_err(Into::into)
    }

    /// [`Executor::fill_inputs`] as copies queued on stage `s`'s stream through its pinned
    /// buffer ([`mtp_window::Queued`]), not one synchronous copy per input (about 20 µs each,
    /// a dozen inputs per GPU, #676): the same bytes, ahead of the graph on the same stream.
    /// [`Self::finish_fills`] after the step (a decode or verify step over the stages,
    /// `step_fill`: [`Self::sync_stage`] after each stage).
    pub(super) fn fill_inputs_queued(
        &mut self,
        s: usize,
        groups: &[Group<'_>],
        shape: &[GroupShape],
    ) {
        let parts = self.parts(s);
        let model = &self.model;
        let st = &mut self.stages[s];
        let kept = kept_in(&st.decode, &st.mixed, shape).expect("prepared");
        let inputs = Inputs::from_parts(shape, parts, |i| kept.tensor(i));
        // Held and copied by one kernel launch at the end (`pinned_io::pack`, #814).
        st.mtp_io
            .begin_packed(&mut st.backend, mtp_window::PINNED_MAX);
        let mut sink = mtp_window::Queued {
            io: &mut st.mtp_io,
            backend: &mut st.backend,
        };
        inputs.fill_into(groups, shape, &mut sink);
        model.fill_extra(
            &inputs,
            groups,
            shape,
            &st.rec,
            &st.kv,
            st.spec.as_ref(),
            &mut sink,
        );
        st.mtp_io.flush(&mut st.backend);
    }

    /// Waits for every stage's queued input copies ([`Self::fill_inputs_queued`]); after the
    /// step's graphs ran (and synchronized), an idle stream's synchronize.
    fn finish_fills(&mut self) {
        for st in &mut self.stages {
            st.mtp_io.finish(&mut st.backend);
        }
    }

    /// Per GPU: the CPUs its launch thread runs on (none: the thread's own).
    pub(super) fn par_cpus(&self) -> Vec<Vec<u32>> {
        self.stage_cpus
            .iter()
            .map(|c| crate::hw::host_core::off_host_core(c))
            .collect()
    }

    /// Counts a parallel step of `n` tokens that took `secs`, the GPUs having waited `waits`
    /// (ns, `[reader][writer]`, [`wait_sums`]) for the others; every [`REPORT_EVERY`] steps says
    /// how long they waited, and for whom.
    fn par_report(&mut self, n: usize, secs: f64, matrix: Vec<Vec<u64>>) {
        let p = &mut self.par;
        p.steps += 1;
        p.tokens += n as u64;
        p.seconds += secs;
        let (waits, late) = wait_sums(&matrix);
        if p.waits.len() != waits.len() {
            p.waits = vec![0; waits.len()];
            p.late = vec![0; waits.len()];
        }
        for (a, w) in p.waits.iter_mut().zip(waits) {
            *a += w;
        }
        for (a, w) in p.late.iter_mut().zip(late) {
            *a += w;
        }
        if p.steps < REPORT_EVERY {
            return;
        }
        // Per GPU: its experts' host reads (each from its own node's memory), per token and
        // over the steps' time.
        let n_dev = p.waits.len();
        let traffic: Vec<_> = (0..n_dev)
            .map(|d| {
                self.model
                    .experts()
                    .map(|r| r.traffic_on(d))
                    .unwrap_or_default()
            })
            .collect();
        if p.host_seen.len() != n_dev {
            p.host_seen = vec![0; n_dev];
            p.device_seen = vec![0; n_dev];
            p.miss_passes_seen = vec![0; n_dev];
        }
        let since = |now: &mut Vec<u64>, seen: &mut Vec<u64>| -> Vec<u64> {
            let d = now
                .iter()
                .zip(&*seen)
                .map(|(a, b)| a.saturating_sub(*b))
                .collect();
            std::mem::swap(now, seen);
            d
        };
        let miss = since(
            &mut traffic.iter().map(|t| t.host).collect(),
            &mut p.host_seen,
        );
        // Its experts read from VRAM: with the misses, how much routed work it had.
        let hit = since(
            &mut traffic.iter().map(|t| t.device).collect(),
            &mut p.device_seen,
        );
        // Its layer passes that read an expert from host memory: the same miss bytes in fewer
        // matvecs read the link better (#677).
        let miss_passes = since(
            &mut traffic.iter().map(|t| t.host_passes).collect(),
            &mut p.miss_passes_seen,
        );
        let round2 = |x: f64| (x * 100.0).round() / 100.0;
        let ms_per_step = |ns: u64| round2(ns as f64 / 1e6 / p.steps as f64);
        tracing::info!(
            steps = p.steps,
            tokens_per_step = round2(p.tokens as f64 / p.steps as f64),
            step_ms = round2(p.seconds * 1e3 / p.steps as f64),
            exchange_wait_ms = ?p.waits.iter().map(|&w| ms_per_step(w)).collect::<Vec<_>>(),
            waited_for_ms = ?p.late.iter().map(|&w| ms_per_step(w)).collect::<Vec<_>>(),
            miss_mib_per_token = ?miss.iter().map(|&b| round2(b as f64 / (1u64 << 20) as f64 / p.tokens as f64)).collect::<Vec<_>>(),
            miss_gib_per_s = ?miss.iter().map(|&b| round2(b as f64 / (1u64 << 30) as f64 / p.seconds)).collect::<Vec<_>>(),
            miss_layers_per_step = ?miss_passes.iter().map(|&n| round2(n as f64 / p.steps as f64)).collect::<Vec<_>>(),
            hit_mib_per_token = ?hit.iter().map(|&b| round2(b as f64 / (1u64 << 20) as f64 / p.tokens as f64)).collect::<Vec<_>>(),
            "parallel steps (per GPU)"
        );
        let (gaps, gap_ms, phases) = p.gap.take_ms();
        let phase = |ph: GapPhase| phases[ph as usize].1;
        tracing::info!(
            gaps,
            gap_ms,
            deferred = phase(GapPhase::Deferred),
            prepare = phase(GapPhase::Prepare),
            fill = phase(GapPhase::Fill),
            launch = phase(GapPhase::Launch),
            output = phase(GapPhase::Output),
            after = phase(GapPhase::After),
            commit = phase(GapPhase::Commit),
            mtp = phase(GapPhase::Mtp),
            states = phase(GapPhase::States),
            engine = phase(GapPhase::Engine),
            graphs_built = p.gap.builds,
            commit_graphs_built = p.gap.commit_builds,
            "parallel steps: host gap between steps (ms per step)"
        );
        if p.mtp_passes > 0 {
            tracing::info!(
                passes_per_step = round2(p.mtp_passes as f64 / p.steps as f64),
                waited_for_ms = ?p.mtp_late.iter().map(|&w| ms_per_step(w)).collect::<Vec<_>>(),
                "parallel MTP passes (per GPU, ms per step)"
            );
            p.mtp_passes = 0;
            p.mtp_late.iter_mut().for_each(|w| *w = 0);
        }
        p.gap.builds = 0;
        p.gap.commit_builds = 0;
        p.steps = 0;
        p.tokens = 0;
        p.seconds = 0.0;
        p.waits.iter_mut().for_each(|w| *w = 0);
        p.late.iter_mut().for_each(|w| *w = 0);
    }
}

#[cfg(test)]
mod tests;

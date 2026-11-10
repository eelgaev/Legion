//! Several sequence groups' decode (and verify) steps in flight over the pipeline stages
//! (#415). A whole step ([`Executor::step`]) crosses the stages one after another, so with
//! `n` stages each device is busy about `1/n` of the time. Here the engine splits its decoding
//! sequences into groups, and every call to [`Executor::pipe_round`] is one round: each step in
//! flight moves on to the next stage, a group's next step may enter stage 0, and all stages
//! with a piece run at once on their own devices. The step leaving the last stage returns its
//! logits; the engine samples it and sends the group's next step in the next round.
//!
//! A group's step is the same computation as a whole step of its sequences: the same graph for
//! its shape, the same inputs. The stages keep these graphs apart from the whole steps' ones,
//! all of a stage's in one compute buffer, since a stage runs one piece at a time: a piece's
//! inputs are written just before it runs, and the residual a piece hands on is taken by the
//! next stage before the stage that made it runs its next piece (later stages first).
//!
//! A round can be split ([`Executor::pipe_round_begin`], [`Executor::pipe_round_end`], #762):
//! with the MTP head, launch threads, local stages only and the expert cache's after-pass deferred
//! (on unless `--no-after-overlap`), the step leaving the last stage is handed out as soon as
//! that stage is done, so the engine samples, commits and drafts for it (the MTP head runs on
//! the last stage's device) while the other stages still compute. A commit for a stage still
//! computing is held and started when its piece is done; anything else that needs the stages
//! settles the round first ([`Executor::wait_commit`]).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::par_step::pool::{Launch as PoolLaunch, LaunchPool};
use super::{Executor, Kept, Stage, bounce_for, read_rows};
use crate::ggml::model_bytes::{Kind, Phase};
use crate::ggml::{ComputeAllocator, Context, GgmlError, KeptGraph, KeptParts};
use crate::model::core::{
    Group, GroupShape, Inputs, Logits, ModelError, SeqInput, StepOpts, StepOutput, batch_shape,
    graph_size, with_kv,
};

/// Default graphs a stage keeps for pieces ([`Executor::set_pipe_graphs`]).
pub(super) const PIPE_KEPT: usize = 4;

/// One sequence's part of a piece, owned: the caller's borrow ends when the step enters.
struct OwnedSeq {
    tokens: Vec<u32>,
    n_past: usize,
    blocks: Vec<u32>,
    ring: usize,
    logits: Logits,
}

/// A [`Group`] owned.
struct OwnedGroup {
    slot: usize,
    seqs: Vec<OwnedSeq>,
    verify: bool,
}

fn own(groups: &[Group<'_>]) -> Vec<OwnedGroup> {
    groups
        .iter()
        .map(|g| OwnedGroup {
            slot: g.slot,
            seqs: g
                .seqs
                .iter()
                .map(|s| OwnedSeq {
                    tokens: s.tokens.to_vec(),
                    n_past: s.n_past,
                    blocks: s.blocks.to_vec(),
                    ring: s.ring,
                    logits: s.logits,
                })
                .collect(),
            verify: g.verify,
        })
        .collect()
}

fn view(groups: &[OwnedGroup]) -> Vec<Group<'_>> {
    groups
        .iter()
        .map(|g| Group {
            slot: g.slot,
            seqs: g
                .seqs
                .iter()
                .map(|s| SeqInput {
                    tokens: &s.tokens,
                    n_past: s.n_past,
                    blocks: &s.blocks,
                    ring: s.ring,
                    logits: s.logits,
                    media: None,
                })
                .collect(),
            verify: g.verify,
        })
        .collect()
}

/// One group's step in flight.
pub(super) struct Piece {
    /// The caller's name for it, returned with its output.
    tag: usize,
    groups: Vec<OwnedGroup>,
    shape: Vec<GroupShape>,
    /// The stage it runs on in the current round.
    stage: usize,
    /// The captured layers' rows so far, in stage order.
    hidden: Vec<(usize, Vec<f32>)>,
    /// Whether its graphs read paged KV (for its KV bytes, #391).
    paged: bool,
    /// Rank 0 of a split model: the residual the last stage handed out, for rank 1 (which
    /// is the stage after the last, [`super::remote`]).
    rows: Vec<f32>,
}

/// A round between [`Executor::pipe_round_begin`] and [`Executor::pipe_round_end`].
pub(super) struct PipeRun {
    t0: Instant,
    /// Every piece started, from `t0`.
    t_go: Duration,
    /// The pieces run on the launch threads (else the one local piece ran on this thread).
    pooled: bool,
    /// Per local piece: its stage and tokens.
    passes: Vec<(usize, usize)>,
    /// Without launch threads: the stages whose output is still to be taken, in order.
    order: VecDeque<usize>,
    /// When each local stage's piece finished, set by its launch thread.
    done: Arc<Mutex<Vec<Option<Instant>>>>,
    /// When each local stage's piece started, from `t0`.
    started: Vec<Option<Duration>>,
    /// When each stage's piece finished, from `t0`.
    stage_done: Vec<Duration>,
    /// Per local stage: its piece is still on its launch thread.
    running: Vec<bool>,
    /// Per local stage: commits held until its piece is done ([`Executor::commit`]).
    held: Vec<Vec<Vec<(usize, usize)>>>,
    /// A step left the last stage.
    exited: bool,
    /// When it was handed out with other stages still computing, from `t0`.
    exit_at: Option<Duration>,
}

impl PipeRun {
    /// Holds `commits` for stage `s` if its piece still computes (true), else leaves them
    /// to the caller (false).
    pub(super) fn hold_commit(&mut self, s: usize, commits: &[(usize, usize)]) -> bool {
        if !self.running.get(s).copied().unwrap_or(false) {
            return false;
        }
        self.held[s].push(commits.to_vec());
        true
    }
}

fn piped_in<'k>(piped: &'k [Kept], shape: &[GroupShape]) -> Option<&'k KeptGraph> {
    piped.iter().find(|(s, _)| s == shape).map(|(_, g)| g)
}

impl Executor {
    /// Whether decode and verify steps can run as groups in flight over the stages
    /// ([`Self::pipe_round`]): several stages, none of them tensor parallel, and the expert
    /// cache not lent to a long prompt, not GLM's parallel steps ([`Self::new_glm_par`]).
    pub fn can_pipe(&self) -> bool {
        self.stages.len() + usize::from(self.remote.is_some()) > 1
            && !self.par.every_step
            && !self.stages.iter().any(|s| s.backend.is_meta())
            && !self.model.experts().is_some_and(|r| r.is_lent())
    }

    /// Steps in flight ([`Self::pipe_round`]).
    pub fn pipe_in_flight(&self) -> usize {
        self.pipe.len()
    }

    /// Graphs each stage keeps for groups' steps: one per shape the groups take (their sizes,
    /// their slots, verify or not), so steps of groups that come back reuse them.
    pub fn set_pipe_graphs(&mut self, n: usize) {
        self.pipe_kept = n.max(1);
    }

    /// Before groups' steps: drops the whole decode steps' graphs (and their compute buffers),
    /// which the groups' smaller ones replace.
    pub fn pipe_begin(&mut self) {
        debug_assert!(self.pipe.is_empty());
        for s in &mut self.stages {
            if !s.backend.is_meta() {
                s.decode.clear();
            }
        }
    }

    /// After the groups' last steps (nothing in flight): drops their graphs and compute
    /// buffers, so prompts and whole steps have the memory they had before.
    pub fn pipe_end(&mut self) {
        debug_assert!(self.pipe.is_empty(), "groups' graphs dropped in flight");
        debug_assert!(self.pipe_run.is_none(), "groups' graphs dropped mid-round");
        for s in &mut self.stages {
            s.piped.clear();
            s.piped_shared = None;
        }
    }

    /// Forgets the steps in flight (after a failure), once every stage is idle, and drops
    /// their graphs.
    pub fn pipe_abort(&mut self) {
        // Pieces a failed round started on the launch threads.
        if let Some(pool) = &self.pipe_pool {
            let _ = pool.join();
        }
        // Their held commits go with the sequences.
        self.pipe_run = None;
        for s in &mut self.stages {
            s.backend.synchronize();
        }
        // Rank 1's answers to the steps sent (#562).
        if let Some(r) = self.remote.as_mut()
            && let Err(e) = r.sync()
        {
            tracing::warn!(error = %e, "node link: steps in flight dropped");
        }
        self.pipe.clear();
        self.pipe_end();
    }

    /// One round of groups' steps (see the module docs): every step in flight moves to the
    /// next stage, `enter` (the caller's tag and a decode or verify step of one group of
    /// sequences, as [`Self::step`] takes it, within any gather budget) starts on stage 0, and
    /// all stages with a piece run at once. Returns the step that left the last stage: its
    /// tag and output, as [`Self::step`] returns it. Nothing runs on the stages when it
    /// returns (each round waits for all), so between rounds the caller may sample, commit
    /// and draft: the states and KV blocks of groups in flight must stay untouched.
    ///
    /// A step entering while one is still on stage 0 (a round must move it first) panics. On
    /// an error every step in flight is dropped ([`Self::pipe_abort`]).
    pub fn pipe_round(
        &mut self,
        enter: Option<(usize, &[Group<'_>])>,
    ) -> Result<Option<(usize, StepOutput)>, ModelError> {
        let out = self.pipe_round_begin(enter)?;
        self.pipe_round_end()?;
        Ok(out)
    }

    /// [`Self::pipe_round`] that returns as soon as the step leaving the last stage is out
    /// (#762), while the other stages may still compute their pieces. Until
    /// [`Self::pipe_round_end`] the caller may sample, [`Self::commit`] (a stage still
    /// computing starts its share when it is done) and run MTP passes (on the last stage,
    /// which is done): what a round's end would have waited for runs beside the other stages.
    /// Anything else that touches the stages' states ([`Self::copy_states`] of some slots,
    /// [`Self::wait_commit`]) first waits for the round.
    pub fn pipe_round_begin(
        &mut self,
        enter: Option<(usize, &[Group<'_>])>,
    ) -> Result<Option<(usize, StepOutput)>, ModelError> {
        let r = self.pipe_round_run(enter);
        if r.is_err() {
            self.pipe_abort();
        }
        r
    }

    /// The end of the round [`Self::pipe_round_begin`] started: the other stages' pieces done
    /// and their outputs taken, the commits held for them started, and every commit waited
    /// for. Nothing runs on the stages when it returns.
    pub fn pipe_round_end(&mut self) -> Result<(), ModelError> {
        let r = self.pipe_round_rest();
        if r.is_err() {
            self.pipe_abort();
        }
        r
    }

    /// The last finished round's piece time per local stage, from its launch to its piece
    /// done (`None`: the stage had no piece), and whether its exiting step was handed out
    /// while other stages computed ([`Self::pipe_round_begin`]). Pieces run on launch threads
    /// only: a round of one piece reports `None` for every stage.
    pub fn pipe_last_round(&self) -> (&[Option<Duration>], bool) {
        (&self.pipe_last.0, self.pipe_last.1)
    }

    /// Whether a round's stages may still compute ([`Self::pipe_round_begin`]).
    pub fn pipe_running(&self) -> bool {
        self.pipe_run.is_some()
    }

    /// Ends a round still running ([`Self::pipe_round_end`]) for a call that touches every
    /// stage; its error waits for the next [`Self::pipe_round_end`].
    pub(super) fn pipe_settle(&mut self) {
        if self.pipe_run.is_some()
            && let Err(e) = self.pipe_round_end()
        {
            self.pipe_failed = Some(e.to_string());
        }
    }

    fn pipe_round_rest(&mut self) -> Result<(), ModelError> {
        if let Some(e) = self.pipe_failed.take() {
            return Err(ModelError::Ggml(GgmlError::External(e)));
        }
        let Some(mut run) = self.pipe_run.take() else {
            return Ok(());
        };
        while let Some(s) = self.pipe_next_done(&mut run)? {
            let out = self.pipe_take(&mut run, s)?;
            debug_assert!(out.is_none(), "a step left the last stage twice");
        }
        self.pipe_round_close(&run);
        // As after a whole step's drafts: the next round's inputs are written with every
        // commit done.
        self.wait_commit();
        Ok(())
    }

    fn pipe_round_run(
        &mut self,
        enter: Option<(usize, &[Group<'_>])>,
    ) -> Result<Option<(usize, StepOutput)>, ModelError> {
        assert!(
            self.pipe_run.is_none(),
            "a round began before the last one ended"
        );
        // Rank 1 of a split model is one more stage, after the local ones.
        let n_local = self.stages.len();
        let n_stages = n_local + usize::from(self.remote.is_some());
        assert!(n_stages > 1, "groups in flight need several stages");
        // Cache work deferred from the last passes (MTP drafts between rounds) before any
        // stage runs again.
        self.flush_deferred()?;
        // A whole verify step's commit no MTP pass queued (`commit_hold`): before any piece.
        self.queue_all_held()?;
        for p in &mut self.pipe {
            p.stage += 1;
        }
        debug_assert!(self.pipe.iter().all(|p| p.stage < n_stages));
        if let Some((tag, groups)) = enter {
            assert!(
                self.pipe.iter().all(|p| p.stage > 0),
                "stage 0 is taken this round"
            );
            let table = self
                .model
                .paged_for(&self.stages[0].kv, groups)
                .then_some(self.max_seq_blocks);
            let arch = &*self.model.hp.arch;
            let shape = with_kv(
                batch_shape(groups, table, Some(arch)),
                groups,
                &self.stages[0].kv,
            );
            assert!(
                self.budget_chunks(groups).len() == 1,
                "a group's step over the gather budget"
            );
            self.pipe.push(Piece {
                tag,
                groups: own(groups),
                shape,
                stage: 0,
                hidden: Vec::new(),
                paged: table.is_some(),
                rows: Vec::new(),
            });
        }
        if self.pipe.is_empty() {
            return Ok(None);
        }
        let t0 = Instant::now();
        // Later stages first: a stage's outgoing residual is taken by the next stage before
        // the stage is given its next piece (all of a stage's pieces share one buffer).
        self.pipe
            .sort_unstable_by_key(|p| std::cmp::Reverse(p.stage));
        let passes: Vec<(usize, usize)> = self
            .pipe
            .iter()
            .filter(|p| p.stage < n_local)
            .map(|p| (p.stage, p.shape.iter().map(|g| g.tokens()).sum()))
            .collect();
        // Several local pieces run on kept launch threads, one per stage (#756): a thread
        // spawned per stage per round cost ~0.9 ms each, one after another (#669). Each piece
        // starts once its own inputs are in, so the later stages (the last one, with the
        // head, the longest) compute while the earlier stages' inputs are filled.
        let pooled = passes.len() > 1;
        if pooled && self.pipe_pool.as_ref().is_none_or(|p| p.len() != n_local) {
            self.pipe_pool = Some(LaunchPool::new(n_local).map_err(|e| {
                ModelError::Unsupported(format!("groups in flight: launch threads: {e}"))
            })?);
        }
        // Off the engine thread's core (#250), worked out here, where it runs now.
        let stage_cpus = if pooled { self.par_cpus() } else { Vec::new() };
        let done: Arc<Mutex<Vec<Option<Instant>>>> = Arc::new(Mutex::new(vec![None; n_local]));
        // When each local stage's piece started, by stage.
        let mut started = vec![None; n_local];
        for i in 0..self.pipe.len() {
            let (s, shape) = (self.pipe[i].stage, self.pipe[i].shape.clone());
            if s == n_local {
                // Rank 1 starts on it now, beside the local stages.
                let rows = std::mem::take(&mut self.pipe[i].rows);
                let groups = view(&self.pipe[i].groups);
                let r = self.remote.as_mut().expect("a remote stage");
                r.send_step(&groups, rows)?;
                continue;
            }
            self.build_piped(s, &shape)?;
            let groups = view(&self.pipe[i].groups);
            self.fill_piped(s, &groups, &shape);
            drop(groups);
            let layers = self.stages[s].layers.clone();
            let n = shape.iter().map(|g| g.tokens()).sum();
            self.begin_pass(n, |il| layers.contains(&il), Kind::Decode)?;
            if !pooled {
                continue;
            }
            let Stage { piped, backend, .. } = &mut self.stages[s];
            let kept = piped_in(piped, &shape).expect("prepared");
            // SAFETY: from here until the pool hands this stage's result back (`join_next`
            // below, or `pipe_abort` on an error), nothing uses the stage's backend or graph:
            // the stages started later this round are earlier ones, which read this stage's
            // previous outgoing residual only before it starts (in `fill_piped`, above).
            let launcher = unsafe { backend.launcher(kept)? };
            let done = done.clone();
            let job: PoolLaunch = Box::new(move || {
                let r = launcher.run();
                if let Ok(mut d) = done.lock() {
                    d[s] = Some(Instant::now());
                }
                r
            });
            let cpus = stage_cpus.get(s).cloned().unwrap_or_default();
            let pool = self.pipe_pool.as_ref().expect("made above");
            started[s] = Some(t0.elapsed());
            pool.start(s, (cpus, job));
        }
        let t_go = t0.elapsed();
        let running = started.iter().map(Option::is_some).collect();
        let mut run = PipeRun {
            t0,
            t_go,
            pooled,
            order: passes.iter().map(|&(s, _)| s).collect(),
            passes,
            done,
            started,
            // When each stage's piece finished, by stage (#391: the stages' busy time, from
            // its start).
            stage_done: vec![t_go; n_stages],
            running,
            held: vec![Vec::new(); n_local],
            exited: false,
            exit_at: None,
        };
        if !pooled && let Some(p) = self.pipe.iter().find(|p| p.stage < n_local) {
            let Stage { piped, backend, .. } = &mut self.stages[p.stage];
            let kept = piped_in(piped, &p.shape).expect("prepared");
            backend.compute_kept(kept)?;
            run.stage_done[p.stage] = t0.elapsed();
        }
        // The step leaving the last stage may be handed out before the other stages are done
        // (#762): their pieces read only their own layers and slots, and what the caller does
        // with it meanwhile (sampling, the commit, MTP passes on the last stage's GPU) leaves
        // them alone (the commit is held for a stage still computing). Only with the MTP head
        // (without it only sampling is left to overlap: plain rounds stay as they were), not
        // with rank 1 (its step comes last anyway), nor where an MTP pass's cache work spans
        // every layer.
        let early = pooled && self.mtp_enabled && self.remote.is_none() && self.after_overlap;
        // Each local piece's output as soon as its stage is done, while the others compute;
        // in stage order without launch threads.
        let mut exited = None;
        while let Some(s) = self.pipe_next_done(&mut run)? {
            if let Some(x) = self.pipe_take(&mut run, s)? {
                exited = Some(x);
                if early {
                    run.exit_at = Some(t0.elapsed());
                    break;
                }
            }
        }
        if let Some(i) = self.pipe.iter().position(|p| p.stage == n_local) {
            let r = self.remote.as_mut().expect("a remote stage");
            let o = r.wait_step()?;
            run.stage_done[n_local] = t0.elapsed();
            exited = Some((
                i,
                StepOutput {
                    argmax: o.argmax,
                    rows: o.rows,
                    ..Default::default()
                },
            ));
        }
        run.exited = exited.is_some();
        self.pipe_run = Some(run);
        let Some((i, out)) = exited else {
            return Ok(None);
        };
        let mut p = self.pipe.remove(i);
        // Captured layers in the order asked for.
        let hidden = self
            .hidden
            .iter()
            .map(|il| {
                let at = p
                    .hidden
                    .iter()
                    .position(|(l, _)| l == il)
                    .expect("every captured layer is in a stage");
                std::mem::take(&mut p.hidden[at].1)
            })
            .collect();
        Ok(Some((p.tag, StepOutput { hidden, ..out })))
    }

    /// The next local stage whose piece is done (its launch thread joined), `None` when every
    /// piece's output was taken.
    fn pipe_next_done(&mut self, run: &mut PipeRun) -> Result<Option<usize>, ModelError> {
        if !run.pooled {
            return Ok(run.order.pop_front());
        }
        let pool = self.pipe_pool.as_ref().expect("made with the round");
        let Some((s, r)) = pool.join_next() else {
            return Ok(None);
        };
        run.running[s] = false;
        r?;
        if let Some(at) = run.done.lock().ok().and_then(|d| d[s]) {
            run.stage_done[s] = at - run.t0;
        }
        Ok(Some(s))
    }

    /// Takes the output of stage `s`'s piece (done), runs the expert cache's work for its
    /// layers and starts the commits held for it. Returns the piece's index and output when
    /// it left the last stage.
    fn pipe_take(
        &mut self,
        run: &mut PipeRun,
        s: usize,
    ) -> Result<Option<(usize, StepOutput)>, ModelError> {
        let n_local = self.stages.len();
        let n_stages = n_local + usize::from(self.remote.is_some());
        let n_vocab = self.model.hp.n_vocab as usize;
        let exact = self.model.hp.arch.argmax_first_of_ties();
        let mut exited = None;
        {
            let i = (self.pipe.iter())
                .position(|p| p.stage == s)
                .expect("a piece ran on the stage");
            let parts = self.parts(s);
            let captured = self.captured(s);
            let Stage {
                piped,
                bounce,
                bounce_small,
                ids_pin,
                backend,
                kv,
                ..
            } = &mut self.stages[s];
            backend.synchronize();
            let p = &mut self.pipe[i];
            let kept = piped_in(piped, &p.shape).expect("prepared");
            let inputs = Inputs::from_parts(&p.shape, parts, |j| kept.tensor(j));
            let groups = view(&p.groups);
            let spare = &mut self.spare_rows;
            let ids = bounce_for(ids_pin, backend, inputs.out_pin_bytes(&groups, n_vocab));
            let picks = super::super::Picks {
                stop: &self.stop_tokens,
                exact,
            };
            let o = inputs.output_with(&groups, n_vocab, ids, picks, |t| {
                read_rows(bounce, bounce_small, spare, backend, t)
            });
            // Model bytes (#391): the piece's graph on this stage and the KV it read; the
            // stage's busy time from the launches to its piece done.
            let tier = (kv.hbm_blocks() < kv.n_blocks()).then_some(kv.hbm_blocks());
            let (kv_dev, kv_host) = self.kv_rates[s].step(&groups, p.paged, tier);
            drop(groups);
            let mut part = Phase {
                steps: 1,
                kv: kv_dev,
                kv_host,
                seconds: (run.stage_done[s].saturating_sub(run.started[s].unwrap_or(run.t_go)))
                    .as_secs_f64(),
                ..Phase::default()
            };
            part.add_graph(kept.model_bytes());
            if let Some(st) = self.bytes.stage_mut(s, Kind::Decode) {
                st.merge(&part);
            }
            let d = self.bytes.phase_mut(Kind::Decode);
            d.add_graph(kept.model_bytes());
            d.kv += kv_dev;
            d.kv_host += kv_host;
            p.hidden.extend(captured.into_iter().zip(o.hidden));
            if s + 1 == n_stages {
                exited = Some((
                    i,
                    StepOutput {
                        argmax: o.argmax,
                        rows: o.rows,
                        ..Default::default()
                    },
                ));
            } else if s + 1 == n_local
                && let Some(t) = inputs.res_out
            {
                // Rank 1 takes it next round.
                p.rows = read_rows(bounce, bounce_small, spare, backend, t);
            }
            // The cache learns the piece's routing and moves experts of its stage's layers now,
            // while the stages still running read only their own layers (as a whole step's
            // shares do, `after_split`): where the work runs changes when copies land in their
            // slots, not what any piece computes.
            let n = (run.passes.iter())
                .find(|&&(x, _)| x == s)
                .map_or(0, |&(_, n)| n);
            let layers = self.stages[s].layers.clone();
            self.after_pass_only(n, |il| layers.contains(&il), Kind::Decode)?;
        }
        // Commits of a step that left meanwhile (#762), in the order they were asked for.
        for commits in std::mem::take(&mut run.held[s]) {
            let st = &mut self.stages[s];
            let spec = (st.spec.as_ref()).expect("commit without Executor::enable_spec");
            // SAFETY: as in `Executor::commit`.
            unsafe {
                spec.commit(
                    &mut st.backend,
                    &mut st.commits,
                    &st.rec,
                    &commits,
                    &self.model,
                )
            }?;
        }
        Ok(exited)
    }

    /// The round's log line and model bytes, once every piece is done.
    fn pipe_round_close(&mut self, run: &PipeRun) {
        // Handed out early (#762): until its last piece was done, not over the caller's work
        // beside it (MTP passes count their own time).
        let wall = match run.exit_at {
            Some(at) => run.stage_done.iter().copied().fold(at, Duration::max),
            None => run.t0.elapsed(),
        };
        tracing::trace!(
            pieces = ?run.passes,
            start_ms = run.t_go.as_secs_f64() * 1e3,
            stage_done_ms = ?run.stage_done.iter().map(|d| d.as_secs_f64() * 1e3).collect::<Vec<_>>(),
            exit_ms = run.exit_at.map(|d| d.as_secs_f64() * 1e3),
            total_ms = wall.as_secs_f64() * 1e3,
            "pipe round"
        );
        // Each local stage's busy time for `--pipeline-mtp-split` (#765).
        self.pipe_last.0.clear();
        self.pipe_last.0.extend(
            (run.started.iter().zip(&run.stage_done))
                .map(|(st, &done)| st.map(|st| done.saturating_sub(st))),
        );
        self.pipe_last.1 = run.exit_at.is_some();
        // Model bytes (#391): every round is decode work, over its wall time; a step counts
        // when it leaves the last stage.
        let d = self.bytes.phase_mut(Kind::Decode);
        d.seconds += wall.as_secs_f64();
        d.steps += u64::from(run.exited);
    }

    /// Builds stage `s`'s graph of a piece of `shape` unless it keeps one.
    fn build_piped(&mut self, s: usize, shape: &[GroupShape]) -> Result<(), ModelError> {
        let keep = self.pipe_kept;
        let st = &mut self.stages[s];
        if let Some(i) = st.piped.iter().position(|(x, _)| x == shape) {
            // Most recent last.
            let g = st.piped.remove(i);
            st.piped.push(g);
            return Ok(());
        }
        let captured = self.captured(s);
        let stage = &mut self.stages[s];
        if stage.piped.len() >= keep {
            stage.piped.remove(0);
        }
        let buft = stage.backend.buffer_type();
        let shared = stage
            .piped_shared
            .get_or_insert_with(|| {
                std::sync::Arc::new(std::sync::Mutex::new(ComputeAllocator::new(buft)))
            })
            .clone();
        let opts = StepOpts {
            spec: stage.spec.as_ref(),
            saved: None,
            hidden: &captured,
            mtp: self.mtp_enabled,
            par_rank: s,
        };
        let t_build = Instant::now();
        let size = graph_size(shape.len());
        let ctx = Context::new(size, size);
        let parts = {
            let (graph, inputs, _) = self.model.build_stage(
                &ctx,
                &stage.kv,
                &stage.rec,
                opts,
                shape,
                false,
                stage.layers.clone(),
            )?;
            if ctx.alloc_graph_shared(&graph, &shared)? {
                // The buffer moved: the other kept graphs point into the old one.
                stage.piped.clear();
            }
            KeptParts::new(&graph, &inputs.list())
        };
        tracing::debug!(
            stage = s,
            shape = ?shape.iter().map(|g| (g.slot, g.n_seqs, g.n_tok)).collect::<Vec<_>>(),
            ms = t_build.elapsed().as_secs_f64() * 1e3,
            mib = ctx.allocator_size() as f64 / (1u64 << 20) as f64,
            "group step graph built"
        );
        // SAFETY: as for the whole steps' graphs (`build_kept`): the graph references the
        // weights and the stage's pools, owned by self and never replaced while it is kept;
        // the kept graphs are dropped first (field order, `forget_graphs`).
        let mut kept = unsafe { KeptGraph::new(ctx, parts) };
        crate::ggml::stage_graph::stats::note_build("piped", s, || {
            crate::ggml::stage_graph::stats::BuildNote {
                shape: GroupShape::brief(shape),
                ..Default::default()
            }
        });
        Self::count_bytes(&mut self.weight_addrs, &self.model, &mut kept);
        self.stages[s].piped.push((shape.to_vec(), kept));
        Ok(())
    }

    /// Fills stage `s`'s inputs of a piece of `groups`, and its incoming residual from stage
    /// `s - 1`'s graph of the same shape (which ran the piece last round).
    fn fill_piped(&self, s: usize, groups: &[Group<'_>], shape: &[GroupShape]) {
        let stage = &self.stages[s];
        let kept = piped_in(&stage.piped, shape).expect("prepared");
        let inputs = Inputs::from_parts(shape, self.parts(s), |i| kept.tensor(i));
        inputs.fill(groups, shape);
        self.model.fill_extra(
            &inputs,
            groups,
            shape,
            &stage.rec,
            &stage.kv,
            stage.spec.as_ref(),
            &mut crate::model::core::SetNow,
        );
        if let Some(dst) = inputs.res_in {
            let prev = piped_in(&self.stages[s - 1].piped, shape).expect("the previous stage ran");
            let src = Inputs::from_parts(shape, self.parts(s - 1), |i| prev.tensor(i))
                .res_out
                .expect("a stage before the last hands its residual on");
            dst.copy_from(src);
        }
    }
}

#[cfg(test)]
mod tests;

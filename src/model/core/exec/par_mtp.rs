//! GLM-5.3-Flash's MTP passes on every GPU (`--glm-parallel-mtp`, #717): a pass with output
//! rows runs one graph per GPU ([`Model::glm_par_mtp`](crate::model::core::Model)), the last
//! GPU's with the head's inputs, attention and outputs, every GPU's with its share of the
//! head's FFN and of its draft logits, one exchange between them per sublayer. A pass without
//! output rows (a prompt's take-in before its last piece) stays on the last GPU
//! ([`Executor::mtp_pass_run`]).
//!
//! The graphs are kept per GPU and kind of pass, as the last GPU's are without it
//! ([`MtpGraphs`]). The expert cache's entries of the head's layer (the last GPU's and the
//! other GPUs' copies, [`crate::model::glm5next::par::mtp_copies`]) learn the routing after the
//! pass, beside the next target step, as the head's single entry did.

use super::mtp_graphs::{MTP_KEPT, MtpGraphs};
use super::par_step::pool::{Launch, LaunchPool};
use super::par_step::wait_sums;
use super::{BOUNCE_MIN_BYTES, Executor, Kind, mtp_window, write_rows};
use crate::ggml::{Context, GgmlError, KeptGraph, KeptParts};
use crate::model::core::mtp::MtpOutput;
use crate::model::core::{GRAPH_SIZE, Group, GroupShape, Inputs, ModelError};

#[cfg(test)]
mod tests;

/// Per GPU, the kept graphs of the parallel MTP passes: draft passes (one token per sequence)
/// and take-in passes with output rows.
pub(super) struct ParMtpGraphs {
    pub decode: MtpGraphs,
    pub mixed: MtpGraphs,
}

impl ParMtpGraphs {
    pub(super) fn new() -> Self {
        Self {
            decode: MtpGraphs::new(MTP_KEPT),
            mixed: MtpGraphs::new(MTP_KEPT),
        }
    }

    pub(super) fn clear(&mut self) {
        self.decode.clear();
        self.mixed.clear();
    }

    pub(super) fn buffer_size(&self) -> usize {
        self.decode.buffer_size() + self.mixed.buffer_size()
    }
}

/// Whether an MTP pass of `n_out` output rows runs on every GPU: under parallel steps
/// (`every_step`) with the head split (`split`, `--glm-parallel-mtp`), and with rows to draft.
pub(super) fn runs_par(every_step: bool, split: bool, n_out: usize) -> bool {
    every_step && split && n_out > 0
}

/// Whether the #723 order applies to a step of `seqs` sequences under
/// `--glm-mtp-overlap-seqs max_seqs` (#725). On four V100s it took C=1 decode 58 -> 61 tok/s
/// but C=4 66.6 -> 63.4 tok/s total at the same draft depth, so by default it stops at one
/// sequence.
pub(super) fn overlap_applies(max_seqs: usize, seqs: usize) -> bool {
    seqs > 0 && seqs <= max_seqs
}

/// A GPU's launch job in the MTP pass on every GPU: its held verify commit and its pass graph,
/// in the order given (the pass first under `--glm-mtp-overlap`, #723). The second is queued
/// even when the first fails to start (each graph is computed as launched); the first error
/// is returned.
pub(super) fn in_order(
    first: impl FnOnce() -> Result<(), GgmlError>,
    second: impl FnOnce() -> Result<(), GgmlError>,
) -> Result<(), GgmlError> {
    let a = first();
    let b = second();
    a.and(b)
}

/// Whether expert-cache entry `il` is the MTP head's (a model of `n_layer` layers): its own
/// entry or one of its copies on the other GPUs (`copies`).
pub(super) fn is_mtp_entry(il: usize, n_layer: usize, copies: &std::ops::Range<usize>) -> bool {
    il == n_layer || copies.contains(&il)
}

impl Executor {
    /// Whether the MTP head's passes run on every GPU (`--glm-parallel-mtp`).
    pub(super) fn mtp_split(&self) -> bool {
        self.par.every_step && self.model.glm_par.as_ref().is_some_and(|p| p.plan.mtp)
    }

    /// `--glm-mtp-overlap-seqs` (#723, #725): with the MTP pass on every GPU, after a step of
    /// at most `max_seqs` sequences each GPU's verify commit is queued after its pass's graph
    /// by its launch thread (the last GPU's too, not from this thread before the pass), and
    /// the target step's cache work runs while the launch threads queue the pass. Larger
    /// steps, and every step with 0, keep the order of #714 and #717.
    pub fn set_glm_mtp_overlap(&mut self, max_seqs: usize) {
        self.par.mtp_overlap_seqs = max_seqs;
    }

    /// Whether [`Self::set_glm_mtp_overlap`] applies to the current step (its commit and the
    /// pass after it), with the MTP pass on every GPU.
    pub(super) fn mtp_overlap(&self) -> bool {
        self.par.overlap_step && self.mtp_split()
    }

    /// The expert cache's entries of the MTP head's copies on the other GPUs (none without
    /// `--glm-parallel-mtp`).
    pub(super) fn mtp_copy_entries(&self) -> std::ops::Range<usize> {
        let n_entries = self.model.hp.n_layer + 1;
        match self.model.glm_par.as_ref() {
            Some(p) if self.par.every_step => {
                crate::model::glm5next::par::mtp_copies(&p.plan, n_entries)
            }
            _ => n_entries..n_entries,
        }
    }

    /// The expert cache's entries an MTP pass reads: the head's, and its copies.
    pub(super) fn mtp_entries(&self) -> impl Fn(usize) -> bool + use<> {
        let n_layer = self.model.hp.n_layer;
        let copies = self.mtp_copy_entries();
        move |il| is_mtp_entry(il, n_layer, &copies)
    }

    /// Whether the MTP pass of `groups` runs on every GPU ([`runs_par`]).
    pub(super) fn mtp_runs_par(&self, groups: &[Group<'_>]) -> bool {
        let n_out = self.mtp_shape(groups).iter().map(|g| g.out_rows()).sum();
        runs_par(self.par.every_step, self.mtp_split(), n_out)
    }

    /// [`Executor::mtp_pass_run`] on every GPU (see the module docs).
    pub(super) fn mtp_pass_par(
        &mut self,
        groups: &[Group<'_>],
        h: &[&[f32]],
    ) -> Result<MtpOutput, ModelError> {
        let t0 = std::time::Instant::now();
        let shape = self.mtp_shape(groups);
        let n: usize = shape.iter().map(GroupShape::tokens).sum();
        // The previous pass's cache work touches this pass's entries: before it.
        self.flush_deferred_mtp()?;
        let target_work = self.take_target_work()?;
        let target = self.target_entries();
        let draft = self.mtp_entries();
        self.begin_pass(n, &draft, Kind::Draft)?;
        let n_stages = self.stages.len();
        let last = n_stages - 1;
        if self.par_mtp.len() != n_stages {
            self.par_mtp = (0..n_stages).map(|_| ParMtpGraphs::new()).collect();
        }
        let decode = shape.iter().all(|g| g.n_tok == 1);
        for s in 0..n_stages {
            self.prepare_par_mtp(s, &shape, decode)?;
        }

        // The last GPU's inputs, as a pass on it alone takes them.
        let arch = &self.model.hp.arch;
        let n_group = if arch.mtp_sparse() {
            arch.group_inputs_len()
        } else {
            0
        };
        let exclusive = !arch.mtp_sees_own_position();
        let swa = self.model.swa_window();
        let kv = self.mtp_kv.as_ref().expect("mtp_step without enable_mtp");
        let graphs = &self.par_mtp[last];
        let slot = if decode {
            &graphs.decode
        } else {
            &graphs.mixed
        };
        let kept = slot.get(&shape).expect("prepared");
        let st = &mut self.stages[last];
        let mut inputs =
            Inputs::from_list(&shape, 0, false, n_group, false, swa, |i| kept.tensor(i));
        inputs.exclusive = exclusive;
        let n_in = inputs.list().len();
        st.mtp_io.begin(&mut st.backend, mtp_window::PINNED_MAX);
        inputs.fill_into(
            groups,
            &shape,
            &mut mtp_window::Queued {
                io: &mut st.mtp_io,
                backend: &mut st.backend,
            },
        );
        if n_group > 0 {
            self.model.fill_extra(
                &inputs,
                groups,
                &shape,
                &st.rec,
                kv,
                None,
                &mut mtp_window::Queued {
                    io: &mut st.mtp_io,
                    backend: &mut st.backend,
                },
            );
        }
        let h_in = kept.tensor(n_in);
        let h_bytes: usize = h.iter().map(|p| std::mem::size_of_val(*p)).sum();
        if h_bytes >= BOUNCE_MIN_BYTES {
            write_rows(&mut st.bounce, &st.backend, h_in, h);
        } else {
            st.mtp_io.write_f32_parts(&mut st.backend, h_in, h);
        }

        // Every GPU's graph queued by its launch thread at once (each waits in its exchanges
        // for the others), then the host goes on to the target step's cache work.
        let par = self.model.glm_par.as_ref().expect("mtp_split");
        let ex = par.ex.clone();
        self.par.step += 1;
        ex.begin_step(self.par.step);
        let stage_cpus = self.par_cpus();
        if self.par.pool.as_ref().is_none_or(|p| p.len() != n_stages) {
            self.par.pool = Some(LaunchPool::new(n_stages).map_err(|e| {
                ModelError::Unsupported(format!("parallel MTP: launch threads: {e}"))
            })?);
        }
        // The other GPUs' verify commits held for the MTP pass (`commit_par`): each queued by
        // its launch thread ahead of the pass's graph, as the last GPU's is on its stream
        // (the stream order step, commit, pass, next step on every GPU). With
        // `--glm-mtp-overlap` every GPU's, the last's too, is queued right after the pass's
        // graph (#723), so no GPU's pass waits for commit launches (the order step, pass,
        // commit, next step): the head reads none of the recurrent states a commit writes,
        // and a commit reads nothing the pass writes.
        let overlap = self.mtp_overlap();
        let held = self.par.take_held(n_stages);
        let mut launchers = Vec::with_capacity(n_stages);
        let mut res = Ok(());
        for (s, st) in self.stages.iter_mut().enumerate() {
            let g = &self.par_mtp[s];
            let kept = if decode { &g.decode } else { &g.mixed }
                .get(&shape)
                .expect("prepared");
            // SAFETY: nothing uses a stage's backend or this graph until its launch is queued
            // (the pool's run returns) and the stage is synchronized below.
            match unsafe { st.backend.launcher(kept) } {
                Ok(l) => launchers.push(l),
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        // A GPU's graph alone would wait for the others' exchanges: none of them if one cannot
        // launch. The held commits are queued either way (each graph computed as launched).
        if res.is_err() {
            launchers.clear();
        }
        let mut launchers = launchers.into_iter();
        let jobs: Vec<(Vec<u32>, Launch)> = held
            .into_iter()
            .enumerate()
            .map(|(s, commit)| {
                let pass = launchers.next();
                let cpus = stage_cpus.get(s).cloned().unwrap_or_default();
                let job = move || {
                    let pass = || pass.map_or(Ok(()), |l| l.start());
                    let commit = || commit.map_or(Ok(()), |launch| launch());
                    if overlap {
                        in_order(pass, commit)
                    } else {
                        in_order(commit, pass)
                    }
                };
                (cpus, Box::new(job) as Launch)
            })
            .collect();
        // The target step's cache work while the GPUs compute (it reads no entry of the head):
        // after the launches, or with `--glm-mtp-overlap` on this thread while they launch
        // (#723; waiting for them first left the GPUs idle through it). Not done under a
        // launch error (the pass fails either way).
        let (model, pair) = (&mut self.model, &mut self.pair);
        let mut cache_work = Some(move || {
            let Some((n, kind, layers, last_share)) = target_work else {
                return (Ok(()), None);
            };
            let ran = |il: usize| layers.contains(&il) && target(il);
            let work = match model.experts_mut() {
                Some(r) if last_share => r.after_step_only(n, pair.widen(ran)),
                Some(r) => r.after_step_more(n, pair.widen(ran)),
                None => Ok(()),
            };
            pair.learned(ran);
            (work, Some(kind))
        });
        // The launchers made so far are started either way (each holds its stage's graph);
        // a GPU whose graph did not start fails the others' exchanges after their timeout.
        let pool = self.par.pool.as_ref().expect("made above");
        let mut done = None;
        let launched = match cache_work.take() {
            Some(f) if overlap && res.is_ok() => {
                let (launched, out) = pool.run_beside(jobs, f);
                done = Some(out);
                launched
            }
            f => {
                cache_work = f;
                pool.run(jobs)
            }
        };
        for r in launched {
            if res.is_ok() {
                res = r;
            }
        }
        let st = &mut self.stages[last];
        let graphs = &self.par_mtp[last];
        let kept = if decode {
            &graphs.decode
        } else {
            &graphs.mixed
        }
        .get(&shape)
        .expect("prepared");
        let reads = res.is_ok().then(|| {
            let argmax = inputs.argmax.expect("argmax");
            [argmax, kept.tensor(n_in + 1), kept.tensor(n_in + 2)]
                .map(|t| st.mtp_io.read(&mut st.backend, t))
        });
        let (work, credit) = match (done, cache_work) {
            (Some(out), _) => out,
            (None, Some(f)) if res.is_ok() => f(),
            _ => (Ok(()), None),
        };
        for (s, st) in self.stages.iter_mut().enumerate() {
            if s == last {
                st.mtp_io.finish(&mut st.backend);
            } else {
                st.backend.synchronize();
            }
        }
        res?;
        if let Err(e) = work {
            return Err(ModelError::Ggml(GgmlError::External(format!(
                "expert cache: {e}"
            ))));
        }
        if ex.failed() {
            return Err(ModelError::Unsupported(
                "parallel MTP: a GPU waited over 10 s for another's exchange".into(),
            ));
        }
        // The waits of this pass, apart from the steps' ([`Executor::par_report`]).
        let (_, late) = wait_sums(&ex.take_wait_matrix());
        if self.par.mtp_late.len() != late.len() {
            self.par.mtp_late = vec![0; late.len()];
        }
        for (a, w) in self.par.mtp_late.iter_mut().zip(late) {
            *a += w;
        }
        self.par.mtp_passes += 1;
        let st = &mut self.stages[last];
        let mut out = MtpOutput::default();
        if let Some([argmax, pmax, hidden]) = reads {
            out.argmax = (st.mtp_io.take_i32(argmax).into_iter())
                .map(|x| x as u32)
                .collect();
            out.pmax = st.mtp_io.take_f32(pmax);
            out.hidden = st.mtp_io.take_f32(hidden);
        }
        if let Some(kind) = credit {
            self.credit_experts(kind);
        }
        // Model bytes (#391): the last GPU's graph, as a pass on it alone counts.
        if let Some(kept) = self.par_mtp[last].get(&shape, decode) {
            self.bytes
                .draft
                .add(kept.model_bytes(), t0.elapsed().as_secs_f64());
        }
        if self.after_overlap {
            self.deferred_mtp = Some(n);
        } else {
            self.after_pass(n, draft, Kind::Draft)?;
        }
        Ok(out)
    }

    /// Stage `s`'s kept graph of the parallel MTP pass of `shape`, built if not kept, made
    /// ready to run.
    fn prepare_par_mtp(
        &mut self,
        s: usize,
        shape: &[GroupShape],
        decode: bool,
    ) -> Result<(), ModelError> {
        let last = s + 1 == self.stages.len();
        let st = &mut self.stages[s];
        let graphs = &mut self.par_mtp[s];
        let slot = if decode {
            &mut graphs.decode
        } else {
            &mut graphs.mixed
        };
        if slot.touch(shape) {
            st.backend
                .ready_kept(slot.get(shape).expect("just touched"))?;
            return Ok(());
        }
        // The last GPU's graph reads the head's cache; the others' read no cache.
        let kv = match &self.mtp_kv {
            Some(kv) if last => kv,
            _ => &st.kv,
        };
        let t_build = std::time::Instant::now();
        slot.evict_for_one();
        let (ctx, parts) = loop {
            let ctx = Context::new(GRAPH_SIZE, GRAPH_SIZE);
            let parts = {
                let (graph, g) = self.model.glm_par_mtp(&ctx, kv, &st.rec, shape, s)?;
                match slot.alloc(&ctx, &graph, st.backend.buffer_type()) {
                    Err(GgmlError::Alloc { .. }) if self.freeing.wait() => continue,
                    r => r?,
                }
                let list = if last {
                    let mut list = g.inputs.list();
                    list.push(g.h);
                    list.extend(g.pmax);
                    list.extend(g.h_out);
                    list
                } else {
                    Vec::new()
                };
                KeptParts::new(&graph, &list)
            };
            break (ctx, parts);
        };
        // SAFETY: as for the last GPU's MTP graphs: the graph references the weights, the
        // exchange and (the last GPU's) the head's pool, which only `resize_kv` replaces, and
        // it drops these graphs first.
        let mut kept = unsafe { KeptGraph::new(ctx, parts) };
        Self::count_bytes(&mut self.weight_addrs, &self.model, &mut kept);
        tracing::trace!(
            gpu = s,
            shape = ?shape.iter().map(|g| (g.n_seqs, g.n_tok)).collect::<Vec<_>>(),
            ms = t_build.elapsed().as_secs_f64() * 1e3,
            "parallel mtp graph built"
        );
        let kept = slot.push(shape.to_vec(), kept);
        st.backend.ready_kept(kept)?;
        Ok(())
    }
}

impl ParMtpGraphs {
    /// The kept graph of `shape` of its kind (`decode`).
    pub(super) fn get(&self, shape: &[GroupShape], decode: bool) -> Option<&KeptGraph> {
        if decode {
            self.decode.get(shape)
        } else {
            self.mixed.get(shape)
        }
    }
}

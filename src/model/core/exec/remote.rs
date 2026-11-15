//! The layers past this machine's share, on another node (#562, [`crate::node`]).
//!
//! On rank 0 the executor's stages hold the layers before the split: its last stage hands its
//! residual out (as a stage before the last does) and every step's rows are sent to rank 1
//! with the step's sequences ([`Remote`]); the logits come back from there. A step that
//! returns no logits (a prompt chunk before the last) does not wait: rank 1 runs it while
//! this node runs the next one. Groups' steps in flight ([`Executor::pipe_round`]) count rank 1
//! as one more stage.
//!
//! On rank 1 the stages hold the layers from the split on: its first stage reads the residual
//! that came over the link ([`Executor::step_from`]) instead of embedding tokens.

use std::collections::VecDeque;

use super::Executor;
use crate::model::core::{Group, Logits, ModelError, SeqInput, StepOutput, batch_shape};
use crate::node::{Link, Reply, Request, WireDraft, WireGroup, WireSeq};

/// Rank 0's link to the layers on rank 1.
pub struct Remote {
    link: Link,
    /// The current step's residual rows so far, one entry per micro-chunk.
    rows: Vec<Vec<f32>>,
    /// Steps since the last timing line.
    logged: u64,
}

fn link_err(e: String) -> ModelError {
    ModelError::Node(e)
}

/// `l` as it goes over the link: a tag and a count (`Logits::FirstPick`'s two counts, verify
/// rows, in 16 bits each).
pub fn logits_tag(l: Logits) -> (u8, u32) {
    match l {
        Logits::None => (0, 0),
        Logits::Last => (1, 0),
        Logits::Greedy => (2, 0),
        Logits::All => (3, 0),
        Logits::First(n) => (4, n as u32),
        Logits::FirstGreedy(n) => (5, n as u32),
        Logits::Pick => (6, 0),
        Logits::FirstPick(n, m) => {
            let n = n.min(0xffff) as u32;
            (7, n | ((m as u32).min(n) << 16))
        }
    }
}

pub fn logits_of(tag: (u8, u32)) -> Result<Logits, String> {
    Ok(match tag {
        (0, _) => Logits::None,
        (1, _) => Logits::Last,
        (2, _) => Logits::Greedy,
        (3, _) => Logits::All,
        (4, n) => Logits::First(n as usize),
        (5, n) => Logits::FirstGreedy(n as usize),
        (6, _) => Logits::Pick,
        (7, nm) => Logits::FirstPick((nm & 0xffff) as usize, (nm >> 16) as usize),
        (t, _) => return Err(format!("unknown logits tag {t}")),
    })
}

/// `groups` as they go over the link.
pub fn to_wire(groups: &[Group<'_>]) -> Vec<WireGroup> {
    groups
        .iter()
        .map(|g| WireGroup {
            slot: g.slot as u32,
            verify: g.verify,
            seqs: g
                .seqs
                .iter()
                .map(|s| WireSeq {
                    tokens: s.tokens.to_vec(),
                    n_past: s.n_past as u32,
                    blocks: s.blocks.to_vec(),
                    ring: s.ring as u32,
                    logits: logits_tag(s.logits),
                })
                .collect(),
        })
        .collect()
}

/// Groups borrowing `wire` (see [`to_wire`]).
pub fn from_wire(wire: &[WireGroup]) -> Result<Vec<Group<'_>>, String> {
    wire.iter()
        .map(|g| {
            Ok(Group {
                slot: g.slot as usize,
                verify: g.verify,
                seqs: g
                    .seqs
                    .iter()
                    .map(|s| {
                        Ok(SeqInput {
                            tokens: &s.tokens,
                            n_past: s.n_past as usize,
                            blocks: &s.blocks,
                            ring: s.ring as usize,
                            logits: logits_of(s.logits)?,
                            media: None,
                        })
                    })
                    .collect::<Result<_, String>>()?,
            })
        })
        .collect()
}

/// Whether any sequence of `groups` returns logits.
fn wants_logits(groups: &[Group<'_>]) -> bool {
    groups
        .iter()
        .any(|g| g.seqs.iter().any(|s| s.logits != Logits::None))
}

impl Remote {
    pub fn new(link: Link) -> Self {
        Self {
            link,
            rows: Vec::new(),
            logged: 0,
        }
    }

    /// A micro-chunk's residual rows, read off the last stage.
    pub(super) fn push_rows(&mut self, rows: Vec<f32>) {
        self.rows.push(rows);
    }

    /// The step's residual rows, every micro-chunk's in order.
    pub(super) fn take_rows(&mut self) -> Vec<f32> {
        let mut parts = std::mem::take(&mut self.rows);
        if parts.len() == 1 {
            return parts.pop().expect("one part");
        }
        parts.concat()
    }

    /// Sends a step of `groups` with the residual `rows` its layers on rank 1 start from;
    /// when it returns logits, waits for them.
    pub(super) fn step(
        &mut self,
        groups: &[Group<'_>],
        rows: Vec<f32>,
    ) -> Result<Option<StepOutput>, ModelError> {
        self.send_step(groups, rows)?;
        if !wants_logits(groups) {
            return Ok(None);
        }
        self.wait_step().map(Some)
    }

    pub(super) fn send_step(
        &mut self,
        groups: &[Group<'_>],
        rows: Vec<f32>,
    ) -> Result<(), ModelError> {
        self.link
            .send(Request::Step {
                groups: to_wire(groups),
                rows,
            })
            .map_err(link_err)
    }

    /// The output of the last step sent (every reply before it read too).
    pub(super) fn wait_step(&mut self) -> Result<StepOutput, ModelError> {
        let out = match self.link.wait().map_err(link_err)? {
            Reply::Step { argmax, rows, .. } => StepOutput {
                argmax,
                rows,
                ..Default::default()
            },
            r => return Err(ModelError::Node(format!("a step answered with {r:?}"))),
        };
        self.logged += 1;
        if self.logged >= 256 {
            let t = std::mem::take(&mut self.link.times);
            let n = t.steps.max(1) as f64;
            if t.drafts > 0 {
                let d = t.drafts as f64;
                tracing::info!(
                    steps = t.steps,
                    rank1_ms = t.remote_ms / n,
                    link_ms = t.link_ms / n,
                    drafts = t.drafts,
                    rank1_draft_ms = t.draft_remote_ms / d,
                    draft_link_ms = t.draft_link_ms / d,
                    "node link: per step and MTP draft round waited for"
                );
            } else {
                tracing::info!(
                    steps = t.steps,
                    rank1_ms = t.remote_ms / n,
                    link_ms = t.link_ms / n,
                    "node link: per step waited for"
                );
            }
            self.logged = 0;
        }
        Ok(out)
    }

    /// Queues a state change rank 1 mirrors; its reply is read with the next step's.
    pub(super) fn mirror(&mut self, req: Request) {
        if let Err(e) = self.link.send(req) {
            tracing::error!(error = %e, "node link: a state change was not sent");
        }
    }

    /// Waits for every request sent so far.
    pub fn sync(&mut self) -> Result<(), ModelError> {
        if self.link.owed() == 0 {
            return Ok(());
        }
        self.link.wait().map(|_| ()).map_err(link_err)
    }

    /// Sends a request and waits for its reply.
    pub fn call(&mut self, req: Request) -> Result<Reply, ModelError> {
        self.link.call(req).map_err(link_err)
    }

    /// `--spec mtp` over two nodes (#560): rank 1's MTP drafts for `reqs` after the last step
    /// sent, from the rows that step captured there, with the head passes they took. Without
    /// `wait` (no sequence drafts: they only take the step in) the round is queued and its
    /// reply read with the next one's: `None`.
    pub fn drafts(
        &mut self,
        reqs: Vec<WireDraft>,
        p_min: f32,
        wait: bool,
    ) -> Result<Option<(Vec<Vec<(u32, f32)>>, usize)>, ModelError> {
        let req = Request::Draft {
            reqs,
            p_min: p_min.to_bits(),
        };
        if !wait {
            self.link.send(req).map_err(link_err)?;
            return Ok(None);
        }
        match self.link.call(req).map_err(link_err)? {
            Reply::Drafts { drafts, passes, .. } => Ok(Some((drafts, passes as usize))),
            r => Err(ModelError::Node(format!(
                "a draft round answered with {r:?}"
            ))),
        }
    }
}

impl Executor {
    /// Rank 0 of a model split over two nodes: the layers after the stages' run on the far
    /// end of `link` (see the module docs).
    pub fn set_remote(&mut self, remote: Remote) {
        assert!(
            self.stages
                .last()
                .is_some_and(|s| s.layers.end < self.model.hp.n_layer),
            "a remote tail needs layers left after the stages"
        );
        self.remote = Some(remote);
    }

    /// The link to rank 1, when this is rank 0 of a split model.
    pub fn remote_mut(&mut self) -> Option<&mut Remote> {
        self.remote.as_mut()
    }

    /// Whether this is rank 0 of a split model (the layers past its stages on rank 1).
    pub fn has_remote(&self) -> bool {
        self.remote.is_some()
    }

    /// Rank 0 of a split model: sequence `seq` has ended, so rank 1's MTP drafter forgets it
    /// (#560); its reply is read with the next step's. Nothing on one machine.
    pub fn release_remote_drafts(&mut self, seq: u64) {
        if let Some(r) = self.remote.as_mut() {
            r.mirror(Request::DraftRelease { seq });
        }
    }

    /// Rank 1 of a split model: a step of `groups` whose first stage starts from `rows`, the
    /// residual stream rank 0's last layer wrote for every token of the step (in the order of
    /// groups, sequences, tokens).
    pub fn step_from(
        &mut self,
        groups: &[Group<'_>],
        rows: Vec<f32>,
    ) -> Result<StepOutput, ModelError> {
        let hp = &self.model.hp;
        let width = (hp.n_embd * hp.arch.residual_streams()) as usize;
        // Rows as the graphs hold them (a group's sequences padded to its longest), split by
        // the micro-chunks the step runs as, which the first stage takes in order.
        let micros = self.micro_chunks(groups);
        let sizes: Vec<usize> = micros
            .iter()
            .map(|m| {
                batch_shape(m, None, Some(&*hp.arch))
                    .iter()
                    .map(|g| g.tokens())
                    .sum::<usize>()
                    * width
            })
            .collect();
        let total: usize = sizes.iter().sum();
        if rows.len() != total || self.stages[0].layers.start == 0 {
            return Err(ModelError::Node(format!(
                "a step came with {} residual values for {total} wanted ({width} per row) by \
                 stages from layer {}",
                rows.len(),
                self.stages[0].layers.start
            )));
        }
        let mut queue = VecDeque::with_capacity(micros.len());
        if micros.len() == 1 {
            queue.push_back(rows);
        } else {
            let mut at = 0;
            for k in sizes {
                queue.push_back(rows[at..at + k].to_vec());
                at += k;
            }
        }
        *self.ext_rows.lock().expect("rows lock") = queue;
        let out = self.step(groups);
        self.ext_rows.lock().expect("rows lock").clear();
        out
    }

    /// The next micro-chunk's residual that came over the link (rank 1's first stage).
    pub(super) fn take_ext_rows(&self) -> Option<Vec<f32>> {
        self.ext_rows.lock().expect("rows lock").pop_front()
    }
}

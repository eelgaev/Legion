//! `--long-dataflow-async` (#807): a stage's residual goes to the next GPU without the host
//! waiting for it, and a stage that reads nothing back queues its next micro-chunk while its
//! graph still runs.
//!
//! Without it, a stage thread runs its graph to completion, then copies the residual into the
//! next stage's mailbox with a synchronous `cudaMemcpy` (staged through host memory without
//! peer access: 7-15 ms per 3072-token chunk on an AC922 with 4 GPUs), and only then reports the graph done;
//! the host thread then fills, hands over and launches the stage's next graph while its GPU
//! idles. #804 measured both inside every stage's cycle (354-359 ms against graph spans of
//! 332-339 ms).
//!
//! Here, per stage before the last, after its thread has queued the graph on the stage's
//! stream:
//! - the stage's stream copies the residual into an output buffer of its own (one residual,
//!   on its GPU, ~0.3 ms), once the previous micro-chunk's residual has left that buffer, and
//!   records `graph` (the graph and that copy are done);
//! - a copy stream of the stage waits for `graph` and for the next stage to have taken what
//!   the mailbox entry held two micro-chunks before (`taken`), copies the output buffer into
//!   the entry and records `sent`;
//! - the thread reports the graph *queued*, and a waiter thread reports it *done* once
//!   `graph` is.
//!
//! The next stage takes a residual once it is queued: its stream waits for `sent`, copies the
//! mailbox entry into its graph's input and records `taken`; its graph then runs behind that
//! on the same stream. No host wait anywhere on the way.
//!
//! The graph's own output tensor and the output buffer make the double buffer: the stage's
//! next graph may overwrite the first while the second is still being copied out.
//!
//! The kernels, their order within each graph and the bytes moved are the same as without the
//! flag; only when the copies run and who waits for them changes. An event is recorded again
//! for a later micro-chunk only after every wait on its earlier use was queued (the host's
//! order, `next_work`): `graph` of micro-chunk `m + 2` after `m` was reported done, `sent` of
//! `m + 2` after the next stage queued its take of `m`, `taken` of `m + 2` after the stage
//! before queued its copy of `m + 2` (which waited on the `taken` of `m`).

// Without CUDA there are no links to make (`Executor::links`).
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::residency::cuda::{Copier, Event, copy_async};

/// Mailbox entries (and events) per stage: micro-chunk `m` uses entry `m % 2`.
pub(super) const TURNS: usize = 2;

/// A stage before the last: its output buffer, its events and its copy stream.
#[cfg(feature = "cuda")]
struct Producer {
    /// Its copy stream (only the stage's thread queues on it).
    copier: std::sync::Mutex<Copier>,
    /// The output buffer, a device address on the stage's GPU.
    out: usize,
    /// Per turn: the graph and the copy into `out` are done.
    graph: [Event; TURNS],
    /// Per turn: the residual is in the next stage's mailbox entry.
    sent: [Event; TURNS],
}

/// A step's async hand-off: per stage its CUDA device, per stage before the last a
/// [`Producer`], per stage after the first its `taken` events. The output buffers and the
/// mailboxes must outlive it: its copiers wait for their streams as they drop.
pub(super) struct Links {
    #[cfg(feature = "cuda")]
    prod: Vec<Option<Producer>>,
    #[cfg(feature = "cuda")]
    taken: Vec<Option<[Event; TURNS]>>,
    #[cfg(feature = "cuda")]
    dev: Vec<usize>,
}

impl Links {
    /// For stages on CUDA devices `dev`, with output buffers `out` (per stage before the last,
    /// a device address of at least a residual's bytes, kept by the caller).
    #[cfg(feature = "cuda")]
    pub fn new(dev: Vec<usize>, out: Vec<usize>) -> Result<Self, String> {
        let n = dev.len();
        let events =
            |d: usize| -> Result<[Event; TURNS], String> { Ok([Event::new(d)?, Event::new(d)?]) };
        let mut prod = Vec::with_capacity(n);
        for (s, &d) in dev.iter().enumerate() {
            prod.push(match out.get(s) {
                Some(&o) if s + 1 < n => Some(Producer {
                    copier: std::sync::Mutex::new(Copier::new(d)?),
                    out: o,
                    graph: events(d)?,
                    sent: events(d)?,
                }),
                _ => None,
            });
        }
        let taken = (0..n)
            .map(|s| (s > 0).then(|| events(dev[s])).transpose())
            .collect::<Result<_, _>>()?;
        Ok(Self { prod, taken, dev })
    }

    /// Queues stage `s`'s take of micro-chunk `m`'s residual on `stream` (the stage's): after
    /// the stage before's copy into mailbox entry `src`, copies `bytes` into `dst` (the graph's
    /// input) and marks the entry taken.
    ///
    /// # Safety
    /// `dst` is the input of the stage's graph queued next on `stream`, `src` the mailbox
    /// entry of `m`, whose copy the stage before has queued (`next_work`).
    #[cfg(feature = "cuda")]
    pub unsafe fn take(
        &self,
        s: usize,
        m: usize,
        dst: usize,
        src: usize,
        bytes: usize,
        stream: *mut std::ffi::c_void,
    ) -> Result<(), String> {
        let t = m % TURNS;
        let up = self.prod[s - 1]
            .as_ref()
            .ok_or("no stage before to take from")?;
        let taken = self.taken[s]
            .as_ref()
            .ok_or("a first stage takes nothing")?;
        // SAFETY: live events and stream; the caller's ranges.
        unsafe {
            up.sent[t].wait_on(stream)?;
            copy_async(self.dev[s], dst, src, bytes, stream)?;
            taken[t].record(stream)
        }
    }

    /// Whether stage `s` hands its residual over through these links.
    pub fn sends(&self, s: usize) -> bool {
        #[cfg(feature = "cuda")]
        return self.prod.get(s).is_some_and(Option::is_some);
        #[cfg(not(feature = "cuda"))]
        {
            let _ = s;
            false
        }
    }
}

/// Stage `s`'s hand-off of micro-chunk `m` for its thread: its residual at `src`, `bytes`,
/// into the next stage's mailbox entry `dst`.
pub(super) struct Out {
    pub links: Arc<Links>,
    pub s: usize,
    pub m: usize,
    pub src: usize,
    pub dst: usize,
    pub bytes: usize,
}

impl Out {
    /// After the graph was queued on `stream` (the stage's): queues the copy into the output
    /// buffer, records `graph`, and queues the copy into the mailbox on the copy stream (see
    /// the module docs).
    ///
    /// # Safety
    /// `src` is the residual output of the graph just queued on `stream`, `dst` the mailbox
    /// entry of micro-chunk `m`, which the next stage's take of `m - 2` was queued for.
    #[cfg(feature = "cuda")]
    pub unsafe fn queue(&self, stream: *mut std::ffi::c_void) -> Result<(), String> {
        let t = self.m % TURNS;
        let l = &self.links;
        let p = l.prod[self.s]
            .as_ref()
            .ok_or("a last stage hands nothing over")?;
        // SAFETY: live events, streams and buffers (the links live until the step's graphs are
        // done); the caller's ranges.
        unsafe {
            // The previous micro-chunk's residual out of the output buffer first.
            p.sent[(self.m + TURNS - 1) % TURNS].wait_on(stream)?;
            copy_async(l.dev[self.s], p.out, self.src, self.bytes, stream)?;
            p.graph[t].record(stream)?;
            let mut c = p.copier.lock().unwrap_or_else(|e| e.into_inner());
            c.wait(&p.graph[t])?;
            if let Some(taken) = l.taken[self.s + 1].as_ref() {
                c.wait(&taken[t])?;
            }
            c.copy(self.dst, p.out, self.bytes)?;
            c.record(&p.sent[t])
        }
    }

    /// Blocks until the graph and the copy into the output buffer are done.
    #[cfg(feature = "cuda")]
    pub fn wait(&self) -> Result<(), String> {
        let p = self.links.prod[self.s]
            .as_ref()
            .ok_or("a last stage hands nothing over")?;
        p.graph[self.m % TURNS].synchronize()
    }
}

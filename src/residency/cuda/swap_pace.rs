//! The expert cache's swap copies paced through the decode step (#37), with the CPU expert
//! share on and every expert in host memory (no bounded host tier).
//!
//! The after-step plans up to `--expert-swaps` copies (about 21 a step at 128 GB on a gaming
//! PC, 3 MB each). Queued at once, they all run in the next step's first layers, beside those
//! layers' pool jobs (which read the same DRAM) and in-place misses (which cross the same
//! link). Paced, they wait in a queue that the share's down op drains a few at a time: the op
//! records an event after its layer's `collect` and the copier's stream waits for it before
//! the layer's copies, so each copy starts once the layer's pool job is done and runs in the
//! rest of that layer and the next layer's attention. The copies are spread over the first
//! [`SPREAD_LAYERS`] share layers, so they land before the step ends.
//!
//! What a step reads does not change: a copied slot is used only after the after-step that
//! finds its copy landed ([`CopyEvent`]), and the slot is out of the tables until then, as
//! without the pacing. What is left in the queue when the step had no share layer (a prompt)
//! goes out at the next after-step or copy wait ([`swap_pace_flush`]), so nothing that waits for the
//! copier (a lend, the shutdown) finds a copy still queued here.
//! `LEGION_DEBUG_SWAP_PACE=0` queues the copies at once, as before.

use std::collections::VecDeque;

use super::*;

/// The share layers a step's swap copies are spread over (of 48 in Qwen3.8-Flash-Next).
pub const SPREAD_LAYERS: usize = 32;

/// One queued copy.
struct Piece {
    dst: usize,
    src: usize,
    bytes: usize,
    /// The last of its swap's pieces (a swap is copied whole in one layer).
    last: bool,
}

/// A device's pacer.
struct Pacer {
    /// The cache's copier stream (its [`Copier`]); `None` until the cache queues.
    stream: Option<cudaStream_t>,
    gate: Option<Event>,
    queue: VecDeque<Piece>,
    /// Swaps still queued and share layers drained since the batch was queued.
    swaps: usize,
    layers: usize,
    /// Recorded after the batch's last copy, once it is queued on the stream.
    done: Option<CopyEvent>,
    /// The batch has copies that went to the stream (or are queued here).
    pending: bool,
    /// Swaps the share layers started, and swaps [`swap_pace_flush`] sent (debug statistics).
    drained: u64,
    flushed: u64,
}

// SAFETY: a stream handle and plain addresses, used under the registry's lock.
unsafe impl Send for Pacer {}

/// Per CUDA device, its pacer (only devices the share runs on, without the host tier).
static PACERS: Mutex<Vec<Option<Pacer>>> = Mutex::new(Vec::new());

/// Whether pacing is on (`LEGION_DEBUG_SWAP_PACE=0` turns it off).
fn wanted() -> bool {
    std::env::var("LEGION_DEBUG_SWAP_PACE").map_or(true, |v| v != "0")
}

/// The share's down ops on CUDA device `device` drain its swap copies from now on (the share
/// set up for it, no bounded host tier). Off with `LEGION_DEBUG_SWAP_PACE=0`.
pub fn swap_pace_enable(device: usize) {
    if !wanted() {
        return;
    }
    let mut p = PACERS.lock().unwrap();
    if p.len() <= device {
        p.resize_with(device + 1, || None);
    }
    p[device] = Some(Pacer {
        stream: None,
        gate: None,
        queue: VecDeque::new(),
        swaps: 0,
        layers: 0,
        done: None,
        pending: false,
        drained: 0,
        flushed: 0,
    });
    tracing::debug!(
        device,
        "expert cache: swap copies paced by the share's layers"
    );
}

/// Whether `device`'s swap copies are paced ([`swap_pace_enable`]).
pub fn swap_pace_on(device: usize) -> bool {
    PACERS
        .lock()
        .unwrap()
        .get(device)
        .is_some_and(Option::is_some)
}

/// Forgets every pacer (the residency that set them up is gone; its copier streams too).
/// The caller flushed them first.
pub fn swap_pace_clear() {
    for (device, p) in PACERS.lock().unwrap().drain(..).enumerate() {
        if let Some(p) = p {
            tracing::debug!(
                target: "legion::residency::cpu_share",
                device,
                drained = p.drained,
                flushed = p.flushed,
                "expert cache: paced swaps started by the share's layers and at the after-step"
            );
        }
    }
}

/// Issues `p`'s queued pieces up to `swaps` whole swaps (all: `usize::MAX`) on its stream;
/// records the batch's done event once the queue is empty.
fn issue(p: &mut Pacer, device: usize, mut swaps: usize) -> Result<(), String> {
    let Some(stream) = p.stream else {
        return Ok(());
    };
    while swaps > 0
        && let Some(c) = p.queue.pop_front()
    {
        // SAFETY: the cache's ranges (a slot out of the tables and its expert's host bytes,
        // [`Copier::copy`]'s terms); the copier's live stream.
        unsafe {
            cudaSetDevice(device as c_int);
            rt(
                cudaMemcpyAsync(
                    c.dst as *mut c_void,
                    c.src as *const c_void,
                    c.bytes,
                    CUDA_MEMCPY_DEFAULT,
                    stream,
                ),
                "cudaMemcpyAsync (paced swap)",
            )?;
        }
        if c.last {
            swaps -= 1;
            p.swaps = p.swaps.saturating_sub(1);
        }
    }
    if p.queue.is_empty() && p.pending && p.done.is_none() {
        let mut e = null_mut();
        // SAFETY: creates and records an event on the copier's stream.
        unsafe {
            cudaSetDevice(device as c_int);
            rt(
                cudaEventCreateWithFlags(&mut e, CUDA_EVENT_DISABLE_TIMING),
                "cudaEventCreate",
            )?;
            let ev = CopyEvent(e);
            rt(cudaEventRecord(e, stream), "cudaEventRecord")?;
            p.done = Some(ev);
        }
    }
    Ok(())
}

/// Queues a batch of swap copies `(dst, src, bytes)` for `device`'s share layers to drain,
/// `per_swap` pieces a swap, on `copier`'s stream. The previous batch must have been
/// [`swap_pace_flush`]ed.
pub fn swap_pace_queue(
    device: usize,
    copier: &Copier,
    pieces: &[(usize, usize, usize)],
    per_swap: usize,
) -> Result<(), String> {
    let mut all = PACERS.lock().unwrap();
    let p = all
        .get_mut(device)
        .and_then(Option::as_mut)
        .ok_or("swap pacing: device not enabled")?;
    if p.pending {
        return Err("swap pacing: the previous batch was not flushed".into());
    }
    p.stream = Some(copier.stream);
    for (i, &(dst, src, bytes)) in pieces.iter().enumerate() {
        p.queue.push_back(Piece {
            dst,
            src,
            bytes,
            last: (i + 1) % per_swap.max(1) == 0 || i + 1 == pieces.len(),
        });
    }
    p.swaps = pieces.len().div_ceil(per_swap.max(1));
    p.layers = 0;
    p.pending = !pieces.is_empty();
    Ok(())
}

/// The share's down op of a layer on `device`, on its compute `stream` after `collect`: the
/// layer's share of the queued copies starts on the copier's stream once the stream gets
/// here. A no-op without a pacer or a queue.
///
/// # Safety
/// `stream` must be a live stream of `device`, the current device.
pub unsafe fn swap_pace_layer(device: usize, stream: *mut c_void) -> Result<(), String> {
    let mut all = PACERS.lock().unwrap();
    let Some(p) = all.get_mut(device).and_then(Option::as_mut) else {
        return Ok(());
    };
    if p.queue.is_empty() {
        return Ok(());
    }
    let Some(copier) = p.stream else {
        return Ok(());
    };
    // This layer's share: the swaps left over the layers left of the spread.
    let left = SPREAD_LAYERS.saturating_sub(p.layers).max(1);
    let take = p.swaps.div_ceil(left);
    p.layers += 1;
    if take == 0 {
        return Ok(());
    }
    p.drained += take.min(p.swaps) as u64;
    if p.gate.is_none() {
        p.gate = Some(Event::new(device)?);
    }
    let gate = p.gate.as_ref().unwrap();
    // SAFETY: the caller's compute stream and the copier's live stream, both on `device`.
    unsafe {
        gate.record(stream)?;
        gate.wait_on(copier)?;
    }
    issue(p, device, take)
}

/// Sends what is still queued for `device` to the copier's stream and hands over the batch's
/// done event (`None`: no batch since the last flush).
pub fn swap_pace_flush(device: usize) -> Result<Option<CopyEvent>, String> {
    let mut all = PACERS.lock().unwrap();
    let Some(p) = all.get_mut(device).and_then(Option::as_mut) else {
        return Ok(None);
    };
    if !p.pending {
        return Ok(None);
    }
    p.flushed += p.swaps as u64;
    issue(p, device, usize::MAX)?;
    p.pending = false;
    p.swaps = 0;
    Ok(p.done.take())
}

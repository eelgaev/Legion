//! The expert cache's work after a decode or verify step, split by pipeline stage (#267), so
//! no part of it runs with every GPU waiting.
//!
//! After a step the cache reads the step's routing, plans swaps and writes its slot tables
//! (`ExpertResidency::after_step`): its host-side plan for the next steps, about 1 ms of host
//! time per step on Qwen3.8-Flash-Next over two V100s. Without MTP drafting that ran after the
//! step's last stage, before the next step, with both GPUs idle (~4.5% of a 24 ms token at one
//! sequence). With the work split per stage ([`stage_split`]):
//!
//! - a stage's share runs once the stage is done, while the next stage computes the step (the
//!   share of stage `s` in round `s + 1`);
//! - the last stage's share runs while the next step's first stage computes, as
//!   `Executor::deferred_tail`.
//!
//! With MTP drafting a step's work already runs beside the next MTP pass (`mtp_window`), so
//! steps of a model with an MTP head keep that order.
//!
//! A share touches only its stage's layers (`after_step_more` / `after_step_only`), and a pass
//! never reads a layer whose share is running: the layers of a done stage until the next step,
//! the last stage's layers while the next step's first stage computes. The cache plans swaps
//! per device; a split that keeps every cache device in one stage plans the swaps one call over
//! all the layers would. Where the work runs changes when copies land in their slots, not what
//! any step computes: a slot and the expert's host bytes are the same bytes.
//!
//! Strata-AC922 (eelgaev/Strata-AC922, `ac922-gpu-plan-wip` 9263a34, MIT) moved its per-layer
//! expert plan to the GPU to drop a host round trip per layer. Legion's per-step plan is
//! already on the GPU (each routed expert maps through the device's table to a slot or its host
//! bytes inside the graph); what remained of a round trip is this per-step work, which keeps
//! its host-side planning (swaps are DMA copies the host queues) and moves off the critical
//! path instead.

use std::ops::Range;

#[cfg(test)]
mod tests;

/// Per stage, the target layers (`< n_layer`) whose share of a step's cache work runs once the
/// stage is done, out of `stages`' layer ranges; `None` (no split) with fewer than two stages,
/// stages that are not back to back from layer 0, a stage without target layers, no tiered
/// layer at all (`device_of`: a layer's cache device, `None` for untiered layers), or a cache
/// device with layers in two stages (two calls would plan its swaps twice).
pub(super) fn stage_split(
    stages: &[Range<usize>],
    n_layer: usize,
    device_of: impl Fn(usize) -> Option<usize>,
) -> Option<Vec<Range<usize>>> {
    if stages.len() < 2 || stages[0].start != 0 {
        return None;
    }
    let mut out: Vec<Range<usize>> = Vec::with_capacity(stages.len());
    let mut seen: Vec<(usize, usize)> = Vec::new();
    let mut tiered = false;
    for (s, st) in stages.iter().enumerate() {
        if out.last().is_some_and(|p| p.end != st.start) {
            return None;
        }
        let r = st.start..st.end.min(n_layer);
        if r.is_empty() {
            return None;
        }
        for d in r.clone().filter_map(&device_of) {
            tiered = true;
            match seen.iter().find(|(dev, _)| *dev == d) {
                Some(&(_, owner)) if owner != s => return None,
                Some(_) => {}
                None => seen.push((d, s)),
            }
        }
        out.push(r);
    }
    tiered.then_some(out)
}

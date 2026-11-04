//! The host time from an MTP pass's picks to the next step's first stage launched (#814): at
//! one sequence the GPUs idle through it. Logged at debug level per step
//! (`RUST_LOG=legion::model::core::exec::step_front=debug`) as microseconds by phase; a few
//! `Instant::now` calls per step then, none otherwise.

use std::time::Instant;

#[cfg(test)]
mod tests;

/// When the last MTP pass's picks reached the host and when its call returned.
#[derive(Default)]
pub(super) struct FrontClock {
    on: Option<bool>,
    picked: Option<Instant>,
    returned: Option<Instant>,
}

impl FrontClock {
    fn on(&mut self) -> bool {
        *self
            .on
            .get_or_insert_with(|| tracing::enabled!(tracing::Level::DEBUG))
    }

    /// An MTP pass's picks are in (its wait ended).
    pub(super) fn picked(&mut self) {
        if self.on() {
            self.picked = Some(Instant::now());
        }
    }

    /// The MTP pass's call returns.
    pub(super) fn returned(&mut self) {
        if self.on() {
            self.returned = Some(Instant::now());
        }
    }

    /// A step starts: its front's laps, measured from now (none when not logging).
    pub(super) fn step(&mut self) -> Front {
        let (picked, returned) = (self.picked.take(), self.returned.take());
        if !self.on() {
            return Front::default();
        }
        let now = Instant::now();
        Front {
            at: Some(now),
            picked,
            returned,
            start: Some(now),
            laps: Vec::with_capacity(LAPS),
        }
    }
}

const LAPS: usize = 12;

/// One step's front: the laps up to its first stage's launch.
#[derive(Default)]
pub(super) struct Front {
    at: Option<Instant>,
    picked: Option<Instant>,
    returned: Option<Instant>,
    start: Option<Instant>,
    laps: Vec<(&'static str, u32)>,
}

impl Front {
    /// Ends the lap `name` now.
    pub(super) fn lap(&mut self, name: &'static str) {
        if let Some(at) = self.at.as_mut() {
            let now = Instant::now();
            self.laps.push((name, us(now - *at)));
            *at = now;
        }
    }

    /// The first stage is launched: logs the laps (once).
    pub(super) fn done(&mut self) {
        let Some(start) = self.start.take() else {
            return;
        };
        self.lap("launch");
        self.at = None;
        let tail = match (self.picked, self.returned) {
            (Some(p), Some(r)) => us(r.saturating_duration_since(p)),
            _ => 0,
        };
        let gap = self
            .returned
            .map_or(0, |r| us(start.saturating_duration_since(r)));
        tracing::debug!(
            pass_tail = tail,
            between = gap,
            laps = ?self.laps,
            "step front us"
        );
    }
}

fn us(d: std::time::Duration) -> u32 {
    d.as_micros().min(u32::MAX as u128) as u32
}

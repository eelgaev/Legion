//! The host's record of deferred commits (#836 part 14): per slot, how many stashed tokens its
//! state still has to advance by, where they start in the packed replay rows, and what the
//! device's copy of those counts holds.
//!
//! The device copy (one I32 [`count_word`] per slot) is what the next verify step's delta rule
//! reads ([`super::super::gdn::rows_replay`]) and what a flush replays. It is uploaded only when a
//! reader needs it to differ from what it holds: after a verify step consumed its slots, the
//! device still holds their counts until the next commit or flush uploads new ones, and a verify
//! of a slot whose device count is stale uploads first.
//!
//! A pool that lets verify steps leave their states (#836, [`super::super::gdn::leaves`]) keeps a
//! slot's kept tokens waiting over more than one round: a verify step whose tokens fit the packed
//! rows after the kept ones replays those in registers only, does not write the state, and
//! stores its own tokens after them; the commit then extends the waiting run by the tokens it
//! keeps. Every rule here mirrors the kernel's, so host and device agree on where each run is.

use std::ops::Range;

use super::super::gdn::{count_word, leaves};

pub(super) struct Pending {
    /// Per slot: the kept tokens its state has not advanced by yet (0: none).
    keep: Vec<usize>,
    /// Per slot: the packed row those tokens start at.
    start: Vec<usize>,
    /// Per slot: the row the last verify step stored its tokens from (where a commit's kept
    /// tokens are).
    at: Vec<usize>,
    /// Per slot: no commit since its last verify step (or flush, or reset), so a commit may be
    /// deferred on top.
    open: Vec<bool>,
    /// What the device's counts hold (as last uploaded).
    dev: Vec<i32>,
    /// The packed rows per slot.
    room: usize,
    /// Verify steps may leave the states where the rows have room.
    leave: bool,
    /// Per slot: whether its last verify step had at least [`Self::min_seqs`] sequences, so its
    /// count word carries the leave bit (#858: the next step is taken to be as wide).
    wide: Vec<bool>,
    /// The fewest sequences a verify step leaves the states in (`--gdn-state-defer-min-seqs`;
    /// narrower steps write them, as without `leave`).
    min_seqs: usize,
}

impl Pending {
    /// A record whose verify steps always write the states (the counts are plain counts).
    #[cfg(test)]
    pub(super) fn new(n_slots: usize) -> Self {
        Self::with_room(n_slots, usize::MAX, false)
    }

    /// A record of `n_slots` slots with `room` packed rows each; `leave`: verify steps may leave
    /// the states as they are where those rows have room ([`leaves`]).
    pub(super) fn with_room(n_slots: usize, room: usize, leave: bool) -> Self {
        Self {
            keep: vec![0; n_slots],
            start: vec![0; n_slots],
            at: vec![0; n_slots],
            open: vec![true; n_slots],
            dev: vec![0; n_slots],
            room,
            leave,
            wide: vec![true; n_slots],
            min_seqs: 0,
        }
    }

    /// Verify steps of fewer than `min_seqs` sequences write the states even where they could
    /// leave them (#858): a slot's count word carries the leave bit only while its last verify
    /// step was at least that wide, and a verify step of another width uploads the words first.
    pub(super) fn leave_from(mut self, min_seqs: usize) -> Self {
        self.min_seqs = min_seqs;
        self
    }

    /// Slot `s`'s count word as the device should hold it.
    fn word(&self, s: usize) -> i32 {
        if self.keep[s] == 0 {
            0
        } else {
            count_word(self.keep[s], self.start[s], self.leave && self.wide[s])
        }
    }

    fn words(&self) -> Vec<i32> {
        (0..self.keep.len()).map(|s| self.word(s)).collect()
    }

    /// Whether any slot's state waits for a deferred commit.
    pub(super) fn any(&self) -> bool {
        self.keep.iter().any(|&k| k > 0)
    }

    /// Whether a slot in `slots` waits for a deferred commit.
    pub(super) fn touches(&self, slots: Range<usize>) -> bool {
        let end = slots.end.min(self.keep.len());
        self.keep[slots.start.min(end)..end].iter().any(|&k| k > 0)
    }

    /// Whether every slot of `commits` had a verify step since its last commit, so the commit
    /// may be deferred on top.
    pub(super) fn can_commit(&self, commits: &[(usize, usize)]) -> bool {
        commits.iter().all(|&(slot, _)| self.open[slot])
    }

    /// A deferred commit of `commits` (`(slot, keep)`, see [`Self::can_commit`]): the count words
    /// to upload to the device now. Each slot's kept tokens are where its last verify step stored
    /// them, right after the tokens it still waits for.
    pub(super) fn commit(&mut self, commits: &[(usize, usize)]) -> Vec<i32> {
        for &(slot, keep) in commits {
            assert!(self.open[slot], "slot {slot} committed twice");
            if self.keep[slot] == 0 {
                self.start[slot] = self.at[slot];
            }
            assert_eq!(
                self.start[slot] + self.keep[slot],
                self.at[slot],
                "slot {slot}: kept tokens apart from the waiting ones"
            );
            self.keep[slot] += keep;
            assert!(
                self.start[slot] + self.keep[slot] <= self.room,
                "slot {slot}: kept tokens past the packed rows"
            );
            self.open[slot] = false;
        }
        let words = self.words();
        self.dev.clone_from(&words);
        words
    }

    /// A verify step of `n_tok` tokens over `slots`, which replays their deferred commits: the
    /// count words to upload first if the device's copy of those slots is not the host's, `None`
    /// if it is. The slots count as advanced from then on, or, where the step leaves a state
    /// ([`leaves`]), as waiting for the same tokens with the step's own stored after them.
    pub(super) fn verify(&mut self, slots: Range<usize>, n_tok: usize) -> Option<Vec<i32>> {
        let wide = slots.len() >= self.min_seqs;
        for s in slots.clone() {
            self.wide[s] = wide;
        }
        let upload = slots.clone().any(|s| self.word(s) != self.dev[s]).then(|| {
            let words = self.words();
            self.dev.clone_from(&words);
            words
        });
        for s in slots {
            if leaves(self.word(s), n_tok, self.room) {
                self.at[s] = self.start[s] + self.keep[s];
            } else {
                self.keep[s] = 0;
                self.start[s] = 0;
                self.at[s] = 0;
            }
            self.open[s] = true;
        }
        upload
    }

    /// A flush (every deferred commit replayed on its own): the count words to upload first and
    /// replay, `None` if nothing waits. Nothing waits afterwards.
    pub(super) fn flush(&mut self) -> Option<Vec<i32>> {
        if !self.any() {
            return None;
        }
        let up = self.words();
        self.dev.clone_from(&up);
        self.keep.fill(0);
        self.start.fill(0);
        self.open.fill(true);
        Some(up)
    }

    /// Slot `slot`'s state was reset: its deferred commit is void.
    pub(super) fn drop_slot(&mut self, slot: usize) {
        self.keep[slot] = 0;
        self.start[slot] = 0;
        self.at[slot] = 0;
        self.open[slot] = true;
    }
}

#[cfg(test)]
mod tests;

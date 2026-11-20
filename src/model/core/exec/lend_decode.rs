//! Kept decode graphs across the expert cache's lend kinds (#37). A decode step's graph is kept
//! by its shape alone, through lends and restores: a step of fewer tokens than a lend stages
//! from ([`ExpertResidency::stage_from`]) builds the same ops lent or not. A bigger one built in
//! a lend waits for and reads the lend's staged copies, and one built outside it does not, so
//! such graphs go whenever the lowest staging size changes (a short prompt's lend starts or
//! ends, [`super::Executor::set_short_lend`]) and when the cache is restored (its staging is
//! released). The next decode step of that shape builds its graph again.
//!
//! [`ExpertResidency::stage_from`]: crate::residency::experts::ExpertResidency::stage_from

use crate::model::core::GroupShape;

#[cfg(test)]
mod tests;

/// Whether a step of `shape` has as many tokens as a lend that stages from `floor` tokens
/// stages, so its graph may wait for and read staged copies.
pub(super) fn may_stage(shape: &[GroupShape], floor: i64) -> bool {
    shape.iter().map(GroupShape::tokens).sum::<usize>() as i64 >= floor
}

/// Takes the graphs of `kept` that [`may_stage`] out (in their order), leaving the others in
/// theirs.
pub(super) fn take_staging<T>(
    kept: &mut Vec<(Vec<GroupShape>, T)>,
    floor: i64,
) -> Vec<(Vec<GroupShape>, T)> {
    let (gone, stay) = std::mem::take(kept)
        .into_iter()
        .partition(|(shape, _)| may_stage(shape, floor));
    *kept = stay;
    gone
}

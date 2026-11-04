use super::*;
use crate::model::core::Logits;

fn seqs<'b>(tokens: &'b [u32], n_past: &[usize]) -> Vec<SeqInput<'b>> {
    n_past
        .iter()
        .map(|&n_past| SeqInput {
            tokens,
            n_past,
            blocks: &[],
            ring: 0,
            logits: Logits::All,
            media: None,
        })
        .collect()
}

fn group<'b>(tokens: &'b [u32], n_past: &[usize], verify: bool) -> Group<'b> {
    Group {
        slot: 0,
        seqs: seqs(tokens, n_past),
        verify,
    }
}

fn shape(n_pool: usize, n_new: usize) -> GroupShape {
    GroupShape {
        slot: 0,
        n_seqs: 16,
        n_tok: 9,
        n_kv: 0,
        table: 2048,
        ring: 0,
        logits: vec![9; 16],
        last_only: false,
        verify: true,
        n_pool,
        n_new,
        pick_all: false,
        kv_slots: Vec::new(),
        par: false,
        pair: false,
        img: 0,
        captures: Vec::new(),
    }
}

#[test]
fn a_verify_step_advances_each_sequence_by_its_tokens() {
    let nine = [0u32; 9];
    let one = [0u32; 1];
    let groups = [group(&nine, &[100, 2000], true), group(&one, &[50], false)];
    let next = advanced(&groups, None).expect("a verify step");
    let past: Vec<Vec<usize>> = next
        .iter()
        .map(|g| g.seqs.iter().map(|s| s.n_past).collect())
        .collect();
    assert_eq!(past, [vec![109, 2009], vec![51]]);
    assert!(next[0].verify && !next[1].verify);
    assert_eq!(next[0].seqs[0].tokens.len(), 9);
}

#[test]
fn only_verify_steps_that_still_fit_their_table_advance() {
    let nine = [0u32; 9];
    let one = [0u32; 1];
    let prompt = [0u32; 512];
    // A decode step, a prompt chunk beside a verify group: not verify steps.
    assert!(advanced(&[group(&one, &[10], false)], None).is_none());
    assert!(
        advanced(
            &[group(&nine, &[10], true), group(&prompt, &[0], false)],
            None
        )
        .is_none()
    );
    // Two blocks of 64 positions hold 128: 110 + 9 + 9 fits, 111 + 9 + 9 does not.
    assert!(advanced(&[group(&nine, &[110], true)], Some(2)).is_some());
    assert!(advanced(&[group(&nine, &[111], true)], Some(2)).is_none());
}

#[test]
fn only_a_move_of_the_pools_is_built_ahead() {
    let cur = [shape(1024, 48)];
    assert!(pools_moved(&cur, &[shape(1088, 48)]));
    let mut all = shape(1024, 48);
    all.pick_all = true;
    assert!(pools_moved(&cur, &[all]));
    // The same shape, another shape, no sparse attention: nothing.
    assert!(!pools_moved(&cur, &cur));
    let mut other = shape(1088, 48);
    other.n_seqs = 15;
    assert!(!pools_moved(&cur, &[other]));
    assert!(!pools_moved(&[shape(0, 0)], &[shape(0, 0)]));
    assert!(!pools_moved(&cur, &[shape(1088, 48), shape(64, 3)]));
}

/// With qwen4exp's 64-padded listing, the advanced step names the next padding step when the
/// sequences are within one step of it.
#[test]
fn the_next_listing_shows_near_the_padding_boundary() {
    let kpool = 4;
    let nine = [0u32; 9];
    let listed = |g: &Group<'_>| crate::model::qwen4::pools_listed(g, kpool);
    // 16 sequences at 4,000 positions: (4009 / 4) * 16 = 16,032 pools, listed as 16,064.
    let near = group(&nine, &[4000; 16], true);
    let next = advanced(std::slice::from_ref(&near), None).expect("a verify step");
    assert_eq!(listed(&near), 16_064);
    assert_eq!(listed(&next[0]), 16_064 + 64);
}

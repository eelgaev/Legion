use super::*;

#[test]
fn copies_apart_keep_their_order() {
    assert_eq!(
        copy_order(&[(0..4, 8..12), (16..20, 24..28)]),
        Some(vec![0, 1])
    );
}

#[test]
fn a_copy_onto_itself_is_left_out() {
    assert_eq!(copy_order(&[(0..4, 0..4), (16..20, 24..28)]), Some(vec![1]));
    assert_eq!(copy_order(&[(0..4, 0..4)]), Some(vec![]));
}

#[test]
fn a_copy_whose_source_another_overwrites_goes_first() {
    // Copy 0 writes over copy 1's source: copy 1 first.
    assert_eq!(
        copy_order(&[(0..4, 16..20), (16..20, 32..36)]),
        Some(vec![1, 0])
    );
    assert_eq!(
        copy_order(&[(16..20, 32..36), (0..4, 16..20)]),
        Some(vec![0, 1])
    );
}

#[test]
fn copies_overwriting_each_others_sources_have_no_order() {
    assert_eq!(copy_order(&[(0..4, 16..20), (16..20, 0..4)]), None);
}

#[test]
fn a_copy_overlapping_itself_or_another_destination_is_refused() {
    assert_eq!(copy_order(&[(0..8, 4..12)]), None);
    assert_eq!(copy_order(&[(0..4, 16..24), (32..36, 20..28)]), None);
    // A copy left out keeps its bytes in place: nothing may write over them.
    assert_eq!(copy_order(&[(0..4, 0..4), (32..36, 2..6)]), None);
}

#[test]
fn the_predicted_pass_feeds_every_row_one_position_on() {
    let a = [3u32, 4];
    let b = [7u32];
    let g = predict([Some((KV_BLOCK, &a[..], 1)), Some((5, &b[..], 0))]).expect("a pass");
    assert_eq!((g.slot, g.verify), (0, false));
    let seqs: Vec<_> = (g.seqs.iter())
        .map(|s| {
            (
                s.n_past,
                s.blocks.as_ptr(),
                s.ring,
                s.tokens.len(),
                s.logits,
            )
        })
        .collect();
    assert_eq!(
        seqs,
        [
            (KV_BLOCK, a.as_ptr(), 1, 1, Logits::Last),
            (5, b.as_ptr(), 0, 1, Logits::Last),
        ]
    );
}

#[test]
fn no_pass_is_predicted_for_a_row_at_its_cap_or_past_its_blocks() {
    let a = [3u32];
    assert!(predict([Some((1, &a[..], 0)), None]).is_none());
    assert!(predict(std::iter::empty()).is_none());
    // The next position is in a block the row does not have.
    assert!(predict([Some((KV_BLOCK, &a[..], 0))]).is_none());
    assert!(predict([Some((KV_BLOCK - 1, &a[..], 0))]).is_some());
}

#[test]
fn keys_tell_passes_apart_but_for_their_tokens() {
    fn seq<'b>(tokens: &'b [u32], n_past: usize, blocks: &'b [u32]) -> SeqInput<'b> {
        SeqInput {
            tokens,
            n_past,
            blocks,
            ring: 0,
            logits: Logits::Last,
            media: None,
        }
    }
    fn group(seqs: Vec<SeqInput<'_>>) -> Vec<GroupKey> {
        key(&[Group {
            slot: 0,
            seqs,
            verify: false,
        }])
    }
    let (blocks, other) = ([1u32, 2], [1u32, 2]);
    let (t1, t2) = ([5u32], [6u32]);
    let k = group(vec![seq(&t1, 9, &blocks)]);
    assert_eq!(k, group(vec![seq(&t2, 9, &blocks)]));
    assert_ne!(k, group(vec![seq(&t1, 10, &blocks)]));
    // Equal block numbers elsewhere: another sequence's list.
    assert_ne!(k, group(vec![seq(&t1, 9, &other)]));
    assert_ne!(k, group(vec![seq(&t1, 9, &blocks), seq(&t1, 9, &blocks)]));
}

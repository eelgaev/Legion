use super::*;

fn seq(tokens: &[u32], n_past: usize, blocks: &[u32]) -> SeqInput<'static> {
    SeqInput {
        tokens: tokens.to_vec().leak(),
        n_past,
        blocks: blocks.to_vec().leak(),
        ring: 0,
        logits: Logits::Last,
        media: None,
    }
}

#[test]
fn owned_groups_view_as_given() {
    let groups = vec![Group {
        slot: 3,
        seqs: vec![seq(&[1, 2], 7, &[0, 4]), seq(&[9], 2, &[5])],
        verify: true,
    }];
    let owned = own(&groups);
    let back = view(&owned);
    assert_eq!(back.len(), 1);
    assert_eq!(back[0].slot, 3);
    assert!(back[0].verify);
    assert_eq!(back[0].seqs.len(), 2);
    assert_eq!(back[0].seqs[0].tokens, &[1, 2]);
    assert_eq!(back[0].seqs[0].n_past, 7);
    assert_eq!(back[0].seqs[0].blocks, &[0, 4]);
    assert_eq!(back[0].seqs[1].tokens, &[9]);
}

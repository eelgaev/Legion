//! Pair parallelism's split of the experts and its expert-cache layout.

use super::*;

#[test]
fn a_pair_splits_the_experts_in_halves() {
    for n_expert in [2u32, 7, 256, 512] {
        for pair in [0usize, 2] {
            let (a, b) = (half(pair, n_expert), half(partner(pair), n_expert));
            assert_eq!(a.start, 0);
            assert_eq!(a.end, b.start, "{n_expert}: contiguous");
            assert_eq!(b.end, n_expert, "{n_expert}: every expert");
        }
    }
    assert_eq!(partner(0), 1);
    assert_eq!(partner(1), 0);
    assert_eq!(partner(2), 3);
    assert_eq!(partner(3), 2);
}

#[test]
fn every_layer_but_the_last_is_paired() {
    // 8 layers over 4 stages of 2.
    let owner = owners(8, |il| il / 2);
    assert_eq!(
        owner,
        vec![
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
            Some(3),
            None
        ]
    );
}

#[test]
fn a_paired_layer_gets_a_copy_on_the_pair() {
    let stage_of = |il: usize| (il / 2).min(3);
    let owner = owners(8, stage_of);
    // 8 layers and the MTP head (entry 8, on the last stage).
    let (ep, cuda) = ep_layers(&owner, 9, 512, stage_of);
    assert_eq!(ep.real.len(), 9 + 7);
    assert_eq!(cuda.len(), ep.real.len());
    for (il, (&real, &c)) in ep.real.iter().zip(&cuda).take(9).enumerate() {
        assert_eq!(real, il);
        assert_eq!(c, stage_of(il));
    }
    // The owner's half on the owner; the last layer and the head serve every expert.
    assert_eq!(ep.owned[0], Some(0..256));
    assert_eq!(ep.owned[2], Some(256..512));
    assert_eq!(ep.owned[4], Some(0..256));
    assert_eq!(ep.owned[7], None);
    assert_eq!(ep.owned[8], None);
    // The copies, in layer order, on the pair with its half.
    for (k, il) in (0..7).enumerate() {
        let e = 9 + k;
        assert_eq!(ep.real[e], il);
        assert_eq!(cuda[e], partner(stage_of(il)));
        assert_eq!(
            ep.owned[e],
            Some(half(partner(stage_of(il)), 512)),
            "layer {il}"
        );
    }
}

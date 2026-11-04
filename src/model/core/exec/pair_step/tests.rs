use super::*;

/// 4 layers and the MTP head (entry 4), then copies of layers 0..3 (entries 5..8).
fn state() -> PairState {
    let copy_of = vec![None, None, None, None, None, Some(0), Some(1), Some(2)];
    PairState {
        on: true,
        fresh: vec![false; copy_of.len()],
        copy_of,
        ..Default::default()
    }
}

#[test]
fn a_copy_learns_with_its_layer_once_a_paired_step_ran_it() {
    let mut st = state();
    let first_half = |il: usize| il < 2;
    // No paired step yet: the copies' logs are stale.
    let ran: Vec<usize> = (0..8).filter(|&e| st.widen(first_half)(e)).collect();
    assert_eq!(ran, vec![0, 1]);
    st.fresh[5] = true;
    st.fresh[7] = true;
    let ran: Vec<usize> = (0..8).filter(|&e| st.widen(first_half)(e)).collect();
    assert_eq!(ran, vec![0, 1, 5]);
    st.learned(first_half);
    assert_eq!(
        st.fresh,
        vec![false, false, false, false, false, false, false, true]
    );
    // Layer 2's copy with layer 2, once.
    let rest = |il: usize| (2..5).contains(&il);
    let ran: Vec<usize> = (0..8).filter(|&e| st.widen(rest)(e)).collect();
    assert_eq!(ran, vec![2, 3, 4, 7]);
    st.learned(rest);
    assert!(st.fresh.iter().all(|f| !f));
}

#[test]
fn without_copies_the_filter_is_unchanged() {
    let mut st = PairState::default();
    let ran = |il: usize| il == 3;
    assert!((0..10).all(|e| st.widen(ran)(e) == ran(e)));
    st.learned(ran);
    assert!(st.fresh.is_empty());
}

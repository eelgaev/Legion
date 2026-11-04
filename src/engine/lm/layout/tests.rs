use super::*;

#[test]
fn layer_names() {
    assert_eq!(layer_of("blk.12.attn_q.weight"), Some(12));
    assert_eq!(layer_of("blk.x.attn_q.weight"), None);
    assert_eq!(layer_of("output.weight"), None);
}

#[test]
fn auto_split_balances_bytes() {
    // 8 equal layers, 2 stages: half each.
    assert_eq!(auto_split(&[10; 8], 0, 2), vec![4]);
    // The last stage also holds 20 bytes of head: it gets fewer layers.
    assert_eq!(auto_split(&[10; 8], 20, 2), vec![5]);
    assert_eq!(auto_split(&[10; 8], 0, 4), vec![2, 4, 6]);
    assert!(auto_split(&[10; 8], 0, 1).is_empty());
    // More stages than layers fit: one layer each where possible.
    assert_eq!(auto_split(&[1, 1, 100], 0, 3), vec![1, 2]);
}

#[test]
fn last_extra_layers_lighten_the_last_stage() {
    // 0: the split by bytes, whatever the layers' bytes.
    assert_eq!(auto_cuts(&[10; 48], 0, 4, 0.0), vec![12, 24, 36]);
    assert_eq!(auto_cuts(&[10; 8], 20, 2, 0.0), auto_split(&[10; 8], 20, 2));
    // 48 layers over 4 stages. One layer's worth more on the last: 13 either way, ties keep
    // the even split.
    assert_eq!(auto_cuts(&[10; 48], 0, 4, 1.0), vec![12, 24, 36]);
    // Two layers' worth: 13/12/12/11.
    assert_eq!(auto_cuts(&[10; 48], 0, 4, 2.0), vec![13, 25, 37]);
    // Three: 13/13/12/10.
    assert_eq!(auto_cuts(&[10; 48], 0, 4, 3.0), vec![13, 26, 38]);
    // One stage: no cuts.
    assert!(auto_cuts(&[10; 48], 0, 1, 2.0).is_empty());
}

#[test]
fn split_ranges() {
    assert_eq!(ranges(8, &[]).unwrap(), vec![0..8]);
    assert_eq!(ranges(8, &[3, 5]).unwrap(), vec![0..3, 3..5, 5..8]);
    assert!(ranges(8, &[0]).is_err());
    assert!(ranges(8, &[5, 5]).is_err());
    assert!(ranges(8, &[8]).is_err());
}

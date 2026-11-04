use super::*;

#[test]
fn a_first_pass_in_order_puts_each_row_in_place() {
    assert_eq!(first_pass_rows(&[0, 1, 2]), Some(vec![0, 1, 2]));
    assert_eq!(first_pass_rows(&[]), Some(vec![]));
}

#[test]
fn a_sorted_first_pass_maps_back_to_the_rounds_order() {
    // The first pass ran the round's rows 2, 0, 1 (a take-in ordered by rows): row 0 of the
    // round is the pass's second output row, row 1 its third, row 2 its first.
    assert_eq!(first_pass_rows(&[2, 0, 1]), Some(vec![1, 2, 0]));
}

#[test]
fn an_order_that_is_not_one_of_the_rows_is_refused() {
    assert_eq!(first_pass_rows(&[0, 0]), None);
    assert_eq!(first_pass_rows(&[0, 2]), None);
    assert_eq!(first_pass_rows(&[1]), None);
}

#[test]
fn the_pass_after_the_first_follows_the_rounds_rows() {
    let blocks = [0u32, 1];
    let rows = [
        Row {
            at: 10,
            blocks: &blocks,
            ring: 0,
        },
        Row {
            at: 20,
            blocks: &blocks,
            ring: 0,
        },
    ];
    let g = next_pass(&rows, &[4, 4], [(0, 0), (1, 0)].into_iter()).expect("both go on");
    let at: Vec<usize> = g.seqs.iter().map(|s| s.n_past).collect();
    assert_eq!(at, [11, 21]);
    // A row at its cap stops the prediction.
    assert!(next_pass(&rows, &[1, 4], [(0, 0), (1, 0)].into_iter()).is_none());
}

use super::*;
use crate::ggml::{Backend, ComputeAllocator, types};

#[test]
fn routed_ids_give_padding_rows_their_first_rows_choices() {
    let mut be = Backend::cpu().unwrap();
    let ctx = Context::new(8, 16);
    // Two sequences of three rows, two experts a row; the router's view of a wider row (as
    // argsort leaves it), so the result is also the contiguous copy `cont` made.
    let all = ctx.new_tensor(types::I32, &[3, 6]).as_input();
    let sel = ctx.view_2d(all, [2, 6], all.nb()[1], 0);
    let rows = ctx.new_tensor(types::I32, &[6]).as_input();
    let out = routed_ids(&ctx, sel, rows);
    assert_eq!(out.ne()[..2], [2, 6]);
    assert!(out.is_contiguous());
    let mut g = ctx.new_graph(16);
    g.expand(out);
    ctx.alloc_graph(&g, ComputeAllocator::new(be.buffer_type()))
        .unwrap();
    all.set_i32(&[
        10, 11, -1, 20, 21, -1, 30, 31, -1, 40, 41, -1, 50, 51, -1, 60, 61, -1,
    ]);
    // The first sequence has two real rows, the second one.
    rows.set_i32(&[0, 1, 0, 3, 3, 3]);
    be.compute(&g).unwrap();
    assert_eq!(
        out.to_i32(),
        [10, 11, 20, 21, 10, 11, 40, 41, 40, 41, 40, 41]
    );
}

#[test]
fn padding_rows_take_their_first_row() {
    let fed = vec![vec![5, 6, 7, 5, 5], vec![9, 5, 9, 9, 9]];
    let _g = register(&fed, [3, 1]);
    assert_eq!(real_rows(&fed[0]), Some(3));
    assert_eq!(real_rows(&fed[1]), Some(1));
    let m = map(fed.iter().map(|f| f.as_slice()), 5);
    assert_eq!(m, vec![0, 1, 2, 0, 0, 5, 5, 5, 5, 5]);
}

#[test]
fn unregistered_rows_keep_their_own() {
    let fed = vec![vec![1, 2, 3], vec![4, 5, 6]];
    // Copies and parts of registered rows are not them.
    let copy = fed.clone();
    let _g = register(&fed, [1, 2]);
    assert_eq!(real_rows(&copy[0]), None);
    assert_eq!(real_rows(&fed[0][..2]), None);
    assert_eq!(
        map(copy.iter().map(|f| f.as_slice()), 3),
        vec![0, 1, 2, 3, 4, 5]
    );
}

#[test]
fn guard_drop_forgets_and_full_rows_are_not_kept() {
    let fed = vec![vec![1, 2, 3, 4], vec![7, 8, 9, 10]];
    {
        // A sequence with every row real (or none, which is never fed) has no padding.
        let g = register(&fed, [4, 0]);
        assert!(g.keys.is_empty());
        let _g = register(&fed, [2, 4]);
        assert_eq!(real_rows(&fed[0]), Some(2));
        assert_eq!(real_rows(&fed[1]), None);
    }
    assert_eq!(real_rows(&fed[0]), None);
    assert_eq!(
        map(fed.iter().map(|f| f.as_slice()), 4),
        vec![0, 1, 2, 3, 4, 5, 6, 7]
    );
}

#[test]
fn tiered_remap_is_off_by_default() {
    // The default keeps tiered experts' lookups as the router chose (byte-identical output).
    assert!(!tiered());
}

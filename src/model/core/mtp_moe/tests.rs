use super::*;

#[test]
fn head_block_is_past_the_target_layers() {
    assert!(!is_head(0, 48));
    assert!(!is_head(47, 48));
    assert!(is_head(48, 48));
}

#[test]
fn floor_off_keeps_the_target_rule() {
    for n in [1, 8, 16, 64] {
        assert_eq!(choose(MoeKernel::Auto, None, n, 0, true), None);
    }
    let big = Some(MoeKernel::Auto);
    assert_eq!(choose(MoeKernel::Auto, big, 64, 0, true), big);
}

#[test]
fn floor_sends_big_enough_passes_to_the_fused_op() {
    assert_eq!(choose(MoeKernel::Auto, None, 8, 9, true), None);
    assert_eq!(
        choose(MoeKernel::Auto, None, 9, 9, true),
        Some(MoeKernel::Auto)
    );
    assert_eq!(
        choose(MoeKernel::Auto, None, 16, 9, true),
        Some(MoeKernel::Auto)
    );
    // Not without CUDA, and not over a forced graph path.
    assert_eq!(choose(MoeKernel::Auto, None, 16, 9, false), None);
    assert_eq!(choose(MoeKernel::Graph, None, 16, 9, true), None);
}

#[test]
fn rows_round_trip() {
    let before = rows();
    set_rows(9);
    assert_eq!(rows(), 9);
    set_rows(before as usize);
}

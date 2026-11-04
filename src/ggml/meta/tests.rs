use super::*;

#[test]
fn shares_split_evenly_in_whole_granules() {
    assert_eq!(shares(8192, 2, 512), vec![4096, 4096]);
    assert_eq!(shares(4096, 2, 256), vec![2048, 2048]);
    // 5 heads of 256 over 2 devices: whole heads, the last device takes the rest.
    assert_eq!(shares(1280, 2, 256), vec![512, 768]);
    assert_eq!(shares(10, 3, 1), vec![3, 3, 4]);
    assert_eq!(shares(7, 1, 1), vec![7]);
}

#[test]
fn lcm_of_heads_and_blocks() {
    assert_eq!(lcm(256, 32), 256);
    assert_eq!(lcm(96, 256), 768);
    assert_eq!(lcm(1, 32), 32);
}

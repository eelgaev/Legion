use super::*;

#[test]
fn no_taper_cuts_full_micro_chunks_and_the_rest() {
    assert_eq!(micro_sizes(10_000, 3072, 0), vec![3072, 3072, 3072, 784]);
    assert_eq!(micro_sizes(6144, 3072, 0), vec![3072, 3072]);
    assert_eq!(micro_sizes(100, 3072, 0), vec![100]);
    // A taper not under the micro-chunk is none.
    assert_eq!(micro_sizes(10_000, 3072, 3072), vec![3072, 3072, 3072, 784]);
}

#[test]
fn a_taper_ramps_down_at_the_end() {
    let v = micro_sizes(122_585, 3072, 1024);
    assert_eq!(v.iter().sum::<usize>(), 122_585);
    assert_eq!(&v[v.len() - 2..], &[2048, 1024]);
    let mid = &v[..v.len() - 2];
    assert!(mid.iter().all(|&m| m <= 3072 && m + 1 >= mid[0]));
    assert!(mid.windows(2).all(|w| w[0] >= w[1]));
    // The ramp halves from under the micro-chunk.
    let v = micro_sizes(50_000, 4096, 512);
    assert_eq!(&v[v.len() - 3..], &[2048, 1024, 512]);
    assert!(v[0] <= 4096 && v[0] + 1 >= v[v.len() - 4]);
}

#[test]
fn a_short_step_is_cut_without_a_taper() {
    // The ramp (3072) and one full micro-chunk need 6144 tokens.
    assert_eq!(micro_sizes(6000, 3072, 1024), vec![3072, 2928]);
    assert_eq!(micro_sizes(6144, 3072, 1024), vec![3072, 2048, 1024]);
}

use super::*;

/// KDA pools (GLM) keep more commit graphs than the others, which keep eight (#725).
#[test]
fn kda_pools_keep_more_commit_graphs() {
    assert_eq!(commit_kept(false), 8);
    assert!(commit_kept(true) >= 30, "C=4's sets of runs fit");
}

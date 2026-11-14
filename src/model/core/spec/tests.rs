use super::*;

/// A rank that defers commits of at least eight sequences runs a narrower one in the commit
/// (#1176); without a floor every commit is deferred.
#[test]
fn wide_commits_are_those_of_the_floor_or_more() {
    assert!(!wide_commit(8, 1));
    assert!(!wide_commit(8, 7));
    assert!(wide_commit(8, 8));
    assert!(wide_commit(8, 16));
    assert!(wide_commit(0, 1));
    assert!(wide_commit(1, 1));
}

/// KDA pools (GLM) keep more commit graphs than the others, which keep eight (#725).
#[test]
fn kda_pools_keep_more_commit_graphs() {
    assert_eq!(commit_kept(false), 8);
    assert!(commit_kept(true) >= 30, "C=4's sets of runs fit");
}
